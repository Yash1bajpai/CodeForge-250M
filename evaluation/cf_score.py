"""CodeForge-250M evaluation: KV-cache greedy inference + HumanEval/MBPP pass@1.
Run on Kaggle (GPU or CPU). Pinned to a specific HF revision of the checkpoint.
"""
import os, sys, json, time, subprocess, tempfile, textwrap, re, shutil
from concurrent.futures import ThreadPoolExecutor

HF_REPO = "Yash1bajpai/CodeForge-250M-rotation"
REV = os.environ.get("CF_REV", "2542b76813c0323db7325a0dfed56431964edbf5")
LIMIT = int(os.environ.get("CF_LIMIT", "0"))      # 0 = all problems
EOS_PREFIX = os.environ.get("CF_EOS_PREFIX","1")=="1"
BAN = os.environ.get("CF_BAN","1")=="1"
TOKEN_HEAL = os.environ.get("CF_HEAL","1")=="1"
MIN_NEW = int(os.environ.get("CF_MIN_NEW","0"))
MB_STYLE = os.environ.get("CF_MB_STYLE","doc")
MAX_NEW = int(os.environ.get("CF_MAX_NEW", "320"))
CKPT_FILE = os.environ.get("CF_CKPT_FILE", "latest_checkpoint.pt")   # e.g. stageA2/latest_checkpoint.pt
EVAL_TAG = os.environ.get("CF_EVAL_TAG", f"stage_a_{REV[:8]}")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/cf_out"
os.makedirs(OUT, exist_ok=True)

def sh(c):
    print("$", c, flush=True); subprocess.run(c, shell=True, check=False)

try:
    import huggingface_hub, datasets  # noqa
except Exception:
    sh("pip -q install huggingface_hub datasets")

tok_secret = os.environ.get("HF_TOKEN","")

import torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import PreTrainedTokenizerFast

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)

# --- model + tokenizer (repo architecture file is fetched from GitHub master) ---
PROJ = os.environ["CF_PROJ"]
sys.path.insert(0, PROJ)
from models.architecture import CodeForgeModel, apply_rotary_pos_emb

CFG = json.loads(os.environ["CF_CFG_JSON"]) if os.environ.get("CF_CFG_JSON") else dict(vocab_size=32000, hidden_size=1024, intermediate_size=2816, num_hidden_layers=16,
           num_attention_heads=16, num_key_value_heads=4, max_position_embeddings=2048,
           rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False)

tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJ, "data/tokenizer"))
EOS = tok.eos_token_id
_SPECIALS = ["<|endoftext|>","<|unk|>","<|pad|>","<|fim_prefix|>","<|fim_middle|>","<|fim_suffix|>","<|tool_call|>","<|tool_result|>","<|thinking|>","<|json_start|>","<|json_end|>"]
BAN_IDS = sorted({tok.convert_tokens_to_ids(t) for t in _SPECIALS} - {EOS, None})
print("special ids", {t: tok.convert_tokens_to_ids(t) for t in _SPECIALS}, flush=True)
print("tokenizer", len(tok), "eos", EOS, "banned special ids", BAN_IDS, "EOS_PREFIX", EOS_PREFIX, "BAN", BAN, flush=True)
ck_path = os.environ["CF_LOCAL_CKPT"]
ck = torch.load(ck_path, map_location="cpu", weights_only=True)
print("ckpt keys", [k for k in ck.keys()][:20], "step", ck.get("step") or ck.get("global_step") or ck.get("optimizer_step_count"), flush=True)
sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
model = CodeForgeModel(CFG)
missing = model.load_state_dict(sd, strict=True)
print("load_state_dict", missing, "params", model.get_parameter_count(), flush=True)
ck_step = ck.get("step")
del ck, sd
model = model.to(dev).eval()

@torch.no_grad()
def fwd(ids, cache, pos0):
    h = model.embed_tokens(ids)
    T = ids.shape[1]
    for i, layer in enumerate(model.layers):
        a = layer.self_attn
        x = layer.input_layernorm(h)
        B = x.shape[0]
        q = a.q_proj(x).view(B, T, a.num_heads, a.head_dim).transpose(1, 2)
        k = a.k_proj(x).view(B, T, a.num_kv_heads, a.head_dim).transpose(1, 2)
        v = a.v_proj(x).view(B, T, a.num_kv_heads, a.head_dim).transpose(1, 2)
        cos, sin = a.rotary_emb(v, seq_len=pos0 + T)
        q, k = apply_rotary_pos_emb(q, k, cos[pos0:], sin[pos0:])
        if cache[i] is not None:
            k = torch.cat([cache[i][0], k], 2); v = torch.cat([cache[i][1], v], 2)
        cache[i] = (k, v)
        g = a.num_key_value_groups
        kk, vv = (k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)) if g > 1 else (k, v)
        o = F.scaled_dot_product_attention(q, kk, vv, is_causal=(T > 1))
        h = h + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.norm(h[:, -1:]))[:, 0]

@torch.no_grad()
def generate_ids(prompt_ids, max_new, stops=(), temperature=0.0, plain=True):
    """Greedy (temperature 0) or sampled decode with KV cache. Stops on EOS or any stop string."""
    if plain and EOS_PREFIX:
        prompt_ids = [EOS] + list(prompt_ids)
    prompt_ids = prompt_ids[-(CFG["max_position_embeddings"] - max_new):]
    cache = [None] * len(model.layers)
    ids = torch.tensor([prompt_ids], device=dev)
    logits = fwd(ids, cache, 0)
    pos = ids.shape[1]; out = []
    for _ in range(max_new):
        if plain and BAN:
            logits[:, BAN_IDS] = float("-inf")
        if plain and len(out) < MIN_NEW:
            logits[:, EOS] = float("-inf")
        if temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(logits[0] / temperature, -1), 1))
        else:
            nxt = int(logits[0].argmax())
        if nxt == EOS:
            break
        out.append(nxt)
        if stops and len(out) % 4 == 0:
            txt = tok.decode(out)
            if any(s in txt for s in stops):
                break
        logits = fwd(torch.tensor([[nxt]], device=dev), cache, pos); pos += 1
    return out

def complete(prompt, stops, max_new=MAX_NEW):
    out = tok.decode(generate_ids(tok.encode(prompt), max_new, stops))
    cut = len(out)
    for s in stops:
        j = out.find(s)
        if j != -1: cut = min(cut, j)
    return out[:cut]

# --- correctness check of the cached decoder against the repo's own forward ---
ids = torch.tensor([tok.encode("def fibonacci(n):\n    if n < 2:\n        return n\n    return")], device=dev)
with torch.no_grad():
    ref, _ = model(ids)
    c = [None] * len(model.layers)
    l1 = fwd(ids, c, 0)
    nxt = l1.argmax(-1, keepdim=True)
    l2 = fwd(nxt, c, ids.shape[1])
    ref2, _ = model(torch.cat([ids, nxt], 1))
d1 = (l1 - ref[:, -1]).abs().max().item(); d2 = (l2 - ref2[:, -1]).abs().max().item()
print(f"KV-cache check: prefill maxdiff={d1:.2e} step maxdiff={d2:.2e}", flush=True)
assert d1 < 1e-2 and d2 < 1e-2, "cached decoder disagrees with model forward"

print("SAMPLE:", repr(complete("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", ["\ndef ", "\nclass "], 80)), flush=True)
fim = "<|fim_prefix|>def add(a, b):\n    return <|fim_suffix|>\n\nprint(add(1, 2))<|fim_middle|>"
print("FIM SAMPLE:", repr(tok.decode(generate_ids(tok.encode(fim), 20, plain=False))), flush=True)


if os.environ.get("CF_DIAG","1")=="1":
    from datasets import load_dataset as _ld
    _he=list(_ld("openai/openai_humaneval", split="test"))[:6]
    for _it in _he:
        _ids=tok.encode(_it["prompt"])
        _raw=tok.decode(generate_ids(_ids, 120))
        print("DIAG prompt_tail=%r last_tokens=%r\n   RAW=%r" % (_it["prompt"][-60:], [tok.decode([t]) for t in _ids[-4:]], _raw), flush=True)

# --- sandboxed execution ---
def run_program(src, timeout=10):
    d = tempfile.mkdtemp(prefix="cfx_")
    try:
        p = os.path.join(d, "t.py"); open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, text=True, timeout=timeout,
                               env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
            return r.returncode == 0, (r.stderr or "")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
    finally:
        shutil.rmtree(d, ignore_errors=True)

from datasets import load_dataset
results = {"label": os.environ.get("CF_EVAL_TAG",""), "checkpoint_file": ck_path, "ckpt_step": ck_step, "decoding": "greedy", "max_new_tokens": MAX_NEW,
           "device": dev, "kv_cache_check": [d1, d2]}
HE_STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", "\nassert "]
MB_STOPS = ["\ndef ", "\nclass ", "\nassert", '\n"""', "\nprint(", "\nif __name__", "\n#"]

EXEC_MODE = os.environ.get("CF_EXEC_MODE", "export")
def evaluate(name, items, make_prompt, stops, make_test):
    t0 = time.time(); recs = []
    if EXEC_MODE == "export":
        import ast, hashlib
        jl = f"{OUT}/{EVAL_TAG}_{name}_programs.jsonl"; n = 0; pr = 0
        with open(jl, "w") as f:
            for it in items:
                if time.time() > float(os.environ.get("CF_SCORE_DEADLINE", "inf")):
                    print(f"INCOMPLETE {name}: ceiling guard stopped generation at {n}/{len(items)}", flush=True)
                    results.setdefault("incomplete", []).append(f"{name} partial {n}/{len(items)}"); break
                prompt = make_prompt(it); comp = complete(prompt, stops); prog = make_test(it, prompt, comp)
                try: ast.parse(prog); ok = True
                except Exception: ok = False
                pr += ok; n += 1
                f.write(json.dumps({"eval": name, "task_id": str(it.get("task_id", it.get("id"))), "prompt": prompt, "completion": comp, "program": prog, "parses": ok}) + "\n")
                if n % 20 == 0: print(f"[{name}] exported {n}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        raw = open(jl, "rb").read()
        json.dump({"sha256": hashlib.sha256(raw).hexdigest(), "count": n, "checkpoint": os.environ.get("CF_EVAL_TAG",""), "eval": name, "decoding": "greedy", "max_new_tokens": MAX_NEW, "eos_prefix": EOS_PREFIX, "ban_specials": BAN}, open(jl.replace(".jsonl", ".meta.json"), "w"), indent=1)
        print(f"EXPORTED {name}: {n}/{len(items)} programs, {pr} parse (ast) [{time.time()-t0:.0f}s]", flush=True)
        results[name] = {"exported": n, "of": len(items), "parsed": pr, "executed": False}
        return
    with ThreadPoolExecutor(4) as ex:
        futs = []
        for n, it in enumerate(items):
            if time.time() > float(os.environ.get("CF_SCORE_DEADLINE", "inf")):
                print(f"INCOMPLETE {name}: ceiling guard stopped generation at {n}/{len(items)}", flush=True); results.setdefault("incomplete", []).append(f"{name} partial {n}/{len(items)}"); break
            prompt = make_prompt(it)
            comp = complete(prompt, stops)
            futs.append((it, prompt, comp, ex.submit(run_program, make_test(it, prompt, comp))))
            if n % 20 == 0:
                print(f"[{name}] generated {n+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        for it, prompt, comp, f in futs:
            ok, err = f.result()
            recs.append({"task_id": str(it.get("task_id")), "passed": ok, "completion": comp, "err": err, "parses": ("SyntaxError" not in err and "IndentationError" not in err)})
    for r in recs[:8]:
        print("DIAGREC", name, r["task_id"], r["passed"], repr(r["completion"][:150]), repr(r["err"][-120:]), flush=True)
    k = sum(r["passed"] for r in recs)
    print(f"RESULT {name}: pass@1 (greedy) = {k}/{len(recs)} = {100*k/len(recs):.2f}%  [{time.time()-t0:.0f}s]", flush=True)
    pr = sum(r["parses"] for r in recs)
    print(f"PARSE {name}: {pr}/{len(recs)} programs without SyntaxError/IndentationError", flush=True)
    results[name] = {"passed": k, "total": len(recs), "pass_at_1": k / len(recs), "parsed": pr}
    json.dump(recs, open(f"{OUT}/{name}_samples.json", "w"), indent=1)
    json.dump(results, open(f"{OUT}/eval_results.json", "w"), indent=1)


import ast as _ast
DEADLINE = float(os.environ.get("CF_SCORE_DEADLINE", "inf"))
def time_left(): return DEADLINE - time.time()
def run_stage(label, fn):
    if time_left() < 60:
        print(f"INCOMPLETE {label}: skipped, ceiling guard", flush=True); results.setdefault("incomplete", []).append(label); return
    try: fn()
    except Exception as e:
        import traceback; traceback.print_exc(); results.setdefault("errors", {})[label] = repr(e)[:300]
    json.dump(results, open(f"{OUT}/eval_results.json", "w"), indent=1)

def stage_val():
    from training.train import LazyShardDataset, evaluate_val_loss
    from torch.utils.data import DataLoader
    vs = sorted(__import__("glob").glob(os.path.join(os.environ["CF_VAL_DIR"], "shard_*.pt")))
    vd = LazyShardDataset(os.environ["CF_VAL_DIR"], seq_length=CFG["max_position_embeddings"], shard_files=vs, vocab_size=CFG["vocab_size"])
    vl = DataLoader(vd, batch_size=int(os.environ.get("CF_VAL_BS","8")), shuffle=False, num_workers=1, pin_memory=(dev == "cuda"), drop_last=True)
    nb = int(os.environ.get("CF_VAL_BATCHES", "27")); bs = []
    for i, b in enumerate(vl):
        bs.append(b)
        if i + 1 >= nb: break
    dt = torch.float16 if (dev == "cuda" and torch.cuda.get_device_capability(0)[0] < 8) else (torch.bfloat16 if dev == "cuda" else torch.float32)
    m = model; m.eval()
    v = evaluate_val_loss(m, bs, dev, dt)
    print(f"VAL_LOSS val_ready {len(bs)} batches x 8 = {len(bs)*8} of {len(vd)} rows dtype={dt}: {v:.6f}", flush=True)
    results["val_loss"] = {"value": v, "rows_scored": len(bs) * 8, "rows_total": len(vd), "dtype": str(dt), "batches": len(bs)}

@torch.no_grad()
def ref_loss(pairs):
    """mean per-token CE on target text given context text, same fp16/fp32 path as eval (no cache)."""
    tot = 0.0; n = 0
    for ctx, tgt in pairs:
        ci = tok.encode(ctx) if ctx else []
        ti = tok.encode(tgt)
        ids = ([EOS] + ci + ti)[: CFG["max_position_embeddings"]]
        x = torch.tensor([ids[:-1]], device=dev); y = torch.tensor([ids[1:]], device=dev)
        y[:, : 1 + len(ci) - 1] = -100
        if (y != -100).sum() == 0: continue
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            out = model(x)
        lg = out[0] if isinstance(out, tuple) else out
        l = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1), ignore_index=-100, reduction="sum")
        tot += float(l); n += int((y != -100).sum())
    return tot / max(n, 1), n

def stage_fresh():
    tasks = json.load(open(os.environ["CF_FRESH_TASKS"])); refs = json.load(open(os.environ["CF_FRESH_REFS"]))
    def t24_prompt(it):
        if it["id"].startswith("b"):
            return it["prompt"] + "\n\n# Corrected version:\ndef " + it["entry"] + "("
        return it["prompt"]
    def t24_test(it, p, c):
        lines = []
        for args, exp in it["tests"]:
            if isinstance(exp, dict) and exp.get("__int_keys__"):
                exp = {int(k): v for k, v in exp["value"].items()}
            lines.append(f"assert {it['entry']}(*{args!r}) == {exp!r}")
        return p + c + "\n\n" + "\n".join(lines) + "\n"
    evaluate("tasks24", [dict(task_id=t["id"], **t) for t in tasks], t24_prompt, HE_STOPS, t24_test)
    l, n = ref_loss([("", refs[t["id"]]) for t in tasks])
    print(f"REFLOSS fresh24 reference solutions: {l:.4f}/token over {n} tokens", flush=True)
    results["refloss_fresh24"] = {"loss_per_token": l, "tokens": n}

def stage_refloss_public():
    he = list(load_dataset("openai/openai_humaneval", split="test"))
    if LIMIT: he = he[:LIMIT]
    l, n = ref_loss([(it["prompt"].rstrip("\n"), it["canonical_solution"]) for it in he])
    print(f"REFLOSS humaneval canonical (all {len(he)}): {l:.4f}/token over {n} tokens", flush=True)
    results["refloss_humaneval_all"] = {"loss_per_token": l, "tokens": n}
    mbx = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    if LIMIT: mbx = mbx[:LIMIT]
    l, n = ref_loss([(mb_prompt(it)[:-1] if False else "", it["code"]) for it in mbx])
    print(f"REFLOSS mbpp gold code (no prompt, {len(mbx)}): {l:.4f}/token over {n} tokens", flush=True)
    results["refloss_mbpp_gold"] = {"loss_per_token": l, "tokens": n}

def stage_he():
  he = list(load_dataset("openai/openai_humaneval", split="test"))
  if LIMIT: he = he[:LIMIT]
  evaluate("humaneval", he, lambda it: it["prompt"].rstrip("\n") if TOKEN_HEAL else it["prompt"], HE_STOPS,
         lambda it, p, c: p + c + "\n\n" + it["test"] + f"\n\ncheck({it['entry_point']})\n")


def mb_fname(it):
    m = re.search(r"assert\s+(?:not\s+)?\(?\s*(?:set\(|sorted\(|list\(|tuple\(|math\.isclose\(|abs\()?\s*([A-Za-z_]\w*)\s*\(", it["test_list"][0])
    return m.group(1) if m else "solution"
def mb_prompt(it):
    if MB_STYLE == "def":
        return '"""\n' + it["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(it["test_list"]) + '\n"""\ndef'
    if MB_STYLE in ("named", "named2"):
        shown = it["test_list"][:2] if MB_STYLE == "named2" else it["test_list"]
        return '"""\n' + it["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(shown) + '\n"""\ndef ' + mb_fname(it) + "("
    if MB_STYLE == "comment":
        return "# " + it["text"] + "\n# Tests:\n" + "\n".join("# " + t for t in it["test_list"]) + "\ndef "
    return '"""\n' + it["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(it["test_list"]) + '\n"""\n'
TWO_SHOT = '''"""
Write a python function to add two numbers.
Your code should pass these tests:

assert add(1, 2) == 3
assert add(-1, 1) == 0
assert add(0, 0) == 0
"""
def add(a, b):
    return a + b


"""
Write a python function to double a number.
Your code should pass these tests:

assert double(3) == 6
assert double(-2) == -4
assert double(0) == 0
"""
def double(n):
    return n * 2


'''
def doc_body(it):
    return it["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(it["test_list"])
def shot2_prompt(it):
    return TWO_SHOT + '"""\n' + doc_body(it) + '\n"""\n'
def shot2_test(it, p, c):
    setup = (it.get("test_setup_code") or "")
    return (setup + "\n" if setup else "") + c + "\n\n" + "\n".join(it["test_list"]) + "\n"
def stage_mb():
    mb = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    if LIMIT: mb = mb[:LIMIT]
    evaluate("mbpp_2shot", mb, shot2_prompt, MB_STOPS, shot2_test)


for lab, fn in [("val_loss", stage_val), ("fresh24", stage_fresh), ("refloss_public", stage_refloss_public), ("humaneval", stage_he), ("mbpp_two_example", stage_mb)]:
    run_stage(lab, fn)
print("FINAL", json.dumps(results), flush=True)
