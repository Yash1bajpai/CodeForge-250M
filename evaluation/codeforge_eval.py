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

tok_secret = os.environ.get("HF_TOKEN")
if not tok_secret:
    from kaggle_secrets import UserSecretsClient
    tok_secret = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = tok_secret

import torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import PreTrainedTokenizerFast

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)

# --- model + tokenizer (repo architecture file is fetched from GitHub master) ---
if not os.path.exists("/tmp/cf_arch/models/architecture.py"):
    os.makedirs("/tmp/cf_arch/models", exist_ok=True)
    sh("curl -sSL https://raw.githubusercontent.com/Yash1bajpai/CodeForge-250M/master/models/architecture.py -o /tmp/cf_arch/models/architecture.py")
    open("/tmp/cf_arch/models/__init__.py", "a").close()
sys.path.insert(0, "/tmp/cf_arch")
from models.architecture import CodeForgeModel, apply_rotary_pos_emb

CFG = dict(vocab_size=32000, hidden_size=1024, intermediate_size=2816, num_hidden_layers=16,
           num_attention_heads=16, num_key_value_heads=4, max_position_embeddings=2048,
           rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False)

tdir = snapshot_download(HF_REPO, revision=REV, allow_patterns=["data/tokenizer/*"], token=tok_secret)
tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(tdir, "data/tokenizer"))
EOS = tok.eos_token_id
_SPECIALS = ["<|endoftext|>","<|unk|>","<|pad|>","<|fim_prefix|>","<|fim_middle|>","<|fim_suffix|>","<|tool_call|>","<|tool_result|>","<|thinking|>","<|json_start|>","<|json_end|>"]
BAN_IDS = sorted({tok.convert_tokens_to_ids(t) for t in _SPECIALS} - {EOS, None})
print("special ids", {t: tok.convert_tokens_to_ids(t) for t in _SPECIALS}, flush=True)
print("tokenizer", len(tok), "eos", EOS, "banned special ids", BAN_IDS, "EOS_PREFIX", EOS_PREFIX, "BAN", BAN, flush=True)
ck_path = hf_hub_download(HF_REPO, CKPT_FILE, revision=REV, token=tok_secret)
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
print("ckpt keys", [k for k in ck.keys()][:20], "step", ck.get("step") or ck.get("global_step") or ck.get("optimizer_step_count"), flush=True)
sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
model = CodeForgeModel(CFG)
missing = model.load_state_dict(sd, strict=True)
print("load_state_dict", missing, "params", model.get_parameter_count(), flush=True)
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
results = {"checkpoint_repo": HF_REPO, "revision": REV, "decoding": "greedy", "max_new_tokens": MAX_NEW,
           "device": dev, "kv_cache_check": [d1, d2]}
HE_STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", "\nassert "]
MB_STOPS = ["\ndef ", "\nclass ", "\nassert", '\n"""', "\nprint(", "\nif __name__", "\n#"]

def evaluate(name, items, make_prompt, stops, make_test):
    t0 = time.time(); recs = []
    with ThreadPoolExecutor(4) as ex:
        futs = []
        for n, it in enumerate(items):
            prompt = make_prompt(it)
            comp = complete(prompt, stops)
            futs.append((it, prompt, comp, ex.submit(run_program, make_test(it, prompt, comp))))
            if n % 20 == 0:
                print(f"[{name}] generated {n+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        for it, prompt, comp, f in futs:
            ok, err = f.result()
            recs.append({"task_id": str(it.get("task_id")), "passed": ok, "completion": comp, "err": err})
    for r in recs[:8]:
        print("DIAGREC", name, r["task_id"], r["passed"], repr(r["completion"][:150]), repr(r["err"][-120:]), flush=True)
    k = sum(r["passed"] for r in recs)
    print(f"RESULT {name}: pass@1 (greedy) = {k}/{len(recs)} = {100*k/len(recs):.2f}%  [{time.time()-t0:.0f}s]", flush=True)
    results[name] = {"passed": k, "total": len(recs), "pass_at_1": k / len(recs)}
    json.dump(recs, open(f"{OUT}/{name}_samples.json", "w"), indent=1)
    json.dump(results, open(f"{OUT}/eval_results.json", "w"), indent=1)

he = list(load_dataset("openai/openai_humaneval", split="test"))
if LIMIT: he = he[:LIMIT]
evaluate("humaneval", he, lambda it: it["prompt"].rstrip("\n") if TOKEN_HEAL else it["prompt"], HE_STOPS,
         lambda it, p, c: p + c + "\n\n" + it["test"] + f"\n\ncheck({it['entry_point']})\n")

mb = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
if LIMIT: mb = mb[:LIMIT]
def mb_fname(it):
    m = re.search(r"assert\s+(?:not\s+)?\(?\s*(?:set\(|sorted\(|list\(|tuple\(|math\.isclose\(|abs\()?\s*([A-Za-z_]\w*)\s*\(", it["test_list"][0])
    return m.group(1) if m else "solution"
def mb_prompt(it):
    if MB_STYLE == "def":
        return '"""\n' + it["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(it["test_list"]) + '\n"""\ndef'
    if MB_STYLE == "named":
        return '"""\n' + it["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(it["test_list"]) + '\n"""\ndef ' + mb_fname(it) + "("
    if MB_STYLE == "comment":
        return "# " + it["text"] + "\n# Tests:\n" + "\n".join("# " + t for t in it["test_list"]) + "\ndef "
    return '"""\n' + it["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(it["test_list"]) + '\n"""\n'
evaluate("mbpp", mb, mb_prompt, MB_STOPS,
         lambda it, p, c: (it.get("test_setup_code") or "") + "\n" + (p[p.rfind("\ndef ")+1:] if MB_STYLE == "named" else ("def" if p.endswith("def") else "")) + c + "\n\n" + "\n".join(it["test_list"]) + "\n")

print("FINAL", json.dumps(results), flush=True)
try:
    assert not os.environ.get("CF_NOUPLOAD")
    from huggingface_hub import HfApi
    api = HfApi(token=tok_secret)
    for f in ("eval_results.json", "humaneval_samples.json", "mbpp_samples.json"):
        api.upload_file(path_or_fileobj=f"{OUT}/{f}", path_in_repo=f"eval/{EVAL_TAG}/{f}", repo_id=HF_REPO, commit_message="Stage A eval results")
    print("uploaded eval results to HF", flush=True)
except Exception as e:
    print("HF upload failed", e, flush=True)
