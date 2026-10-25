# CodeForge step-420 evaluation (private). Same original protocol as the gate step-100 eval:
# trusted harness cf_score.py (byte-verified), EXEC_MODE=export (generation only, NO execution
# of generated programs anywhere in this session). Internet ON for public HF eval datasets only
# (openai/openai_humaneval, google-research-datasets/mbpp). No HF token, no Kaggle secrets.
import os,sys,time,json,subprocess,hashlib,glob,shutil,re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
SESSION_START=time.time()-300          # conservative: session boot precedes cell start
def el(): return (time.time()-SESSION_START)/60
def P(*a): print(f"[{el():6.1f}m]",*a,flush=True)
PHASE="eval420"
# Session budget: eval generation measured ~10-30 min on one T4; cap the session at 150 min.
SESSION_LIMIT_MIN=150; MARGIN_MIN=10
HARD_END=SESSION_START+(SESSION_LIMIT_MIN-MARGIN_MIN)*60
W="/kaggle/working"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<24),b""): h.update(b)
    return h.hexdigest(), os.path.getsize(p)
P("CELL VERSION eval420; PHASE=",PHASE,"CF_EXEC_MODE=",os.environ.get("CF_EXEC_MODE"))
P("SESSION_START_EPOCH",SESSION_START,"HARD_END",HARD_END)
import torch
P("torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.device_count(),
  torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available(),"NO GPU"
# --- mounts (read-only): gate-v7 dataset (proj source + hf tokenizer + harness + fresh tasks),
#     cont-step420 dataset (step-420 checkpoint), cf-packsnap (val shards) ---
GATE="/kaggle/input/datasets/yashbajpai2027/cf-gate-v7-gate-step100-output"
CONT="/kaggle/input/datasets/yashbajpai2027/cf-cont-step420-output"
PACK="/kaggle/input/datasets/yashbajpai2027/cf-packsnap"
for d in (GATE,CONT,PACK): assert os.path.isdir(d), f"missing mount {d}"
P("mounts ok")
# --- proj source -> working (tar minus .git/checkpoints), tokenizer from gate hf/ ---
PR=f"{W}/proj"
if not os.path.exists(PR):
    subprocess.run(f"mkdir -p {PR} && tar --exclude=.git --exclude=checkpoints -cf - -C {GATE}/proj . | tar -xf - -C {PR}",shell=True,check=True)
shutil.copytree(f"{GATE}/hf/data/tokenizer",f"{PR}/data/tokenizer",dirs_exist_ok=True)
tpsha,_=sha(f"{PR}/training/train.py")
P("train.py sha256",tpsha)
assert tpsha=="5e0ec2639492046f663e03ffed3aebfaea684f0fd7b07034562a569609e60fbb","train.py sha mismatch"
# --- harness + fresh tasks/refs from gate dataset root; byte-verify against known originals ---
for name in ("cf_score.py","fresh_tasks.json","fresh_refs.json"):
    shutil.copy(f"{GATE}/{name}",f"{W}/{name}")
hsha,hlen=sha(f"{W}/cf_score.py"); P("cf_score.py sha256",hsha,hlen)
assert hlen==18192,"cf_score.py size mismatch"
# --- step-420 checkpoint; assert audited sha ---
# NOTE: periodic checkpoint_step_420.pt (2,951,298,643 B, sha 8bf7b940...) and clean-shutdown
# latest_checkpoint.pt (2,951,299,239 B, sha bc1af134...) differ by 596 B; CPU inspection (cf-ckpt-inspect
# V2) shows both carry identical top-level keys [loss, model_state_dict, optimizer_state_dict,
# resume_state, rng_state, scaler_state_dict, step, val_loss], step=420, 147 sd tensors. Whether the
# model weights themselves are tensor-identical was NOT compared; the audited byte-chain sha
# bc1af134... was computed on latest_checkpoint.pt (final save), so use that exact file.
CK=f"{W}/ckpt_step420.pt"
if not os.path.exists(CK):
    shutil.copy(f"{CONT}/proj/checkpoints/gate/latest_checkpoint.pt",CK)
csha,clen=sha(CK); P("ckpt sha256",csha,clen)
assert csha=="bc1af13481a4ea4567f0718887cebe5a3f8a047712ad3f651145135756a62315" and clen==2951299239,"ckpt sha mismatch"
ck=torch.load(CK,map_location="cpu",weights_only=True); rs=ck.get("resume_state") or {}
P("preflight step",ck.get("step"),"sample_cursor",rs.get("sample_cursor"),"completed_updates",rs.get("completed_updates"),"val_loss",ck.get("val_loss"),"sd_tensors",len(ck.get("model_state_dict") or {}))
assert ck.get("step")==420 and rs.get("completed_updates")==420 and rs.get("sample_cursor")==107520,"ckpt metadata mismatch"
assert abs((ck.get("val_loss") or 0)-1.691350150991369)<1e-6,"ckpt val_loss mismatch"
assert len(ck.get("model_state_dict") or {})==147,"sd tensor count mismatch"
del ck,rs
# --- val shards ---
VAL_DIR=f"{PACK}/val_ready"
assert glob.glob(f"{VAL_DIR}/shard_*.pt"),"no val shards"
P("val shards",len(glob.glob(f"{VAL_DIR}/shard_*.pt")))
# --- model config json (same as V7 score()) ---
import yaml
cfg=yaml.safe_load(open(f"{PR}/configs/config_250M.yaml"))
CFGJ=json.dumps({k:v for k,v in cfg["model"].items() if k not in("name","use_cache")})
# --- run the harness exactly as the original score() did ---
LABEL="cont_step420"
env=dict(os.environ,CF_PROJ=PR,CF_LOCAL_CKPT=CK,CF_VAL_DIR=VAL_DIR,CF_VAL_BATCHES="27",
         CF_FRESH_TASKS=f"{W}/fresh_tasks.json",CF_FRESH_REFS=f"{W}/fresh_refs.json",
         CF_NOUPLOAD="1",CF_EVAL_TAG=LABEL,CF_SCORE_DEADLINE=str(HARD_END),CF_CFG_JSON=CFGJ,
         CUDA_VISIBLE_DEVICES="0")
P("SCORE START",LABEL)
r=subprocess.run([sys.executable,f"{W}/cf_score.py"],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
keep=[l for l in r.stdout.splitlines() if re.match(r"(VAL_LOSS|EXPORTED|REFLOSS|INCOMPLETE|FINAL|RESULT|PARSE|Traceback|\w*Error|device|ckpt keys|load_state_dict|KV-cache)",l)]
for l in keep: print("  ",l[:600],flush=True)
os.makedirs(f"{W}/res_{LABEL}",exist_ok=True)
for f in glob.glob(f"{W}/*_samples.json")+[f"{W}/eval_results.json"]:
    if os.path.exists(f): shutil.move(f,f"{W}/res_{LABEL}/")
open(f"{W}/res_{LABEL}/log.txt","w").write(r.stdout[-60000:])
P("SCORE END",LABEL,"rc",r.returncode)
try: res=json.load(open(f"{W}/res_{LABEL}/eval_results.json"))
except Exception: res={}
# --- export manifest ---
man={"label":LABEL,"rc":r.returncode,"ckpt_sha256":csha,"ckpt_step":420,
     "harness_sha256":hsha,"exports":{}}
for f in sorted(glob.glob(f"{W}/{LABEL}_*_programs.jsonl")):
    s,n=sha(f); man["exports"][os.path.basename(f)]={"sha256":s,"bytes":n}
    mf=f.replace(".jsonl",".meta.json")
    if os.path.exists(mf): man["exports"][os.path.basename(mf)]={"sha256":sha(mf)[0],"bytes":sha(mf)[1]}
man["eval_results"]=res
json.dump(man,open(f"{W}/summary_eval420.json","w"),indent=1)
P("SUMMARY",json.dumps({k:man[k] for k in("label","rc","ckpt_step")}))
P("EXPORTS",json.dumps(man["exports"])[:900])
P("DONE elapsed_min",round(el(),1))
