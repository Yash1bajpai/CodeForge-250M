# CodeForge continuation training (private). Single T4 via CUDA_VISIBLE_DEVICES=0.
# Resumes the audited single-pass run from the preserved gate checkpoint (step 100),
# source + checkpoint mounted read-only from the private dataset cf-gate-v7-gate-step100-output.
# No HF token, no Kaggle secrets: tokenizer + checkpoint + patched source all come from the mount.
import os,sys,time,json,subprocess,hashlib,glob,shutil,threading,re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
SESSION_START=time.time()-300   # conservative: Kaggle session boot precedes cell start
def el(): return (time.time()-SESSION_START)/60
def P(*a): print(f"[{el():6.1f}m]",*a,flush=True)
import math, datetime
PHASE="cont1"
# Session budget: Kaggle GPU session limit ~9h; train 320 updates (~7.6-8h) then stop cleanly.
SESSION_LIMIT_MIN=525; MARGIN_MIN=12
HARD_END=SESSION_START+(SESSION_LIMIT_MIN-MARGIN_MIN)*60
TRAIN_STOP=HARD_END-40*60   # stop training 40 min before hard end: checkpoint save + summary + shutdown margin
P("CELL VERSION cont1; PHASE=",PHASE,"CF_EXEC_MODE=",os.environ.get("CF_EXEC_MODE"))
P("SESSION_START_EPOCH",SESSION_START,"HARD_END",HARD_END,"TRAIN_STOP",TRAIN_STOP)
os.environ["CUDA_VISIBLE_DEVICES"]="0"; os.environ["CF_TELEMETRY_DISABLED"]="1"
os.environ["CF_EXEC_MODE"]="export"   # never execute generated code on Kaggle
W="/kaggle/working"; PR=f"{W}/proj"
def sh(c,check=True):
    r=subprocess.run(c,shell=True,capture_output=True,text=True)
    if check and r.returncode: raise SystemExit(f"cmd failed: {c[:120]}\n{(r.stdout+r.stderr)[-800:]}")
    return r
sh("pip -q install huggingface_hub datasets 2>&1 | tail -1",check=False)
import torch
P("torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
if not torch.cuda.is_available(): raise SystemExit("NO GPU")
# --- mounts ---
SRC_CKPT=glob.glob("/kaggle/input/**/proj/checkpoints/gate/latest_checkpoint.pt",recursive=True)
SRC_TOK=glob.glob("/kaggle/input/**/hf/data/tokenizer/tokenizer.json",recursive=True)
P("mounted gate ckpt",SRC_CKPT); P("mounted tokenizer",SRC_TOK)
assert len(SRC_CKPT)==1 and len(SRC_TOK)==1, "preservation dataset mount not found exactly once"
SRC_PROJ=os.path.dirname(os.path.dirname(os.path.dirname(SRC_CKPT[0])))  # .../proj
P("source proj mount",SRC_PROJ)
trd=[d for d in glob.glob("/kaggle/input/**/train_ready",recursive=True)]; vad=[d for d in glob.glob("/kaggle/input/**/val_ready",recursive=True)]
P("train_ready dirs",trd,"val_ready dirs",vad); assert len(trd)==1 and len(vad)==1
ts=sorted(glob.glob(trd[0]+"/shard_*.pt")); tv=sorted(glob.glob(trd[0]+"/valid_*.pt")); vs=sorted(glob.glob(vad[0]+"/shard_*.pt"))
P("train shards",len(ts),"valid sidecars",len(tv),"val shards",len(vs)); assert len(ts)==119 and len(vs)>=1
# --- project from mounted V7 output (exact provenance; no clone, no patch drift) ---
sh(f"rm -rf {PR}; mkdir -p {PR}")
sh(f"cd '{SRC_PROJ}' && tar cf - --exclude=./checkpoints --exclude=./.git . | tar xf - -C {PR}")
h=hashlib.sha256(open(f"{PR}/training/train.py","rb").read()).hexdigest()
P("train.py sha256",h); assert h.startswith("5e0ec2639492046f663e"),h
os.makedirs(f"{PR}/data",exist_ok=True)
shutil.copytree(os.path.dirname(SRC_TOK[0]),f"{PR}/data/tokenizer",dirs_exist_ok=True)
P("tokenizer.json sha256",hashlib.sha256(open(f"{PR}/data/tokenizer/tokenizer.json","rb").read()).hexdigest())
tk=f"{PR}/data/tokenized"; os.makedirs(tk,exist_ok=True)
for f in ts+tv: os.symlink(f,f"{tk}/{os.path.basename(f)}")
VAL_DIR=vad[0]
# --- resume checkpoint from mount ---
os.makedirs(f"{PR}/checkpoints/gate",exist_ok=True)
CK=f"{PR}/checkpoints/gate/latest_checkpoint.pt"
shutil.copy(SRC_CKPT[0],CK)
hh=hashlib.sha256()
with open(CK,"rb") as f:
    for b in iter(lambda:f.read(1<<24),b""): hh.update(b)
P("resume ckpt sha256",hh.hexdigest(),os.path.getsize(CK))
assert hh.hexdigest()=="5b253630e561b71843000d0cfd92a1284c2edf133cac18f6bf5a2326e954d750"
c=torch.load(CK,map_location="cpu",weights_only=True); rs=c.get("resume_state") or {}
P("RESUME ckpt step",c.get("step"),"sample_cursor",rs.get("sample_cursor"),"completed_updates",rs.get("completed_updates"),"has_scaler",bool(c.get("scaler_state_dict")),"has_opt",bool(c.get("optimizer_state_dict")),"val",c.get("val_loss"))
assert c.get("step")==100 and rs.get("sample_cursor")==25600 and rs.get("completed_updates")==100 and c.get("optimizer_state_dict"), "resume state mismatch - refusing to train"
del c
# --- config: identical to gate run except output dir (same) and session cap ---
import yaml
cfg=yaml.safe_load(open(f"{PR}/configs/config_250M.yaml"))
cfg["training"].update(dict(learning_rate=1.0e-4,min_learning_rate=1.0e-5,warmup_steps=50,max_steps=462,val_steps=50,val_batches=27,batch_size_per_device=8,gradient_accumulation_steps=32))
cfg["checkpointing"]=dict(save_steps=1000,save_time_interval_minutes=20,output_dir="checkpoints/gate",push_to_hub=False)
yaml.safe_dump(cfg,open(f"{PR}/configs/run.yaml","w")); P("config",json.dumps(cfg["training"]))
# --- training (single run, resume) ---
GV=1.748785138130188   # gate step-100 val loss: abort guard reference
def train(args,tag):
    env=dict(os.environ,CF_VAL_DIR=VAL_DIR,CF_ABSOLUTE_STOP_EPOCH=str(TRAIN_STOP))
    lg=open(f"{W}/train_{tag}.log","w")
    p=subprocess.Popen([sys.executable,"training/train.py","--config","configs/run.yaml","--max_hours","8.2"]+args,cwd=PR,env=env,stdout=lg,stderr=subprocess.STDOUT)
    st={"abort":None}
    def mon():
        seen=0; mp=f"{PR}/metrics.json"
        while p.poll() is None:
            time.sleep(20)
            try: rows=[json.loads(l) for l in open(mp) if l.strip()]
            except Exception: continue
            for r in rows[seen:]:
                ev=r.get("event")
                if ev=="VAL":
                    P("VAL step",r["step"],"val_loss",r["val_loss"],"gate_ref",round(GV,4))
                    if r["val_loss"]>GV+0.02 or r["val_loss"]!=r["val_loss"]: st["abort"]=f"val {r['val_loss']} > gate {GV:.4f}+0.02"
                elif ev=="TRAIN":
                    P("train step",r["step"],"loss",r["loss"],"lr",f"{r['lr']:.2e}")
                    if r["loss"]!=r["loss"] or r["loss"]>20: st["abort"]=f"bad train loss {r['loss']}"
                elif ev in("CKPT","WATCHDOG","CRASH"): P(ev,json.dumps(r)[:400])
            seen=len(rows)
            if st["abort"] and not os.path.exists(f"{PR}/STOP_AND_SAVE"):
                P("ABORT RULE HIT:",st["abort"],"-> STOP_AND_SAVE"); open(f"{PR}/STOP_AND_SAVE","w").write("x")
    t=threading.Thread(target=mon,daemon=True); t.start(); rc=p.wait(); t.join(timeout=25); lg.close()
    tail=open(f"{W}/train_{tag}.log",errors="replace").read().splitlines()[-25:]
    for l in tail: print("  |",l[:300],flush=True)
    return rc,st["abort"]
if os.path.exists(f"{PR}/metrics.json"): os.remove(f"{PR}/metrics.json")
P("TRAIN CONT: --resume from step 100, 320 updates target (to ~step 420)")
rc1,ab1=train(["--resume","--session-steps","320"],"c1"); P("cont rc",rc1,"abort",ab1)
c1log=open(f"{W}/train_c1.log",errors="replace").read()
m=re.search(r"data position: ([\d,]+)/([\d,]+)",c1log); P("RESUME data position line:",m.group(0) if m else "NOT FOUND")
c=torch.load(CK,map_location="cpu",weights_only=True); fstep=c.get("step"); fval=c.get("val_loss"); frs=c.get("resume_state") or {}
P("FINAL ckpt step",fstep,"val",fval,"sample_cursor",frs.get("sample_cursor"),"completed_updates",frs.get("completed_updates"))
del c
hh=hashlib.sha256()
with open(CK,"rb") as f:
    for b in iter(lambda:f.read(1<<24),b""): hh.update(b)
final_sha=hh.hexdigest(); P("FINAL ckpt sha256",final_sha,os.path.getsize(CK))
json.dump({"phase":PHASE,"session_start_epoch":SESSION_START,"elapsed_min_at_end":el(),"resume_from_step":100,"final_step":fstep,"final_val":fval,"final_ckpt_sha256":final_sha,"abort":ab1,"rc":rc1},open(f"{W}/summary_cont.json","w"),indent=1,default=str)
P("DONE elapsed_min",round(el(),1))
