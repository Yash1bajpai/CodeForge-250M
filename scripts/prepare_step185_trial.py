"""Prepare only: self-contained CPU preflight and bounded single-T4 launch artifacts."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

CPU_BODY = r'''
import os, time, signal
STARTED = time.time()
os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
import argparse, gc, hashlib, importlib.util, json, resource, sys, tempfile, unittest
from pathlib import Path

def deadline(signum, frame):
    raise TimeoutError("CPU preflight 540-second local stop; backend must be600seconds")
signal.signal(signal.SIGALRM, deadline)
signal.alarm(540)
receipt = {"status": "INCOMPLETE_HOLD", "phase": "source_and_inputs", "training_started": False, "optimizer_steps": 0,
           "backward_calls": 0, "model_constructions": 0, "live_waiver_claims": 0,
           "backend_timeout_seconds_required": 600, "resources": {"cpu": 4, "ram_gb": 30}}
output = Path("/kaggle/working/step185-cpu-preflight-receipt.json")
try:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Prepared script is for the approved Kaggle CPU job only")
    if (os.cpu_count() or 0) not in range(1, 5):
        raise RuntimeError("CPU allocation exceeds approval")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer-json", default="/kaggle/input/notebooks/yashbajpai2027/codeforge-fresh-cpu-prep-b711eca4/codeforge-production/tokenizer/tokenizer.json")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="cf-step185-preflight-source-"))
    for name, encoded in SOURCE_B64.items():
        raw = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA[name]:
            raise RuntimeError("Bundled source checksum mismatch")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    sys.path.insert(0, str(root))
    from training.step185_trial import BINDING, checkpoint_metadata, file_sha256, load_checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        from huggingface_hub import hf_hub_download
        checkpoint_path = Path(hf_hub_download(repo_id=BACKUP["repo_id"], filename=BACKUP["path"], revision=BACKUP["commit"], force_download=True, cache_dir="/tmp/cf-step185-backup-cache"))
    if checkpoint_path.stat().st_size != BACKUP["bytes"] or file_sha256(checkpoint_path) != BACKUP["sha256"]:
        raise RuntimeError("Backed-up checkpoint size/SHA mismatch")
    tokenizer_path = Path(args.tokenizer_json)
    if file_sha256(tokenizer_path) != BINDING["tokenizer_sha256"]:
        raise RuntimeError("Frozen tokenizer byte hash mismatch")
    receipt["phase"] = "metadata_only_load"
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("GPU unexpectedly visible in CPU preflight")
    state = load_checkpoint(checkpoint_path)
    metadata = checkpoint_metadata(state)
    del state
    gc.collect()
    actual_validation_identity = metadata["resume_state"].get("extension", {}).get("validation_identity")
    if actual_validation_identity != BINDING["validation_identity"]:
        raise RuntimeError("Real checkpoint extension.validation_identity mismatch")
    receipt.update(checkpoint_sha256=BACKUP["sha256"], actual_validation_identity=actual_validation_identity,
                   step=metadata["step"], sample_cursor=metadata["resume_state"]["sample_cursor"])
    receipt["phase"] = "model_free_contracts_and_cpu_entry"
    specification = importlib.util.spec_from_file_location("trial_contract", root / "tests/test_step185_trial.py")
    contract = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(contract)
    contract.Step185TrialTest.real_checkpoint_metadata = metadata
    contract.Step185TrialTest.source_root = root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(contract.Step185TrialTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Real-metadata trial contracts failed")
    # Exercise the actual train.py CPU entry path, with hard traps on forbidden work.
    import training.train as engine
    import training.trial_storage as storage
    def forbidden(*unused, **kwargs):
        raise RuntimeError("Forbidden model/optimizer/CUDA/claim operation in CPU preflight")
    engine.CodeForgeModel = forbidden
    storage.HFTrialStore = forbidden
    torch.optim.AdamW = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.cuda.set_device = forbidden
    config = dict(metadata["resume_state"]["identity_context"]["config"])
    config["checkpointing"] = {"output_dir": "/tmp/cf-preflight-no-checkpoint-output"}
    config_path = root / "preflight-config.json"
    config_path.write_text(json.dumps(config))
    base = ["train.py", "--resume", "--cpu-resume-preflight", "--resume-checkpoint", str(checkpoint_path),
            "--config", str(config_path), "--validation-history-sidecar", str(root / "configs/step185_validation_history.json"),
            "--tokenizer-dir", str(tokenizer_path.parent)]
    from training.validation_history import ValidationGuardStopped
    sys.argv = base
    try:
        engine.train()
    except ValidationGuardStopped:
        receipt["ordinary_resume_refused"] = True
    else:
        raise RuntimeError("Ordinary185 resume incorrectly passed")
    sys.argv = base + ["--trial-waiver", str(root / "configs/step185_trial_waiver.json"), "--trial-start-epoch", str(STARTED)]
    engine.train()
    receipt.update(status="CPU_PREFLIGHT_PASS_ONLY_TRAINING_STILL_HOLD", phase="complete", tests_run=result.testsRun,
                   waiver_preflight_passed_not_claimed=True, source_files=SOURCE_SHA,
                   validation_check_scope="Real checkpoint/sidecar identity; actual corpus-shard validation remains in trainer")
except BaseException as error:
    receipt["error_type"] = type(error).__name__
    # Do not print provider exceptions or signed URLs.
    receipt["status"] = "CPU_PREFLIGHT_FAILED_HOLD"
finally:
    signal.alarm(0)
    receipt.update(elapsed_seconds=time.time() - STARTED, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if output.parent.is_dir():
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, sort_keys=True), flush=True)
if receipt["status"] != "CPU_PREFLIGHT_PASS_ONLY_TRAINING_STILL_HOLD":
    raise SystemExit(1)
'''

GPU_BODY = r'''
import time
STARTED = time.time()
import argparse, hashlib, json, os, signal, subprocess, sys, tempfile
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--production-root", required=True)
parser.add_argument("--validation-root", required=True)
parser.add_argument("--tokenizer-dir", required=True)
parser.add_argument("--admission-receipt", required=True)
args = parser.parse_args()
if not Path("/kaggle/working").is_dir() or os.environ.get("CF_TRAINING_APPROVED") != "1":
    raise SystemExit("Only the explicitly approved cloud training invocation may run this launcher")
receipt = json.loads(Path(args.admission_receipt).read_text())
if (receipt.get("owner") != "yashbajpai2027" or receipt.get("free_compute") is not True
        or receipt.get("run_id") != RUN_ID or receipt.get("gpu_name") != "Tesla T4"
        or receipt.get("allocated_gpu_count") != 1
        or receipt.get("remaining_quota_seconds", 0) < 3600
        or not 0 <= STARTED - receipt.get("checked_at_epoch", 0) <= 600
        or receipt.get("backend_timeout_seconds") != 3600):
    raise SystemExit("Fresh free-quota/admission proof for one allocated T4 and backend3600seconds is required; no fallback")
for field in ("checkpoint", "config", "production_root", "validation_root", "tokenizer_dir"):
    setattr(args, field, str(Path(getattr(args, field)).resolve()))
root = Path(tempfile.mkdtemp(prefix="cf-step185-trial-source-"))
import base64
for name, encoded in SOURCE_B64.items():
    raw = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA[name]:
        raise SystemExit("Source checksum mismatch")
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
sys.path.insert(0, str(root))
from training.step185_trial import file_sha256
checks = [
    (Path(args.config), "f009b67a8b0d302b7f487e0c2d73a4be4d553f295c92564e14d68ee7a410793a"),
    (Path(args.production_root) / "preparation-manifest.json", "fc8b923d05339dcc17379d84287c51e6d2be273d37b6b71350c4cd804d9ad70f"),
    (Path(args.validation_root) / "preparation-manifest.json", "c4e7e61ea7c7288b4329522e4ecda4279133feb0049e40e62d56f251c230ff3c"),
]
if any(file_sha256(path) != expected for path, expected in checks):
    raise SystemExit("Original config/production/frozen validation input mismatch")
os.environ.update(CUDA_VISIBLE_DEVICES="0", WORLD_SIZE="1", TOKENIZERS_PARALLELISM="false", CF_TELEMETRY_DISABLED="1")
command = [sys.executable, "-m", "training.train", "--resume", "--resume-checkpoint", args.checkpoint,
           "--config", args.config, "--production-root", args.production_root,
           "--validation-root", args.validation_root, "--tokenizer-dir", args.tokenizer_dir,
           "--validation-history-sidecar", str(root / "configs/step185_validation_history.json"),
           "--trial-waiver", str(root / "configs/step185_trial_waiver.json"),
           "--trial-start-epoch", str(STARTED), "--trial-output-root", "/kaggle/working/step185-trials",
           "--session-steps", "8", "--max_hours", "1"]
process = subprocess.Popen(command, cwd=root, start_new_session=True)
try:
    returncode = process.wait(timeout=max(0.01, 3600 - (time.time() - STARTED)))
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    Path("/kaggle/working/step185-trial-launcher-hold.json").write_text(json.dumps({"status": "WALL_TIMEOUT_HOLD", "run_id": RUN_ID, "automatic_retry": False}))
    raise SystemExit(124)
raise SystemExit(returncode)
'''


def prepare(destination):
    names = ["models/architecture.py", "models/init_weights.py", "training/train.py",
             "training/utils.py", "training/resume_safety.py", "training/production_data.py",
             "training/token_accounting.py", "training/dataset_extension.py",
             "training/validation_history.py", "training/step185_trial.py", "training/trial_storage.py",
             "configs/step185_validation_history.json", "configs/step185_trial_waiver.json",
             "tests/test_step185_trial.py"]
    payloads = {name: (PROJECT / name).read_bytes() for name in names}
    encoded = {name: base64.b64encode(raw).decode("ascii") for name, raw in payloads.items()}
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
    backup = {"repo_id": "Yash1bajpai/CodeForge-250M-rotation",
              "path": "backups/fresh-step185-20261029/latest_checkpoint.pt",
              "commit": "cd3a9f085f5ca31e5ea62d6c818db973f87b9387",
              "bytes": 2951302119,
              "sha256": "2b69e2c21778bb494e281285e852edb4ac0b1e4d89fb55bc58b5a3990587f5b5"}
    prefix = "SOURCE_B64 = " + repr(encoded) + "\nSOURCE_SHA = " + repr(hashes) + "\nBACKUP = " + repr(backup) + "\nRUN_ID = 'fresh-step185-trial-20261029-183152'\n"
    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    artifacts = {}
    for name, body in (("cpu_preflight.py", CPU_BODY), ("single_t4_launch.py", GPU_BODY)):
        path = destination / name
        path.write_text(prefix + body)
        # Inspect the actual on-disk artifact, not a hypothetical roundtrip.
        import ast
        tree = ast.parse(path.read_bytes())
        embedded = ast.literal_eval(tree.body[0].value)
        embedded_hashes = ast.literal_eval(tree.body[1].value)
        for source, value in embedded.items():
            raw = base64.b64decode(value, validate=True)
            if raw != payloads[source] or hashlib.sha256(raw).hexdigest() != embedded_hashes[source]:
                raise ValueError("Actual artifact payload mismatch: " + source)
        compile(path.read_bytes(), str(path), "exec")
        artifacts[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    launch = {"run_id": "fresh-step185-trial-20261029-183152",
              "approval_reference": "Yash WhatsApp 2026-10-29 18:31:52 IST",
              "cpu": {"script": "cpu_preflight.py", "cores": 4, "ram_gb": 30, "accelerator": "None",
                      "backend_timeout_seconds": 600, "local_stop_seconds": 540},
              "gpu": {"script": "single_t4_launch.py", "allocated_gpus": 1, "visible_gpus": 1,
                      "gpu_name": "Tesla T4", "backend_timeout_seconds": 3600,
                      "max_attempted_updates": 8, "max_applied_updates": 8,
                      "no_update_after_seconds": 2100, "val_steps": 1, "val_batches": 1,
                      "output": "/kaggle/working/step185-trials/fresh-step185-trial-20261029-183152",
                      "physical_allocation_note": "A T4x2 allocation is NOT silently accepted as one T4; no DDP or paid fallback"},
              "admission_receipt_contract": {
                  "owner": "yashbajpai2027", "run_id": "fresh-step185-trial-20261029-183152",
                  "free_compute": True, "gpu_name": "Tesla T4", "allocated_gpu_count": 1,
                  "remaining_quota_seconds_minimum": 3600, "checked_at_epoch_maximum_age_seconds": 600,
                  "backend_timeout_seconds": 3600,
                  "evidence_required": "Instinct verifies live provider quota/allocation and records actual values; this generator does not check or submit anything"},
              "pinned_inputs": {
                  "training_config_kernel": "yashbajpai2027/codeforge-fresh-20261028-extend-gpu/1",
                  "training_config_file": "cf-fresh-20261027-overnight/smoke.yaml",
                  "training_config_sha256": "f009b67a8b0d302b7f487e0c2d73a4be4d553f295c92564e14d68ee7a410793a",
                  "production_kernel": "yashbajpai2027/codeforge-fresh-cpu-prep-b711eca4/1",
                  "frozen_parent_manifest_sha256": "c4e7e61ea7c7288b4329522e4ecda4279133feb0049e40e62d56f251c230ff3c",
                  "frozen_validation_identity": "af30f98d37e10b39846363903d5a2b01a626ff929746b0e3a05fc689d6ee9a1d"},
              "backup": backup, "source_hashes": hashes, "artifacts": artifacts,
              "execution_started": False, "submissions": 0, "automatic_retry": False,
              "required_launcher_arguments": ["--checkpoint", "--config", "--production-root", "--validation-root", "--tokenizer-dir", "--admission-receipt"]}
    (destination / "launch-config.json").write_text(json.dumps(launch, indent=2, sort_keys=True))
    print(json.dumps({"prepared_only": True, "directory": str(destination), "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
