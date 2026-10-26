"""Launch an isolated private CPU preparation notebook; never enables training."""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.kaggle_drive_backup import OWNER, owner_api

DEPENDENCIES = ["datasets==4.8.5", "datasketch==1.10.0", "transformers==5.13.0",
                "tokenizers==0.22.2", "huggingface-hub==1.24.0", "pyyaml==6.0.3", "requests==2.33.0"]


def source_bundle(config):
    config = config.resolve()
    if config.parent != PROJECT / "configs":
        raise ValueError("CPU preparation config must be inside this project's configs directory")
    names = ["data/prepare_fresh.py", "data/filter_quality.py", "data/deduplicate.py",
             "data/train_tokenizer.py", "data/tokenize_dataset.py", "kaggle/check_hf_secret.py",
             "configs/config_250M_fresh.yaml", "configs/" + config.name]
    output = io.BytesIO()
    identities = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            payload = (PROJECT / name).read_bytes()
            identities[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
    payload = output.getvalue()
    return payload, identities


def runner_source(payload, identities, config_name):
    prefix = "SOURCE_ARCHIVE = " + repr(base64.b64encode(payload).decode("ascii")) + "\n"
    prefix += "SOURCE_IDENTITIES = " + repr(identities) + "\n"
    prefix += "PREPARATION_CONFIG = " + repr(config_name) + "\n"
    prefix += "DEPENDENCIES = " + repr(DEPENDENCIES) + "\n"
    return prefix + r'''
import base64, hashlib, io, json, os, pathlib, subprocess, sys, time, zipfile
started = time.time()
os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_DISABLE_TELEMETRY="1",
                  HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                  HF_HOME="/tmp/codeforge-public-hf-cache")
root = pathlib.Path("/tmp/codeforge-preparation-source")
root.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
    archive.extractall(root)
for name, expected in SOURCE_IDENTITIES.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Source-bundle checksum mismatch")
pathlib.Path("/kaggle/working/source-manifest.json").write_text(json.dumps({
    "files": SOURCE_IDENTITIES, "GPU_enabled": False, "training_started": False
}, indent=2))
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "--disable-pip-version-check", *DEPENDENCIES], check=True, timeout=600)
env = {**os.environ, "PYTHONPATH": str(root)}
subprocess.run([sys.executable, str(root / "kaggle/check_hf_secret.py"), "--output",
                "/kaggle/working/hf-secret-status.json"], cwd=root, env=env, check=True, timeout=180)
remaining = max(1, int(3300 - (time.time() - started)))
subprocess.run([sys.executable, "-m", "data.prepare_fresh", "--config",
                str(root / "configs" / PREPARATION_CONFIG), "--output",
                "/kaggle/working/codeforge-fresh-pilot"], cwd=root, env=env,
               check=True, timeout=remaining)
print("CPU preparation finished. No model-training process was launched.", flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/fresh_data_pilot.yaml")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("Launch report needs a new filename in an existing directory")
    owner_api()  # authenticates the owner's token and rejects identity fallback
    payload, identities = source_bundle(args.config)
    suffix = uuid.uuid4().hex[:8]
    slug = f"{OWNER}/codeforge-fresh-cpu-prep-{suffix}"
    metadata = {"id": slug, "title": f"CodeForge Fresh CPU Prep {suffix}",
                "code_file": "fresh_cpu_prep.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    source = runner_source(payload, identities, args.config.name)
    compile(source, "fresh_cpu_prep.py", "exec")
    with tempfile.TemporaryDirectory(prefix="codeforge_cpu_push_") as staging:
        Path(staging, "fresh_cpu_prep.py").write_text(source, encoding="utf-8")
        Path(staging, "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        pushed = subprocess.run([sys.executable, "-B", "-m", "kaggle", "kernels", "push", "-p", staging,
                                 "--timeout", "3600"], check=True, capture_output=True, text=True)
        print(pushed.stdout, end="")
        url = re.search(r"https://www\.kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", pushed.stdout)
        version = re.search(r"Kernel version (\d+) successfully pushed", pushed.stdout)
        if not url or not version or url.group(1).split("/")[0] != OWNER:
            raise RuntimeError("Cannot confirm the owner's actual notebook URL and version")
        slug = url.group(1)
    report = {"kernel": slug, "version": int(version.group(1)), "owner": OWNER, "gpu_enabled": False,
              "training_started": False, "source_bundle_sha256": hashlib.sha256(payload).hexdigest(),
              "source_files": identities, "dependencies": DEPENDENCIES}
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
