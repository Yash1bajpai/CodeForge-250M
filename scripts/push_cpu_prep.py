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
    names = ["configs/config_250M_fresh.yaml", "configs/" + config.name]
    # Ship source only, never caches, credentials or dataset artifacts.
    for directory in ("data", "models", "training", "evaluation", "serving", "scripts", "tests"):
        names.extend(str(path.relative_to(PROJECT)).replace("\\", "/")
                     for path in (PROJECT / directory).glob("*.py"))
    names.append("kaggle/check_hf_secret.py")
    output = io.BytesIO()
    identities = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            payload = (PROJECT / name).read_bytes()
            identities[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
    payload = output.getvalue()
    return payload, identities


def runner_source(payload, identities, config_name, production=False, resume=None, verify_only=False, session_max_tokens=None, tests_only=False, tokenizer_dataset=None):
    prefix = "SOURCE_ARCHIVE = " + repr(base64.b64encode(payload).decode("ascii")) + "\n"
    prefix += "SOURCE_IDENTITIES = " + repr(identities) + "\n"
    prefix += "PREPARATION_CONFIG = " + repr(config_name) + "\n"
    prefix += "DEPENDENCIES = " + repr(DEPENDENCIES) + "\n"
    prefix += "PRODUCTION = " + repr(production) + "\n"
    prefix += "RESUME = " + repr(resume) + "\n"
    prefix += "VERIFY_ONLY = " + repr(verify_only) + "\n"
    prefix += "SESSION_MAX_TOKENS = " + repr(session_max_tokens) + "\n"
    prefix += "TESTS_ONLY = " + repr(tests_only) + "\n"
    prefix += "TOKENIZER_DATASET = " + repr(tokenizer_dataset) + "\n"
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
if not TESTS_ONLY:
    subprocess.run([sys.executable, str(root / "kaggle/check_hf_secret.py"), "--output",
                    "/kaggle/working/hf-secret-status.json"], cwd=root, env=env, check=True, timeout=180)
if PRODUCTION:
    # Contract tests and ALL corpus/tokenizer work execute on Kaggle, not the PC.
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests")],
                   cwd=root, env=env, check=True, timeout=600)
    print("CPU_CONTRACT_TESTS_PASSED", flush=True)
    if TESTS_ONLY:
        pathlib.Path("/kaggle/working/contract-test-verification.json").write_text(json.dumps({
            "status": "passed", "corpus_processing_started": False, "model_training_started": False,
            "source_files": SOURCE_IDENTITIES}, indent=2))
        raise SystemExit(0)
    sys.path.insert(0, str(root))
    import yaml
    from data.prepare_production import source_rows, admit
    config = yaml.safe_load((root / "configs" / PREPARATION_CONFIG).read_text())
    sources = {}
    for spec in config["sources"]:
        accepted = False
        counts = {}
        stream = source_rows(spec, {"file": 0, "row": 0})
        for index, (row, filename, row_index, license_value) in enumerate(stream):
            record, rejected = admit(row, spec, config, set(), license_value)
            if record is not None:
                sources[spec["name"]] = {"format_verified": True, "file": filename,
                                         "fields": sorted(row), "characters": len(record["text"])}
                accepted = True
                break
            counts[rejected] = counts.get(rejected, 0) + 1
            if index >= 99:
                break
        if not accepted:
            raise ValueError("Source contract rejected first 100 rows: " + spec["name"] + " " + str(counts))
    pathlib.Path("/kaggle/working/source-contract-verification.json").write_text(json.dumps({
        "status": "verified", "sources": sources, "model_training_started": False}, indent=2))
    print("PUBLIC_SOURCE_CONTRACTS_PASSED " + json.dumps(sources), flush=True)
    if VERIFY_ONLY:
        raise SystemExit(0)
    if TOKENIZER_DATASET:
        import shutil
        from data.prepare_production import tokenizer_identity, validate_tokenizer
        from data.prepare_fresh import file_sha256
        candidates = [path.parent for path in pathlib.Path("/kaggle/input").rglob("tokenizer-identity.json")
                      if (path.parent / "tokenizer/tokenizer.json").is_file()
                      and (path.parent / "tokenizer-sample.json").is_file()]
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one frozen-tokenizer recovery candidate")
        prior = candidates[0]
        sample = json.loads((prior / "tokenizer-sample.json").read_text())
        config_hash = file_sha256(root / "configs" / PREPARATION_CONFIG)
        identity = json.loads((prior / "tokenizer-identity.json").read_text())
        if (sample.get("config_sha256") != config_hash or sample.get("train_only") is not True
                or sample.get("sha256") != file_sha256(prior / "tokenizer-sample/foundation_dedup.jsonl")
                or identity != tokenizer_identity(prior / "tokenizer")):
            raise ValueError("Recovery tokenizer/sample/recipe identity mismatch")
        validate_tokenizer(prior / "tokenizer", config["vocab_size"])
        destination = pathlib.Path("/kaggle/working/codeforge-production")
        destination.mkdir()
        shutil.copytree(prior / "tokenizer", destination / "tokenizer")
        (destination / "tokenizer-sample").mkdir()
        shutil.copyfile(prior / "tokenizer-sample/foundation_dedup.jsonl",
                        destination / "tokenizer-sample/foundation_dedup.jsonl")
        for name in ("tokenizer-identity.json", "tokenizer-sample.json"):
            shutil.copyfile(prior / name, destination / name)
        (destination / "tokenizer-adoption.json").write_text(json.dumps({
            "dataset": TOKENIZER_DATASET, "tokenizer": identity,
            "config_sha256": config_hash, "sample_sha256": sample["sha256"],
            "old_state_or_windows_imported": False, "old_tokens_promoted": False,
            "model_training_started": False}, indent=2))
        print("FROZEN_TOKENIZER_ADOPTED no_refit=true sha256=" + identity["sha256"], flush=True)
    command = [sys.executable, "-u", "-m", "data.prepare_production", "--config",
               str(root / "configs" / PREPARATION_CONFIG), "--output",
               "/kaggle/working/codeforge-production"]
    if RESUME:
        candidates = [path.parent for path in pathlib.Path("/kaggle/input").rglob("state.sqlite3")
                      if (path.parent / "tokenizer/tokenizer.json").is_file()]
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one attached preparation checkpoint database")
        command.extend(["--resume-from", str(candidates[0])])
        env["CF_PREP_PARENT_KERNEL"] = RESUME
    if SESSION_MAX_TOKENS is not None:
        env["CF_PREP_SESSION_TOKEN_LIMIT"] = str(SESSION_MAX_TOKENS)
    subprocess.run(command, cwd=root, env=env, check=True, timeout=15500)
else:
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
    parser.add_argument("--resume-kernel", help="Completed owner notebook with saved preparation checkpoint output")
    parser.add_argument("--tokenizer-dataset", help="Exact owner/slug/version recovery dataset; adopt tokenizer/sample only into fresh data state")
    parser.add_argument("--verify-only", action="store_true", help="CPU contract tests and public-source preflight only")
    parser.add_argument("--tests-only", action="store_true", help="Run isolated contract checks on Kaggle without source downloads or preparation")
    parser.add_argument("--session-max-tokens", type=int, help="Operational input-token cap; does not change recipe identity")
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("Launch report needs a new filename in an existing directory")
    api = owner_api()  # authenticates the owner's token and rejects identity fallback
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    production = config.get("execution_host") == "kaggle"
    if (args.resume_kernel or args.tokenizer_dataset or args.verify_only or args.tests_only) and not production:
        raise ValueError("Production operations require the Kaggle-only preparation config")
    if args.tests_only and (args.resume_kernel or args.tokenizer_dataset or args.verify_only):
        raise ValueError("--tests-only cannot resume, adopt a tokenizer or run source preflight")
    if args.tokenizer_dataset:
        if args.resume_kernel or args.verify_only:
            raise ValueError("Tokenizer adoption requires fresh preparation, not resume or verification-only")
        if not re.fullmatch(re.escape(OWNER) + r"/[a-z0-9][a-z0-9-]*/[1-9][0-9]*", args.tokenizer_dataset):
            raise ValueError("Tokenizer recovery requires an exact owner dataset version")
    if args.session_max_tokens is not None and (not production or args.session_max_tokens < 1):
        raise ValueError("A positive session-token cap requires production preparation")
    resume_version = None
    resume_status = resume_image = None
    if args.resume_kernel:
        if not re.fullmatch(re.escape(OWNER) + r"/[a-z0-9][a-z0-9-]*", args.resume_kernel):
            raise ValueError("Only the owner's saved preparation notebook may be attached")
        resume_status = api.kernels_status(args.resume_kernel).status.name
        if resume_status != "COMPLETE":
            raise ValueError("Kaggle rejects ERROR notebook inputs; resume requires saved COMPLETE outputs")
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        with api.build_kaggle_client() as client:
            request = ApiGetKernelRequest()
            request.user_name, request.kernel_slug = args.resume_kernel.split("/")
            parent_metadata = client.kernels.kernels_api_client.get_kernel(request).metadata
            resume_version = parent_metadata.current_version_number
            resume_image = parent_metadata.docker_image
            if not resume_image:
                raise ValueError("Cannot pin the checkpoint's Kaggle runtime image")
    payload, identities = source_bundle(args.config)
    suffix = uuid.uuid4().hex[:8]
    slug = f"{OWNER}/codeforge-fresh-cpu-prep-{suffix}"
    metadata = {"id": slug, "title": f"CodeForge Fresh CPU Prep {suffix}",
                "code_file": "fresh_cpu_prep.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    if args.tokenizer_dataset:
        metadata["dataset_sources"] = [args.tokenizer_dataset]
    if args.resume_kernel:
        metadata["kernel_sources"] = [args.resume_kernel + "/" + str(resume_version)]
        metadata["docker_image_pinning_type"] = "original"
        metadata["docker_image"] = resume_image
    resume_reference = args.resume_kernel + "/" + str(resume_version) if args.resume_kernel else None
    source = runner_source(payload, identities, args.config.name, production, resume_reference,
                           args.verify_only, args.session_max_tokens, args.tests_only, args.tokenizer_dataset)
    compile(source, "fresh_cpu_prep.py", "exec")
    with tempfile.TemporaryDirectory(prefix="codeforge_cpu_push_") as staging:
        Path(staging, "fresh_cpu_prep.py").write_text(source, encoding="utf-8")
        Path(staging, "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        pushed = subprocess.run([sys.executable, "-B", "-m", "kaggle", "kernels", "push", "-p", staging,
                                 "--timeout", "1800" if args.tests_only else ("18000" if production else "3600")],
                                check=True, capture_output=True, text=True)
        print(pushed.stdout, end="")
        url = re.search(r"https://www\.kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", pushed.stdout)
        version = re.search(r"Kernel version (\d+) successfully pushed", pushed.stdout)
        if not url or not version or url.group(1).split("/")[0] != OWNER:
            raise RuntimeError("Cannot confirm the owner's actual notebook URL and version")
        slug = url.group(1)
    report = {"kernel": slug, "version": int(version.group(1)), "owner": OWNER, "gpu_enabled": False,
              "training_started": False, "source_bundle_sha256": hashlib.sha256(payload).hexdigest(),
              "source_files": identities, "dependencies": DEPENDENCIES}
    report.update(production_preparation=production and not (args.tests_only or args.verify_only),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  resume_kernel=args.resume_kernel, resume_version=resume_version,
                  resume_source_status=resume_status, resume_image=resume_image,
                  tokenizer_dataset=args.tokenizer_dataset,
                  execution_host="kaggle", model_training_authorized=False)
    report["verification_only"] = args.verify_only
    report["tests_only"] = args.tests_only
    report["session_max_tokens"] = (args.session_max_tokens or config.get("session_max_tokens")) if report["production_preparation"] else None
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
