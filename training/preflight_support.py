"""Small CPU-entry and single-used-GPU checks; no torch, credentials or cloud access."""
import contextlib
import copy
import io
import json
import re
import sys
from pathlib import Path

from training.step185_trial import RUN_ID


def write_cpu_config(context, destination):
    """Match train.py's YAML reader, including scientific-notation float types."""
    import yaml
    config = copy.deepcopy(context)
    config["checkpointing"] = {"output_dir": "/tmp/cf-preflight-no-checkpoint-output"}
    path = Path(destination)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if {key: loaded.get(key) for key in context} != context:
        raise ValueError("CPU preflight YAML roundtrip changed checkpoint config")
    return path


def sanitize(text):
    text = str(text)
    text = re.sub(r"https?://[^\s<>\"']+", "[URL_REDACTED]", text)
    text = re.sub(r"(?im)(authorization\s*[:=]\s*)[^\r\n]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:hf_token|access_token|api[_-]?key|password|secret|token)[\"']?\s*[:=]\s*[\"']?)[^\s\"',;}]+",
                  r"\1[REDACTED]", text)
    text = re.sub(r"\b(?:hf_|gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]+", "[CREDENTIAL_REDACTED]", text)
    text = re.sub(r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "[JWT_REDACTED]", text)
    # Redact unlabelled opaque values too. Known uppercase status names remain useful.
    def opaque(match):
        value = match.group()
        if value.isupper() and "_" in value and not any(char.isdigit() for char in value):
            return value
        return "[OPAQUE_REDACTED]"
    return re.sub(r"[A-Za-z0-9_+/=-]{32,}", opaque, text)


def capture_entry(entry, argv, name):
    """Capture only metadata-only entries; redact full streams BEFORE retaining tails."""
    stdout, stderr = io.StringIO(), io.StringIO()
    record = {"name": name, "exit_code": 0, "system_exit_code": None,
              "system_exit_message": None, "exception_type": None, "exception_message": None}
    original = sys.argv
    try:
        sys.argv = list(argv)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                entry()
            except SystemExit as error:
                code = error.code
                record.update(exit_code=0 if code is None else code if isinstance(code, int) else 1,
                              system_exit_code=code if code is None or isinstance(code, int) else sanitize(code)[-4000:],
                              system_exit_message=sanitize(error)[-4000:], exception_type="SystemExit")
            except BaseException as error:
                record.update(exit_code=1, exception_type=type(error).__name__,
                              exception_message=sanitize(error)[-4000:])
    finally:
        sys.argv = original
    statuses = []
    for line in stdout.getvalue().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("status"), str):
            statuses.append(sanitize(value["status"])[:100])
    record.update(stdout_tail=sanitize(stdout.getvalue())[-4000:],
                  stderr_tail=sanitize(stderr.getvalue())[-4000:], reported_statuses=statuses)
    return record


def validate_gpu_admission(receipt, now, cuda_visible_devices):
    allocated = receipt.get("allocated_gpu_count")
    visible = receipt.get("visible_gpu_count", 1 if allocated == 1 else None)
    mask = receipt.get("cuda_visible_devices", "0" if allocated == 1 else None)
    if (receipt.get("owner") != "yashbajpai2027" or receipt.get("free_compute") is not True
            or receipt.get("run_id") != RUN_ID or receipt.get("gpu_name") != "Tesla T4"
            or type(allocated) is not int or allocated not in (1, 2)
            or visible != 1 or mask != "0" or cuda_visible_devices != "0"
            or receipt.get("remaining_quota_seconds", 0) < 3600
            or not 0 <= now - receipt.get("checked_at_epoch", 0) <= 600
            or receipt.get("backend_timeout_seconds") != 3600):
        raise ValueError("Fresh free T4 admission needs allocated1or2, visible/used1, mask0, backend3600; no fallback")
    return allocated


def single_used_t4(visible_count, gpu_name, world_size, cuda_visible_devices):
    if world_size != 1 or visible_count != 1 or "T4" not in gpu_name or cuda_visible_devices != "0":
        raise ValueError("Trial needs one process, CUDA_VISIBLE_DEVICES=0 and exactly one visible T4")
    return {"visible_gpu_count": visible_count, "used_gpu_count": 1,
            "gpu_name": gpu_name, "world_size": world_size, "cuda_visible_devices": cuda_visible_devices}
