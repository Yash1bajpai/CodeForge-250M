"""Phone integration checks and isolated cloud roundtrips; never starts jobs.

Smoke commands (use fresh report filenames in an existing external directory):
  python scripts/phone_control.py status --report ~/codeforge-storage/status-new.json
  python scripts/phone_control.py drive-probe --remote codeforge-drive:codeforge-storage
  python scripts/phone_control.py check

Kaggle credentials: ~/.kaggle/access_token, owner only. HF credentials: HF_TOKEN
or the standard Hugging Face token cache. Drive needs rclone OAuth configured
separately. Notebook status is not proof that Kaggle Secrets are ready to train.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.kaggle_drive_backup import OWNER, owner_api

PILOT = OWNER + "/codeforge-fresh-cpu-preparation-139cbd2a"
READINESS = OWNER + "/codeforge-hf-readiness-8cd44a26"
HF_OWNER = "Yash1bajpai"
HF_REPO = HF_OWNER + "/CodeForge-250M-rotation"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def safe_error(exc):
    """Never serialize an exception body, request, token, or signed URL."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(status, int) or not 100 <= status <= 599:
        status = None
    return {"exception": type(exc).__name__, "http_status": status}


def run_check(checks, name, function, required=True):
    # Client imports/authentication can print diagnostics or raise SystemExit.
    # Discard those streams rather than risk including credentials in reports.
    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                result = function()
        checks[name] = {"ok": True, "required": required, **result}
        return result
    except (Exception, SystemExit) as exc:
        checks[name] = {"ok": False, "required": required, "error": safe_error(exc)}
        return None
    finally:
        logging.disable(previous_logging)


def seconds(value):
    """Parse protobuf JSON seconds, including the SDK's failing zero '0s'."""
    if isinstance(value, bool):
        raise ValueError("Invalid duration")
    text = str(value)
    if not re.fullmatch(r"\d+(?:\.\d{1,9})?s?", text):
        raise ValueError("Invalid duration")
    number = Decimal(text.removesuffix("s"))
    if not number.is_finite() or number < 0:
        raise ValueError("Invalid duration")
    result = float(number)
    if not math.isfinite(result):
        raise ValueError("Invalid duration")
    return int(number) if number == number.to_integral_value() else result


def quota_status(api):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    captured = []
    # owner_api() resolves and validates the token-file credential. The SDK
    # builder accepts the resolved token, unlike KAGGLE_API_TOKEN's path syntax.
    client = api.build_kaggle_client_with_params(
        args=[], username=OWNER, api_token=api.config_values[api.CONFIG_NAME_TOKEN],
        response_processor=lambda response: captured.append(response.json()),
    )
    with client:
        try:
            client.kernels.kernels_api_client.get_accelerator_quota_statistics(
                ApiGetAcceleratorQuotaStatisticsRequest())
        except ValueError:
            if not captured:
                raise
            # 2.2.2 can fail deserialization on valid zero durations. Only a
            # captured successful HTTP JSON response is eligible for parsing.
    if not captured or not isinstance(captured[-1], dict):
        raise ValueError("Missing quota response")
    raw = captured[-1]
    result = {}
    for name, key in (("gpu", "gpuQuota"), ("tpu", "tpuQuota")):
        quota = raw[key]
        if not isinstance(quota, dict):
            raise ValueError("Invalid quota response")
        result[name] = {
            output: seconds(quota[field]) for output, field in (
                ("used_seconds", "timeUsed"), ("reserved_seconds", "timeReserved"),
                ("allowed_seconds", "totalTimeAllowed"),
                ("minimum_seconds", "minimumTimeAllowed"))
        }
    refresh = raw.get("quotaRefreshTime")
    if refresh is not None:
        result["refresh_at_utc"] = datetime.fromisoformat(
            refresh.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    return result


def validate_kernel(kernel):
    if not re.fullmatch(re.escape(OWNER) + r"/[a-z0-9][a-z0-9-]*", kernel):
        raise ValueError("Only owner notebooks are allowed")
    return kernel.split("/", 1)[1]


def notebook_status(api, kernel):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    slug = validate_kernel(kernel)
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = OWNER, slug
        metadata = client.kernels.kernels_api_client.get_kernel(request).metadata
    version = metadata.current_version_number
    if not isinstance(version, int) or version < 1:
        raise ValueError("No saved notebook version")
    state = api.kernels_status(kernel).status.name
    if not re.fullmatch(r"[A-Z_]+", state):
        raise ValueError("Unknown notebook state")
    return {"kernel": kernel, "saved_version": version, "status": state,
            "private": bool(metadata.is_private), "gpu_enabled": bool(metadata.enable_gpu),
            "tpu_enabled": bool(metadata.enable_tpu),
            "checkpoint_credentials_verified": False}


def hf_identity(api):
    name = api.whoami(token=True)["name"]
    if not isinstance(name, str) or name.casefold() != HF_OWNER.casefold():
        raise ValueError("Unexpected Hugging Face identity")
    return {"authenticated_owner": HF_OWNER}


def hf_repo_status(api):
    info = api.model_info(HF_REPO, token=True, files_metadata=False)
    if info.private is not True:
        raise ValueError("Private repository read was not verified")
    return {"repo": HF_REPO, "private": True, "metadata_read_verified": True,
            "checkpoint_downloaded": False, "checkpoint_deserialized": False}


def rclone(*args):
    # No shell and no stderr forwarding: OAuth errors may contain secrets.
    return subprocess.run(
        ["rclone", *args], check=True, capture_output=True, text=True, timeout=150)


def drive_configuration():
    configured = "codeforge-drive:" in rclone("listremotes").stdout.splitlines()
    return {"codeforge_drive_configured": configured, "oauth_verified": False,
            "cloud_readback_verified": False}


def status(pilot=PILOT, readiness=READINESS):
    checks = {}
    authenticated = []

    def kaggle_identity():
        api = owner_api()  # Performs live token introspection, not username trust.
        authenticated.append(api)
        return {"authenticated_owner": OWNER, "auth_method": "ACCESS_TOKEN"}

    run_check(checks, "storage", lambda: {"free_bytes": shutil.disk_usage(Path.home()).free})
    run_check(checks, "kaggle_identity", kaggle_identity)
    for name, operation in (
        ("kaggle_quota", lambda api: quota_status(api)),
        ("fresh_pilot", lambda api: notebook_status(api, pilot)),
        ("readiness_notebook", lambda api: notebook_status(api, readiness)),
    ):
        if authenticated:
            run_check(checks, name, lambda op=operation: op(authenticated[0]))
        else:
            checks[name] = {"ok": False, "required": True,
                            "error": {"exception": "DependencyUnavailable", "http_status": None}}
    hf_clients = []

    def check_hf_identity():
        from huggingface_hub import HfApi
        api = HfApi()
        result = hf_identity(api)
        hf_clients.append(api)
        return result

    run_check(checks, "hf_identity", check_hf_identity)
    if hf_clients:
        run_check(checks, "hf_private_repo", lambda: hf_repo_status(hf_clients[0]))
    else:
        checks["hf_private_repo"] = {"ok": False, "required": True,
                                     "error": {"exception": "DependencyUnavailable", "http_status": None}}
    run_check(checks, "drive_configuration", drive_configuration, required=False)
    return {"command": "status", "timestamp_utc": timestamp(),
            "model_training_started": False,
            "training_readiness_verified": False,
            "ok": all(item["ok"] for item in checks.values() if item["required"]),
            "checks": checks}


def validate_remote(remote):
    # Require a named rclone remote and a bounded, non-root relative folder.
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", remote):
        raise ValueError("Invalid Drive remote")
    name, folder = remote.split(":", 1)
    if any(part in (".", "..") for part in folder.split("/")):
        raise ValueError("Invalid Drive remote folder")
    return name + ":"


def drive_probe(remote, runner=rclone):
    report = {"command": "drive-probe", "timestamp_utc": timestamp(), "ok": False,
              "model_training_started": False, "cloud_readback_verified": False,
              "own_object_removed": False, "checks": {}}
    checks = report["checks"]

    def remote_configuration():
        name = validate_remote(remote)
        if name not in runner("listremotes").stdout.splitlines():
            raise ValueError("Drive remote is not configured")
        return {"remote": remote}

    if run_check(checks, "remote_configuration", remote_configuration) is None:
        return report
    object_path = remote + "/logs/migration-checks/" + uuid.uuid4().hex + "/phone-probe.txt"
    report["object"] = object_path
    flags = ("--retries", "1", "--low-level-retries", "1", "--contimeout", "15s", "--timeout", "60s")
    with tempfile.TemporaryDirectory(prefix="codeforge-drive-probe-") as directory:
        source, readback = Path(directory) / "source.txt", Path(directory) / "readback.txt"
        payload = ("CodeForge phone migration readback " + uuid.uuid4().hex + "\n").encode("ascii")
        source.write_bytes(payload)
        source_hash = hashlib.sha256(payload).hexdigest()

        def upload():
            runner("copyto", str(source), object_path, *flags)
            return {"bytes": len(payload), "sha256": source_hash}

        if run_check(checks, "upload", upload) is None:
            return report

        def download_and_compare():
            runner("copyto", object_path, str(readback), *flags)
            returned = readback.read_bytes()
            digest = hashlib.sha256(returned).hexdigest()
            if len(returned) != len(payload) or digest != source_hash:
                raise ValueError("Drive readback mismatch")
            return {"object": object_path, "bytes": len(returned),
                    "source_sha256": source_hash, "readback_sha256": digest,
                    "verified_at_utc": timestamp(), "verification": "cloud-download-sha256"}

        if run_check(checks, "cloud_readback", download_and_compare) is None:
            return report
        report["cloud_readback_verified"] = True

        def remove_own_object():
            # Only this unique test object, and only after verified readback.
            runner("deletefile", object_path, *flags)
            return {"object_removed": object_path}

        if run_check(checks, "cleanup", remove_own_object) is None:
            return report
        report["own_object_removed"] = True
        report["ok"] = True
    return report


def _controller_tool(name):
    if shutil.which(name) is None:
        raise FileNotFoundError("Required controller tool is not installed")
    return {"installed": True}


def _github_identity():
    result = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"], stdin=subprocess.DEVNULL,
        check=True, capture_output=True, text=True, timeout=30)
    if result.stdout.strip().casefold() != HF_OWNER.casefold():
        raise ValueError("Unexpected GitHub identity")
    return {"authenticated_owner": HF_OWNER}


def _hf_write_probe():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    hf_identity(api)
    checks = {}
    object_path = "logs/phone-checks/" + uuid.uuid4().hex + ".txt"
    payload = ("CodeForge controller check " + uuid.uuid4().hex + "\n").encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()

    def upload():
        commit = api.upload_file(
            path_or_fileobj=payload, path_in_repo=object_path, repo_id=HF_REPO,
            repo_type="model", token=True, commit_message="Verify phone controller access")
        revision = commit.oid
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError("Missing immutable upload revision")
        return {"revision": revision, "bytes": len(payload), "sha256": digest}

    def readback(revision):
        downloaded = hf_hub_download(
            repo_id=HF_REPO, filename=object_path, repo_type="model", revision=revision,
            token=True, force_download=True)
        returned = Path(downloaded).read_bytes()
        returned_hash = hashlib.sha256(returned).hexdigest()
        if len(returned) != len(payload) or returned_hash != digest:
            raise ValueError("Hugging Face readback mismatch")
        return {"revision": revision, "bytes": len(returned), "sha256": returned_hash,
                "verification": "immutable-revision-download-sha256"}

    def cleanup():
        api.delete_file(
            path_in_repo=object_path, repo_id=HF_REPO, repo_type="model",
            token=True, commit_message="Remove own phone controller check")
        return {"own_object_removed": True}

    # Even an upload with an interrupted response may have committed the file.
    # Cleanup is restricted to this invocation's unique, tiny object.
    try:
        uploaded = run_check(checks, "upload", upload)
        if uploaded is not None:
            run_check(checks, "immutable_readback", lambda: readback(uploaded["revision"]))
    finally:
        run_check(checks, "cleanup", cleanup)
    return {"ok": all(item["ok"] for item in checks.values()),
            "repo": HF_REPO, "object": object_path, "checks": checks,
            "checkpoint_downloaded": False, "checkpoint_deserialized": False}


def _model_probe(model):
    marker = "CF_CHECK_" + uuid.uuid4().hex
    result = subprocess.run(
        ["omp", "--no-session", "--no-tools", "--model", model,
         "--max-time", "80s", "-p", "Reply with exactly this line and nothing else: " + marker],
        cwd=PROJECT, stdin=subprocess.DEVNULL, check=True,
        capture_output=True, text=True, timeout=90)
    # A prompt/error that merely mentions the marker is not a provider reply.
    if marker not in result.stdout.splitlines():
        raise ValueError("Exact provider reply was not received")
    return {"model": model, "live_request_passed": True, "exact_marker_received": True,
            "tools_enabled": False, "session_saved": False}


def _codex_check():
    path = Path.home() / "codeforge-storage/logs/provider-verification.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        gate = receipt.get("openai_codex")
        if not isinstance(gate, dict):
            raise ValueError("Missing provider security gate")
    except (Exception, SystemExit) as exc:
        return {"ok": False, "required": True, "state": "ACTION",
                "security_state": "unknown", "live_request_attempted": False,
                "action": "Owner must establish private Codex revocation and fresh-login evidence.",
                "error": safe_error(exc)}
    flags = ("provider_revocation_verified", "fresh_login_after_revocation_verified")
    verified = {name: gate.get(name) is True for name in flags}
    blocked = gate.get("revocation_required") is True or any(
        gate.get(name) is False for name in flags)
    security_verified = gate.get("revocation_required") is False and all(verified.values())
    evidence = {"required": True, "owner_security_verified": security_verified,
                "provider_revocation_verified": verified[flags[0]],
                "fresh_login_after_revocation_verified": verified[flags[1]],
                "live_request_attempted": False}
    if blocked or not security_verified:
        return {**evidence, "ok": False, "state": "ACTION",
                "security_state": "BLOCKED" if blocked else "unknown",
                "action": "Owner must revoke the exposed authorization, then verify a fresh login; no Codex request made."}
    checks = {}
    run_check(checks, "request", lambda: _model_probe("openai-codex/gpt-6.1-sol"))
    return {**evidence, "ok": checks["request"]["ok"], "live_request_attempted": True,
            "security_state": "owner-verified", "checks": checks}


def check(pilot=PILOT, readiness=READINESS,
          remote="codeforge-drive:codeforge-storage"):
    checks = {}
    for name in ("python", "git", "gh", "kaggle", "hf", "rclone", "omp", "bun", "tmux"):
        run_check(checks, "tool_" + name, lambda tool=name: _controller_tool(tool))
    run_check(checks, "github_identity", _github_identity)
    live_status = status(pilot, readiness)
    checks.update(live_status["checks"])
    # A Drive roundtrip below, not a local configuration entry, verifies OAuth.
    run_check(checks, "hf_write_readback_delete", _hf_write_probe)
    run_check(checks, "drive_write_readback_delete", lambda: drive_probe(remote))
    run_check(checks, "google_antigravity",
              lambda: _model_probe("google-antigravity/gemini-3.8-flash"))
    checks["openai_codex"] = _codex_check()
    return {"command": "check", "timestamp_utc": timestamp(),
            "ok": all(item["ok"] for item in checks.values() if item["required"]),
            "model_training_started": False, "training_readiness_verified": False,
            "training_note": "No model training launched. Kaggle notebook HF_TOKEN binding and production gates remain separate.",
            "checks": checks}


def _print_check(report, report_path):
    for name, item in report.get("checks", {}).items():
        state = item.get("state") or ("PASS" if item["ok"] else "FAIL")
        detail = item.get("action", "")
        if not detail and not item["ok"]:
            error = item.get("error", {})
            detail = error.get("exception", "See sanitized report for failed subchecks")
            if error.get("http_status") is not None:
                detail += " (HTTP " + str(error["http_status"]) + ")"
        if name == "openai_codex" and item.get("security_state") == "BLOCKED":
            detail = "BLOCKED: " + detail
        print(f"{state:6} {name}" + (": " + detail if detail else ""))
    if "error" in report:
        print("FAIL   check: " + report["error"]["exception"])
    print("Private JSON report: " + str(report_path))
    print("Overall: " + ("PASS — controller services verified." if report["ok"]
                        else "ATTENTION REQUIRED — one or more services need action."))
    print("No model training launched; training readiness NOT established.")
    print("Kaggle notebook HF_TOKEN binding and production gates remain separate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status_parser = commands.add_parser("status", help="Read-only Kaggle/HF integration status")
    status_parser.add_argument("--pilot", default=PILOT, help="Owner's saved fresh CPU pilot notebook")
    status_parser.add_argument("--readiness", default=READINESS, help="Owner's saved HF readiness notebook")
    drive_parser = commands.add_parser("drive-probe", help="Verify one tiny isolated Drive object")
    drive_parser.add_argument("--remote", required=True, help="Named remote and folder, e.g. codeforge-drive:codeforge-storage")
    check_parser = commands.add_parser("check", help="Live controller checks; never launches training")
    check_parser.add_argument("--pilot", default=PILOT)
    check_parser.add_argument("--readiness", default=READINESS)
    check_parser.add_argument("--remote", default="codeforge-drive:codeforge-storage")
    for command in (status_parser, drive_parser, check_parser):
        command.add_argument("--report", type=Path, help="New JSON file in an existing directory; never overwrite")
    args = parser.parse_args()
    stream = None
    report_path = args.report.expanduser() if args.report is not None else None
    try:
        if args.command == "check":
            if report_path is None:
                directory = Path.home() / "codeforge-storage/logs"
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                report_path = directory / ("cf-check-" + uuid.uuid4().hex + ".json")
            descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        elif args.report is not None:
            stream = args.report.expanduser().open("x", encoding="utf-8")
        if args.command == "check":
            report = check(args.pilot, args.readiness, args.remote)
        else:
            report = status(args.pilot, args.readiness) if args.command == "status" else drive_probe(args.remote)
    except (Exception, SystemExit) as exc:
        report = {"command": args.command, "timestamp_utc": timestamp(), "ok": False,
                  "model_training_started": False, "error": safe_error(exc)}
    try:
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if stream is not None:
            stream.write(encoded)
            stream.flush()
        if args.command == "check":
            _print_check(report, report_path if stream is not None else "(not saved)")
        else:
            print(encoded, end="")
    except Exception as exc:
        failure = {"ok": False, "model_training_started": False, "error": safe_error(exc)}
        if args.command == "check":
            _print_check(failure, "(not saved)")
        else:
            print(json.dumps(failure))
        return 1
    finally:
        if stream is not None:
            stream.close()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
