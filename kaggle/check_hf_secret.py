"""CPU-only HF credential diagnosis; never prints tokens or arbitrary errors."""
import argparse
import io
import json
import os
import re
import tempfile
import uuid
from pathlib import Path


def safe_failure(exc):
    text = str(exc).lower()
    if any(phrase in text for phrase in ("no user secrets", "secret not found", "not attached", "not bound")):
        reason = "secret-not-available-to-this-notebook"
    elif "permission" in text or "unauthorized" in text or "forbidden" in text:
        reason = "secret-or-repository-permission-denied"
    elif "timeout" in text or "timed out" in text:
        reason = "backend-timeout"
    else:
        reason = "backend-request-failed"
    return {"type": type(exc).__name__, "reason": reason,
            "http_status": getattr(getattr(exc, "response", None), "status_code", None),
            "backend_status_codes": sorted(set(re.findall(r"\b(?:400|401|403|404|429|500|502|503|504)\b", text)))}


def check():
    import requests
    result = {"purpose": "kaggle-HF_TOKEN-readiness", "training_started": False}
    try:
        with requests.get("https://huggingface.co/api/models/gpt2", timeout=30) as response:
            result["public_hf_https_status"] = response.status_code
    except Exception as exc:
        result["public_hf_error"] = safe_failure(exc)
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if not token:
            raise ValueError("No user secrets token returned")
    except Exception as exc:
        result["HF_TOKEN"] = "UNAVAILABLE"
        result["error"] = safe_failure(exc)
        result["checkpoint_credentials_ready"] = False
        return result
    result["HF_TOKEN"] = "AVAILABLE"
    remote = None
    uploaded = False
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=token)
        who = api.whoami()
        result["hf_username"] = who.get("name")
        if who.get("name") != "Yash1bajpai":
            raise ValueError("Unexpected checkpoint owner identity")
        repo = "Yash1bajpai/CodeForge-250M-rotation"
        api.model_info(repo, files_metadata=False)
        remote = "credential-checks/" + uuid.uuid4().hex + "/readiness.txt"
        payload = ("CodeForge CPU checkpoint readiness " + uuid.uuid4().hex).encode()
        revision = api.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=remote,
                                   repo_id=repo, commit_message="Verify CPU checkpoint readiness")
        uploaded = True
        with tempfile.TemporaryDirectory() as cache:
            path = hf_hub_download(repo_id=repo, filename=remote, revision=revision.oid,
                                   token=token, cache_dir=cache)
            if Path(path).read_bytes() != payload:
                raise ValueError("Checkpoint readiness readback mismatch")
        result["private_repo_write_readback"] = "PASS"
        result["checkpoint_credentials_ready"] = True
    except Exception as exc:
        result["error"] = safe_failure(exc)
        result["checkpoint_credentials_ready"] = False
    finally:
        if uploaded:
            try:
                api.delete_file(path_in_repo=remote, repo_id=repo,
                                commit_message="Remove CPU checkpoint readiness artifact")
                result["probe_cleanup"] = "PASS"
            except Exception as exc:
                result["cleanup_error"] = safe_failure(exc)
                result["checkpoint_credentials_ready"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
