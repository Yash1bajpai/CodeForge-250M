"""Download an expected current owner notebook version into a fresh Drive folder.

This is a PC-mediated backup of saved Kaggle outputs, not a Drive mount inside
Kaggle. It never starts a notebook, reads a checkpoint with pickle, or deletes
remote/local artifacts. A manifest records locally verified bytes, not cloud sync.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

OWNER = "yashbajpai2027"


def output_path(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or ":" in name or not relative.parts
            or relative.parts[0] == "backup-manifest.json"):
        raise ValueError("Unsafe or reserved Kaggle output path")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Kaggle output escapes destination")
    return target


def download_verified(url, target, expected_bytes=None):
    import requests
    temporary = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    try:
        with requests.get(url, stream=True, timeout=(30, 300),
                          headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("Cannot verify an encoded artifact's transport length")
            if expected_bytes is None:
                length = response.headers.get("Content-Length")
                if length is None:
                    raise ValueError("Artifact response has no verifiable transport length")
                expected_bytes = int(length)
            with temporary.open("xb") as stream:
                for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if block:
                        stream.write(block)
                        digest.update(block)
                        count += len(block)
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Never include signed download URLs or request headers in diagnostics.
        raise RuntimeError(f"Artifact download failed: {type(exc).__name__}, HTTP {status}") from None
    if count != expected_bytes:
        raise ValueError(f"Artifact size mismatch: {target.name}: {count} != {expected_bytes}")
    readback = hashlib.sha256()
    with temporary.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            readback.update(block)
    if readback.digest() != digest.digest():
        raise ValueError(f"Artifact readback mismatch: {target.name}")
    os.replace(temporary, target)
    return {"bytes": count, "sha256": digest.hexdigest()}


def owner_api():
    token = Path.home() / ".kaggle" / "access_token"
    if not token.is_file():
        raise ValueError("Owner Kaggle access-token file is missing")
    for name in ("KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        os.environ.pop(name, None)
    os.environ["KAGGLE_API_TOKEN"] = str(token)
    os.environ["KAGGLE_USERNAME"] = OWNER
    import kaggle
    api = kaggle.api
    if (api.config_values.get(api.CONFIG_NAME_USER) != OWNER
            or str(api.config_values.get(api.CONFIG_NAME_AUTH_METHOD)) != "ACCESS_TOKEN"):
        raise ValueError("Refusing backup: authenticated identity is not the owner's access token")
    return api


def backup(kernel, version, destination):
    parts = kernel.split("/")
    if len(parts) != 2 or parts[0] != OWNER or not parts[1]:
        raise ValueError("Only yashbajpai2027/<notebook> outputs are allowed")
    if version < 1:
        raise ValueError("A positive saved notebook version is required")
    if not destination.parent.is_dir():
        raise ValueError("Destination parent must exist on the mounted Drive")
    if destination.exists():
        raise ValueError("Destination already exists; choose a fresh backup folder")
    api = owner_api()
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelRequest, ApiListKernelSessionOutputRequest,
    )
    outputs = []
    status = str(api.kernels_status(kernel).status).split(".")[-1]
    if status not in ("COMPLETE", "ERROR"):
        raise ValueError("Only saved COMPLETE/ERROR outputs can be backed up; active notebooks are refused")
    with api.build_kaggle_client() as client:
        identity = ApiGetKernelRequest()
        identity.user_name, identity.kernel_slug = OWNER, parts[1]
        if client.kernels.kernels_api_client.get_kernel(identity).metadata.current_version_number != version:
            raise ValueError("Current notebook version does not match --version")
        token = ""
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = OWNER, parts[1]
            request.page_size, request.page_token = 200, token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            outputs.extend((item.file_name, item.url) for item in response.files)
            token = response.next_page_token
            if not token:
                break
        if client.kernels.kernels_api_client.get_kernel(identity).metadata.current_version_number != version:
            raise ValueError("Notebook version changed while listing saved outputs")
    names = [name for name, _ in outputs]
    # Session-output names preserve directories; the files-metadata endpoint
    # reports basenames only and legitimately contains duplicates.
    if not names or len(names) != len(set(os.path.normcase(name) for name in names)):
        raise ValueError("Saved output listing is empty or has duplicate relative paths")
    # Validate all names before writing any output bytes.
    targets = [(name, url, output_path(destination, name)) for name, url in outputs]
    destination.mkdir()
    records = []
    for name, url, target in targets:
        record = download_verified(url, target)
        records.append({"path": name, **record})
        print(f"Verified {name}: {record['bytes']} bytes", flush=True)
    report = {"owner": OWNER, "kernel": kernel, "version": version, "source_status": status,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "verification": "http-length-and-local-sha256-readback",
              "cloud_sync_verified": False, "files": records}
    manifest = destination / "backup-manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--version", type=int, required=True,
                        help="Expected current saved version; refuses a newer or running notebook")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = backup(args.kernel, args.version, args.destination)
    except Exception as exc:
        # HTTP exception messages can contain signed URLs; keep these private.
        if isinstance(exc, (ValueError, RuntimeError)):
            print(f"Backup failed: {exc}", file=sys.stderr)
        else:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            print(f"Backup failed: {type(exc).__name__}, HTTP {status}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"files": len(report["files"]), "destination": str(args.destination),
                      "cloud_sync_verified": False}, indent=2))


if __name__ == "__main__":
    main()
