"""CPU-only artifact for checking Kaggle -> PC -> Drive desktop transfers."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main():
    output = Path(os.environ.get("CF_STORAGE_CHECK_OUTPUT", "/kaggle/working"))
    output.mkdir(parents=True, exist_ok=True)
    payload = (
        "CODEFORGE-KAGGLE-DRIVE-CHECK\n"
        "Owner: yashbajpai2027\n"
        "Created on Kaggle CPU; no training, GPU, secrets or external datasets.\n"
    ).encode("utf-8")
    artifact = output / "kaggle-drive-check.txt"
    artifact.write_bytes(payload)
    manifest = {
        "owner": "yashbajpai2027",
        "purpose": "storage-smoke-check",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": artifact.name, "bytes": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest()}],
    }
    (output / "storage-check-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
