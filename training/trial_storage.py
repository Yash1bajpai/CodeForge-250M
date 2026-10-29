"""Private HF conditional commits for the approved trial; imports/network are lazy."""
import hashlib
import json
import tempfile
from pathlib import Path

from training.step185_trial import ClaimError, REPO_ID, RUN_ID, WAIVER_ID, canonical, file_sha256


class HFTrialStore:
    def __init__(self):
        from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
        self.api = HfApi()
        self.add = CommitOperationAdd
        self.download = hf_hub_download

    def _private_parent(self):
        info = self.api.repo_info(repo_id=REPO_ID, repo_type="model")
        if not info.private or not info.sha:
            raise ClaimError("Trial storage must be a private repository with a pinned parent")
        return info.sha

    def _fresh_read(self, path, revision, cache):
        return Path(self.download(repo_id=REPO_ID, repo_type="model", filename=path,
                                  revision=revision, force_download=True, cache_dir=cache))

    def create_once(self, waiver_id, claim):
        if waiver_id != WAIVER_ID:
            raise ClaimError("Unapproved waiver storage key")
        path = "waivers/" + waiver_id + ".json"
        payload = canonical(claim)
        try:
            parent = self._private_parent()
            if path in self.api.list_repo_files(repo_id=REPO_ID, repo_type="model", revision=parent):
                raise ClaimError("Waiver already consumed")
            # CAS on the parent commit prevents two concurrent absent->created claims.
            commit = self.api.create_commit(
                repo_id=REPO_ID, repo_type="model", parent_commit=parent,
                operations=[self.add(path_in_repo=path, path_or_fileobj=payload)],
                commit_message="Consume approved bounded step185 waiver")
            revision = commit.oid
            if not revision:
                raise ClaimError("Claim has no immutable commit")
            with tempfile.TemporaryDirectory(prefix="cf-claim-readback-") as cache:
                readback = self._fresh_read(path, revision, cache).read_bytes()
            if readback != payload:
                raise ClaimError("Claim full readback mismatch")
        except Exception:
            # Includes timeout after a successful server commit: consumed, never retry.
            raise ClaimError("Conditional durable claim failed or is ambiguous") from None
        return {"claim": claim, "revision": revision,
                "sha256": hashlib.sha256(readback).hexdigest()}

    def publish_final(self, checkpoint, receipt):
        """One new prefix, immutable revision, fresh full-byte checkpoint/receipt readback."""
        checkpoint, receipt = Path(checkpoint), Path(receipt)
        prefix = "trials/" + RUN_ID + "/"
        paths = [prefix + "latest_checkpoint.pt", prefix + "waiver-guard-receipt.json"]
        try:
            parent = self._private_parent()
            existing = self.api.list_repo_files(repo_id=REPO_ID, repo_type="model", revision=parent)
            if any(name.startswith(prefix) for name in existing):
                raise ClaimError("Trial output prefix already exists; never overwrite")
            checkpoint_info = {"bytes": checkpoint.stat().st_size,
                               "sha256": file_sha256(checkpoint)}
            body = json.loads(receipt.read_text())
            body["checkpoint"] = checkpoint_info
            receipt.write_text(json.dumps(body, indent=2, sort_keys=True))
            expected = [(checkpoint_info["bytes"], checkpoint_info["sha256"]),
                        (receipt.stat().st_size, file_sha256(receipt))]
            commit = self.api.create_commit(
                repo_id=REPO_ID, repo_type="model", parent_commit=parent,
                operations=[self.add(path_in_repo=remote, path_or_fileobj=str(local))
                            for remote, local in zip(paths, (checkpoint, receipt))],
                commit_message="Preserve bounded step185 trial checkpoint and HOLD receipt")
            if not commit.oid:
                raise ClaimError("Trial backup has no immutable revision")
            with tempfile.TemporaryDirectory(prefix="cf-trial-readback-") as cache:
                for remote, (size, sha) in zip(paths, expected):
                    local = self._fresh_read(remote, commit.oid, cache)
                    if local.stat().st_size != size or file_sha256(local) != sha:
                        raise ClaimError("Trial full readback mismatch")
            return {"repo_id": REPO_ID, "prefix": prefix, "revision": commit.oid,
                    "checkpoint_bytes": expected[0][0], "checkpoint_sha256": expected[0][1],
                    "receipt_sha256": expected[1][1], "full_readback_verified": True}
        except Exception:
            raise ClaimError("Trial backup/readback failed; HOLD, no fallback or retry") from None
