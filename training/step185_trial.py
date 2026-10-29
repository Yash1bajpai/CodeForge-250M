"""The single approved step185 trial. Pure metadata/state; no torch or network imports."""
import copy
import hashlib
import json
import time
from pathlib import Path

from training.validation_history import RULE_VERSION, STOP_REASON, ValidationHistory

APPROVAL = "Yash WhatsApp 2026-10-29 18:31:52 IST"
WAIVER_ID = "step185-yash-20261029-183152"
RUN_ID = "fresh-step185-trial-20261029-183152"
REPO_ID = "Yash1bajpai/CodeForge-250M-rotation"
BINDING = {
    "checkpoint_sha256": "2b69e2c21778bb494e281285e852edb4ac0b1e4d89fb55bc58b5a3990587f5b5",
    "checkpoint_step": 185, "sample_cursor": 47232,
    "data_fingerprint": "46f9c60037993a32dd3ecb28acb4f388b1d3919adda221f07a2f4e5af8e248bc",
    "validation_identity": "af30f98d37e10b39846363903d5a2b01a626ff929746b0e3a05fc689d6ee9a1d",
    "tokenizer_sha256": "5d8adf0601c14398390fc4ffcde2a4f0c6152b149ec35c8eedd593d15de454dc",
    "guard_rule_version": RULE_VERSION,
}
LIMITS = {"max_attempted_updates": 8, "max_applied_updates": 8,
          "update_cutoff_seconds": 2100, "job_timeout_seconds": 3600,
          "val_steps": 1, "val_batches": 1, "world_size": 1, "visible_gpus": 1,
          "gpu_name": "Tesla T4"}
ORIGINAL_STOP = {"reason": STOP_REASON, "step": 185}
ORIGINAL_HISTORY = [
    {"step": 182, "loss": 6.0613, "precision": "rounded_4dp"},
    {"step": 183, "loss": 6.0621, "precision": "rounded_4dp"},
    {"step": 184, "loss": 6.0847, "precision": "rounded_4dp"},
    {"step": 185, "loss": 6.085826873779297, "precision": "full"},
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def approved_waiver():
    return {"schema_version": 1, "waiver_id": WAIVER_ID, "run_id": RUN_ID,
            "approval_reference": APPROVAL, "binding": copy.deepcopy(BINDING),
            "stop_id": digest({"rule_version": RULE_VERSION, "stop": ORIGINAL_STOP,
                               "history": ORIGINAL_HISTORY}), "limits": copy.deepcopy(LIMITS)}


class ClaimError(RuntimeError):
    """An unsuccessful or ambiguous claim is never retried automatically."""


class TrialLimit(RuntimeError):
    pass


class Step185Trial:
    def __init__(self, waiver, *, start_epoch, clock=time.time, monotonic=time.monotonic):
        if canonical(waiver) != canonical(approved_waiver()):
            raise ValueError("Only the exact approved step185 waiver is permitted")
        now = clock()
        if type(start_epoch) not in (int, float) or not 0 <= now - start_epoch < 2100:
            raise ValueError("Invalid/expired trial job-start time")
        self.waiver = copy.deepcopy(waiver)
        self.start_epoch = start_epoch
        self.clock = clock
        self.monotonic = monotonic
        self.update_deadline = monotonic() + 2100 - (now - start_epoch)
        self.claim_receipt = None
        self.claim_attempted = False
        self.attempted = 0
        self.applied = 0
        self.pending_attempt = False
        self.terminal_reason = None

    def permits_stop(self, stop, history):
        return (stop == ORIGINAL_STOP and history == ORIGINAL_HISTORY
                and canonical(self.waiver) == canonical(approved_waiver()))

    def claim(self, store):
        if self.claim_attempted:
            raise ClaimError("Waiver claim already attempted; no reuse")
        self.claim_attempted = True
        if (self.clock() >= self.start_epoch + 2100
                or self.monotonic() >= self.update_deadline):
            self.terminal_reason = "UPDATE_WALL_LIMIT_HOLD"
            raise TrialLimit("No claim after the trial update deadline")
        claim = {**copy.deepcopy(self.waiver), "state": "consumed",
                 "claimed_before_optimizer_update": True, "consumed_at_epoch": self.clock()}
        try:
            receipt = store.create_once(WAIVER_ID, claim)
            if (not isinstance(receipt, dict) or canonical(receipt.get("claim")) != canonical(claim)
                    or receipt.get("sha256") != digest(claim)
                    or not isinstance(receipt.get("revision"), str) or not receipt["revision"]):
                raise ClaimError("Durable claim readback was not established")
        except Exception:
            self.terminal_reason = "CLAIM_FAILED_OR_AMBIGUOUS_HOLD"
            raise ClaimError("Durable waiver claim failed or is ambiguous; abort without reuse") from None
        self.claim_receipt = copy.deepcopy(receipt)
        return copy.deepcopy(receipt)

    def require_claimed(self):
        if self.claim_receipt is None or self.terminal_reason is not None:
            raise ClaimError("No active consumed claim for this bounded run")

    def _check_budget(self):
        self.require_claimed()
        if (self.clock() >= self.start_epoch + 2100
                or self.monotonic() >= self.update_deadline):
            self.terminal_reason = "UPDATE_WALL_LIMIT_HOLD"
            raise TrialLimit("No optimizer update after35 minutes")
        if self.attempted >= 8 or self.applied >= 8:
            self.terminal_reason = "UPDATE_COUNT_LIMIT_HOLD"
            raise TrialLimit("Bounded trial update limit reached")

    def begin_attempt(self):
        self._check_budget()
        if self.pending_attempt:
            raise TrialLimit("Previous update attempt has not completed")
        self.attempted += 1
        self.pending_attempt = True

    def before_optimizer_step(self):
        self.require_claimed()
        if not self.pending_attempt:
            raise TrialLimit("Optimizer step without a counted attempt")
        if (self.clock() >= self.start_epoch + 2100
                or self.monotonic() >= self.update_deadline):
            self.terminal_reason = "UPDATE_WALL_LIMIT_HOLD"
            raise TrialLimit("Discard accumulated gradients: update deadline reached")
        if self.applied >= 8:
            raise TrialLimit("Applied update limit reached")

    def complete_attempt(self, applied):
        self.require_claimed()
        if not self.pending_attempt or type(applied) is not bool:
            raise TrialLimit("Invalid update completion")
        self.pending_attempt = False
        self.applied += int(applied)
        if self.applied > 8:
            raise TrialLimit("Applied update cap exceeded")

    def new_guard_stop(self):
        self.terminal_reason = "NEW_VALIDATION_GUARD_HOLD"

    def finish(self, reason="BOUNDED_TRIAL_COMPLETE_HOLD"):
        self.terminal_reason = self.terminal_reason or reason
        return self.snapshot()

    def snapshot(self):
        return {"schema_version": 1, "waiver": copy.deepcopy(self.waiver),
                "claim_receipt": copy.deepcopy(self.claim_receipt),
                "claim_attempted": self.claim_attempted, "start_epoch": self.start_epoch,
                "attempted_updates": self.attempted, "applied_updates": self.applied,
                "pending_attempt": self.pending_attempt,
                "status": self.terminal_reason or "BOUNDED_RUN_IN_PROGRESS",
                "automatic_resume_authorized": False}


def resume_preflight(checkpoint, *, validation_interval, tokenizer_sha256,
                     checkpoint_sha256, sidecar_path=None, waiver=None, start_epoch=None,
                     clock=time.time, monotonic=time.monotonic):
    """Shared by train.py and CPU checks, before devices/models/optimizers exist."""
    resume = checkpoint.get("resume_state") or {}
    if resume.get("step185_trial") is not None:
        raise ClaimError("A consumed trial result is HOLD; this waiver cannot authorize a new process")
    trial = None
    if waiver is not None:
        trial = Step185Trial(waiver, start_epoch=start_epoch, clock=clock, monotonic=monotonic)
        actual = {"checkpoint_sha256": checkpoint_sha256, "checkpoint_step": checkpoint.get("step"),
                  "sample_cursor": resume.get("sample_cursor"),
                  "data_fingerprint": resume.get("fingerprint"),
                  "validation_identity": (resume.get("extension") or {}).get("validation_identity"),
                  "tokenizer_sha256": tokenizer_sha256, "guard_rule_version": RULE_VERSION}
        if actual != BINDING or validation_interval != 1:
            raise ValueError("Step185 waiver checkpoint/tokenizer/data binding mismatch")
    guard = ValidationHistory.restore(
        checkpoint, validation_interval, sidecar_path=sidecar_path,
        checkpoint_sha256=checkpoint_sha256, tokenizer_sha256=tokenizer_sha256, waiver=trial)
    return guard, trial


def load_checkpoint(path):
    # Only invoked on the explicitly approved cloud CPU/GPU path, never by phone tests.
    import torch
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def checkpoint_metadata(checkpoint):
    return {key: copy.deepcopy(checkpoint.get(key))
            for key in ("step", "loss", "val_loss", "resume_state")}
