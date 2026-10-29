"""Persist the original strict four-reading guard; never clear a recorded stop."""
import copy
import json
import math
from pathlib import Path


RULE_VERSION = "strict-four-increasing-v1"
STOP_REASON = "VAL_LOSS_DIVERGENCE"


class ValidationGuardStopped(RuntimeError):
    """A persisted divergence stop is not continuation authorization."""


class ValidationHistory:
    def __init__(self, validation_interval):
        if type(validation_interval) is not int or validation_interval < 1:
            raise ValueError("Validation interval must be a positive integer")
        self.interval = validation_interval
        self.history = []
        self.stop = None

    def _rising(self):
        last4 = [reading["loss"] for reading in self.history]
        return (len(last4) == 4
                and last4[0] < last4[1] < last4[2] < last4[3])

    def require_resume_allowed(self):
        if self.stop is not None:
            raise ValidationGuardStopped(
                f"{STOP_REASON} at step {self.stop['step']} remains active; "
                "restoring history does not authorize continuation")

    def record(self, step, loss):
        self.require_resume_allowed()
        expected_step = self.history[-1]["step"] + self.interval if self.history else self.interval
        if type(step) is not int or step != expected_step:
            raise ValueError("Validation history cannot skip or duplicate a reading")
        if type(loss) not in (int, float):
            raise ValueError("Validation loss must be numeric")
        self.history.append({"step": step, "loss": float(loss), "precision": "full"})
        self.history = self.history[-4:]
        if self._rising():
            self.stop = {"reason": STOP_REASON, "step": step}
        return self.stop is not None

    def snapshot(self):
        return {"schema_version": 1, "rule_version": RULE_VERSION,
                "validation_interval": self.interval,
                "history": copy.deepcopy(self.history), "stop": copy.deepcopy(self.stop)}

    @classmethod
    def restore(cls, checkpoint, validation_interval, *, sidecar_path=None,
                checkpoint_sha256=None, tokenizer_sha256=None):
        """Read metadata only; the caller hashes/loads the actual checkpoint.

        Embedded history travels inside resume_state through ordinary saves and
        dataset handoffs. Older checkpoints require an explicit bound sidecar;
        absence is not permission to restart the history window.
        """
        result = cls(validation_interval)
        step = checkpoint.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("Invalid checkpoint step for validation history")
        resume = checkpoint.get("resume_state") or {}
        saved = resume.get("validation_guard")
        if sidecar_path is not None:
            if saved is not None:
                raise ValueError("A sidecar cannot override embedded validation history")
            sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
            hashes = (checkpoint_sha256, tokenizer_sha256, resume.get("fingerprint"))
            if any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in hashes):
                raise ValueError("A history sidecar requires checkpoint, tokenizer and data hashes")
            if type(resume.get("sample_cursor")) is not int or resume["sample_cursor"] < 0:
                raise ValueError("A history sidecar requires an exact saved sample cursor")
            expected_binding = {
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": step,
                "sample_cursor": resume.get("sample_cursor"),
                "data_fingerprint": resume.get("fingerprint"),
                "validation_identity": (resume.get("extension") or {}).get("validation_identity"),
                "tokenizer_sha256": tokenizer_sha256,
                "guard_rule_version": RULE_VERSION,
            }
            if (sidecar.get("format_version") != 1
                    or sidecar.get("binding") != expected_binding):
                raise ValueError("Validation history sidecar binding mismatch")
            saved = sidecar.get("validation_guard")
        expected_count = min(4, step // validation_interval)
        if saved is None:
            if expected_count == 0 and checkpoint.get("val_loss") is None:
                return result
            raise ValueError("Missing validation history; an audited bound sidecar is required")
        if (not isinstance(saved, dict) or saved.get("schema_version") != 1
                or saved.get("rule_version") != RULE_VERSION
                or saved.get("validation_interval") != validation_interval):
            raise ValueError("Validation history rule or cadence changed")
        history = saved.get("history")
        last_step = step - step % validation_interval
        expected_steps = list(range(last_step - (expected_count - 1) * validation_interval,
                                    last_step + 1, validation_interval)) if expected_count else []
        if (not isinstance(history, list) or len(history) != expected_count
                or any(not isinstance(reading, dict) for reading in history)
                or [reading.get("step") for reading in history] != expected_steps):
            raise ValueError("Validation history is empty, incomplete, or duplicates a reading")
        for reading in history:
            loss, precision = reading.get("loss"), reading.get("precision")
            if type(loss) not in (int, float) or precision not in ("full", "rounded_4dp"):
                raise ValueError("Invalid validation loss provenance")
            if precision == "rounded_4dp" and (not math.isfinite(loss) or round(loss, 4) != loss):
                raise ValueError("Historical loss is not a four-decimal logged value")
        if history:
            last, actual = history[-1]["loss"], checkpoint.get("val_loss")
            if (history[-1]["precision"] != "full" or type(actual) not in (int, float)
                    or not (last == actual or (math.isnan(last) and math.isnan(actual)))):
                raise ValueError("Latest validation reading must match the full checkpoint loss")
        result.history = copy.deepcopy(history)
        stop = saved.get("stop")
        if stop is not None:
            if (not result._rising()
                    or stop != {"reason": STOP_REASON, "step": last_step}):
                raise ValueError("Invalid recorded validation divergence stop")
            result.stop = copy.deepcopy(stop)
        elif result._rising():
            # Dropping the stop field must not erase an already-triggered guard.
            result.stop = {"reason": STOP_REASON, "step": last_step}
        result.require_resume_allowed()
        return result
