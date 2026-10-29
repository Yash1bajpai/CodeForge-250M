"""Guard-state round trips only: no model, optimizer, torch or cloud access."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.validation_history import RULE_VERSION, ValidationGuardStopped, ValidationHistory


class ValidationHistoryTest(unittest.TestCase):
    def test_save_resume_preserves_next_decision_and_divergence_stop(self):
        # A synthetic run ending at185 with three rising readings, not a stop.
        def uninterrupted():
            history = ValidationHistory(1)
            for step in range(1, 183):
                history.record(step, 6.1)
            for step, loss in ((183, 6.0), (184, 6.1), (185, 6.2)):
                history.record(step, loss)
            return history

        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "resume-metadata.json"
            for next_loss, expected in ((6.3, True), (6.2, False), (6.0, False)):
                with self.subTest(next_loss=next_loss):
                    baseline = uninterrupted()
                    checkpoint = {"step": 185, "val_loss": 6.2,
                                  "resume_state": {"sample_cursor": 47232,
                                                   "validation_guard": baseline.snapshot()}}
                    metadata_path.write_text(json.dumps(checkpoint))
                    restored = ValidationHistory.restore(json.loads(metadata_path.read_text()), 1)
                    with self.assertRaises(ValueError):
                        restored.record(185, 6.2)
                    self.assertEqual(restored.record(186, next_loss), expected)
                    self.assertEqual(baseline.record(186, next_loss), expected)
                    self.assertEqual(restored.snapshot(), baseline.snapshot())
                    self.assertEqual(sum(point["step"] == 185 for point in restored.history), 1)
                    reset = copy.deepcopy(checkpoint)
                    reset["resume_state"]["validation_guard"]["history"] = []
                    with self.assertRaises(ValueError):
                        ValidationHistory.restore(reset, 1)
                    del reset["resume_state"]["validation_guard"]
                    with self.assertRaises(ValueError):
                        ValidationHistory.restore(reset, 1)

            # Reconstruct a stopped state, retaining the historical rounding labels.
            stopped = ValidationHistory(1)
            for step in range(1, 182):
                stopped.record(step, 6.0945)
            for step, loss in ((182, 6.0613), (183, 6.0621), (184, 6.0847),
                               (185, 6.085826873779297)):
                stopped.record(step, loss)
            guard_state = stopped.snapshot()
            for point in guard_state["history"][:3]:
                point["precision"] = "rounded_4dp"
            checkpoint = {"step": 185, "val_loss": 6.085826873779297,
                          "resume_state": {"sample_cursor": 47232, "fingerprint": "3" * 64,
                                           "extension": {"validation_identity": "4" * 64}}}
            embedded = copy.deepcopy(checkpoint)
            embedded["resume_state"]["validation_guard"] = guard_state
            with self.assertRaisesRegex(ValidationGuardStopped, "185.*remains active"):
                ValidationHistory.restore(embedded, 1)
            binding = {"checkpoint_sha256": "1" * 64, "checkpoint_step": 185,
                       "sample_cursor": 47232, "data_fingerprint": "3" * 64,
                       "validation_identity": "4" * 64, "tokenizer_sha256": "2" * 64,
                       "guard_rule_version": RULE_VERSION}
            sidecar = {"format_version": 1, "binding": binding, "validation_guard": guard_state}
            sidecar_path = Path(directory) / "guard-history.json"
            for remove_stop in (False, True):
                with self.subTest(remove_stop=remove_stop):
                    candidate = copy.deepcopy(sidecar)
                    if remove_stop:
                        candidate["validation_guard"]["stop"] = None
                    sidecar_path.write_text(json.dumps(candidate))
                    with self.assertRaisesRegex(ValidationGuardStopped, "185.*remains active"):
                        ValidationHistory.restore(checkpoint, 1, sidecar_path=sidecar_path,
                                                  checkpoint_sha256="1" * 64,
                                                  tokenizer_sha256="2" * 64)
            for field in binding:
                with self.subTest(binding_field=field):
                    candidate = copy.deepcopy(sidecar)
                    value = candidate["binding"][field]
                    candidate["binding"][field] = value + 1 if type(value) is int else "wrong"
                    sidecar_path.write_text(json.dumps(candidate))
                    with self.assertRaisesRegex(ValueError, "binding mismatch"):
                        ValidationHistory.restore(checkpoint, 1, sidecar_path=sidecar_path,
                                                  checkpoint_sha256="1" * 64,
                                                  tokenizer_sha256="2" * 64)
            with self.assertRaises(ValidationGuardStopped):
                stopped.record(186, 5.0)


if __name__ == "__main__":
    unittest.main()
