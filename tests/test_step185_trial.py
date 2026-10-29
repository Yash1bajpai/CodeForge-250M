"""Phone-safe trial contracts. Temporary local ledgers are tests, never GPU fallbacks."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from training.step185_trial import (BINDING, ORIGINAL_HISTORY, ORIGINAL_STOP, ClaimError,
                                   TrialLimit, approved_waiver, canonical, digest, resume_preflight)
from training.validation_history import ValidationGuardStopped, ValidationHistory


class LocalTestLedger:
    def __init__(self, directory, fail=None):
        self.directory = Path(directory)
        self.fail = fail

    def create_once(self, key, claim):
        if self.fail == "before":
            raise OSError("Unavailable test ledger")
        path = self.directory / (key + ".json")
        with path.open("xb") as stream:
            stream.write(canonical(claim))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if self.fail == "after":
            raise TimeoutError("Committed but reply lost")
        actual = json.loads(path.read_text())
        return {"claim": actual, "sha256": "wrong" if self.fail == "readback" else digest(actual),
                "revision": digest(actual)}


class Step185TrialTest(unittest.TestCase):
    # The prepared cloud CPU script replaces this with actual backed-up checkpoint metadata.
    real_checkpoint_metadata = None
    source_root = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.sidecar = self.source_root / "configs/step185_validation_history.json"
        self.metadata = copy.deepcopy(self.real_checkpoint_metadata or {
            "step": 185, "val_loss": ORIGINAL_HISTORY[-1]["loss"],
            "resume_state": {"sample_cursor": 47232, "fingerprint": BINDING["data_fingerprint"],
                             "extension": {"validation_identity": BINDING["validation_identity"]}}})
        self.now = [100.0]
        self.store = LocalTestLedger(self.temporary.name)

    def prepare(self, metadata=None, waiver="approved", **overrides):
        arguments = {"validation_interval": 1, "tokenizer_sha256": BINDING["tokenizer_sha256"],
                     "checkpoint_sha256": BINDING["checkpoint_sha256"], "sidecar_path": self.sidecar,
                     "waiver": approved_waiver() if waiver == "approved" else waiver, "start_epoch": 100.0,
                     "clock": lambda: self.now[0], "monotonic": lambda: self.now[0]}
        arguments.update(overrides)
        return resume_preflight(self.metadata if metadata is None else metadata, **arguments)

    def update(self, guard, trial, step, loss):
        trial.begin_attempt()
        trial.before_optimizer_step()
        trial.complete_attempt(True)
        return guard.record(step, loss)

    def test_ordinary_resume_refuses_and_exact_waiver_needs_claim(self):
        with self.assertRaises(ValidationGuardStopped):
            self.prepare(waiver=None)
        guard, trial = self.prepare()
        self.assertEqual(guard.history, ORIGINAL_HISTORY)
        with self.assertRaises(ClaimError):
            trial.begin_attempt()
        with self.assertRaises(ClaimError):
            guard.record(186, 6.09)
        trial.claim(self.store)
        self.assertTrue(trial.snapshot()["claim_receipt"]["claim"]["claimed_before_optimizer_update"])
        self.assertEqual(guard.stop, ORIGINAL_STOP)

    def test_all_waiver_and_checkpoint_bindings_fail_closed(self):
        waiver = approved_waiver()
        for key in waiver:
            with self.subTest(waiver_field=key):
                changed = copy.deepcopy(waiver)
                changed[key] = "wrong"
                with self.assertRaises(ValueError):
                    self.prepare(waiver=changed)
        for key, value in BINDING.items():
            with self.subTest(binding_field=key):
                changed = copy.deepcopy(waiver)
                changed["binding"][key] = value + 1 if type(value) is int else "wrong"
                with self.assertRaises(ValueError):
                    self.prepare(waiver=changed)
        changed = copy.deepcopy(self.metadata)
        changed["resume_state"]["sample_cursor"] += 256
        with self.assertRaises(ValueError):
            self.prepare(metadata=changed)
        with self.assertRaises(ValueError):
            self.prepare(checkpoint_sha256="0" * 64)
        with self.assertRaises(ValueError):
            self.prepare(tokenizer_sha256="0" * 64)

    def test_consumed_claim_cannot_be_reused_in_new_instance(self):
        _, trial = self.prepare()
        trial.claim(self.store)
        with self.assertRaises(ClaimError):
            trial.claim(self.store)
        _, another = self.prepare()
        with self.assertRaises(ClaimError):
            another.claim(self.store)
        with self.assertRaises(ClaimError):
            another.begin_attempt()

    def test_failed_ambiguous_and_corrupt_readback_claims_abort(self):
        for failure in ("before", "after", "readback"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                _, trial = self.prepare()
                store = LocalTestLedger(root, fail=failure)
                with self.assertRaises(ClaimError):
                    trial.claim(store)
                with self.assertRaises(ClaimError):
                    trial.begin_attempt()
                with self.assertRaises(ClaimError):
                    trial.claim(store)
                if failure != "before":
                    _, another = self.prepare()
                    with self.assertRaises(ClaimError):
                        another.claim(LocalTestLedger(root))

    def test_first_new_rising_window_stops_after_one_update(self):
        guard, trial = self.prepare()
        trial.claim(self.store)
        self.assertTrue(self.update(guard, trial, 186, 6.09))
        self.assertEqual([point["step"] for point in guard.history], [183, 184, 185, 186])
        self.assertEqual(guard.stop, {"reason": "VAL_LOSS_DIVERGENCE", "step": 186})
        self.assertEqual(guard.stop_events[0]["history"], ORIGINAL_HISTORY)
        self.assertEqual(guard.stop_events[0]["stop"], ORIGINAL_STOP)
        self.assertEqual(trial.applied, 1)
        with self.assertRaises(ClaimError):
            trial.begin_attempt()
        with self.assertRaises(ValidationGuardStopped):
            guard.record(187, 1.0)

    def test_equal_or_lower_keeps_old_readings_and_persists_hold(self):
        for first in (ORIGINAL_HISTORY[-1]["loss"], 6.0):
            with self.subTest(first_loss=first), tempfile.TemporaryDirectory() as root:
                guard, trial = self.prepare()
                trial.claim(LocalTestLedger(root))
                for step in (186, 187, 188):
                    self.assertFalse(self.update(guard, trial, step, first + (step - 186) * 0.01))
                    self.assertEqual(guard.stop, ORIGINAL_STOP)
                    self.assertEqual(guard.stop_events[0]["history"], ORIGINAL_HISTORY)
                self.assertEqual([point["step"] for point in guard.history], [185, 186, 187, 188])
                state = copy.deepcopy(self.metadata)
                state.update(step=188, val_loss=first + 0.02)
                state["resume_state"].update(validation_guard=guard.snapshot(), step185_trial=trial.snapshot())
                path = Path(root) / "metadata.json"
                path.write_text(json.dumps(state))
                restored = json.loads(path.read_text())
                with self.assertRaises(ClaimError):
                    self.prepare(metadata=restored)
                with self.assertRaises(ValidationGuardStopped):
                    ValidationHistory.restore(restored, 1)
                self.assertTrue(self.update(guard, trial, 189, first + 0.03))

    def test_attempt_applied_and_wall_limits(self):
        _, trial = self.prepare()
        trial.claim(self.store)
        for _ in range(8):
            trial.begin_attempt()
            trial.before_optimizer_step()
            trial.complete_attempt(True)
        self.assertEqual((trial.attempted, trial.applied), (8, 8))
        with self.assertRaises(TrialLimit):
            trial.begin_attempt()
        with tempfile.TemporaryDirectory() as root:
            _, overflow = self.prepare()
            overflow.claim(LocalTestLedger(root))
            for _ in range(8):
                overflow.begin_attempt()
                overflow.before_optimizer_step()
                overflow.complete_attempt(False)
            self.assertEqual(overflow.applied, 0)
            with self.assertRaises(TrialLimit):
                overflow.begin_attempt()
        with tempfile.TemporaryDirectory() as root:
            _, timed = self.prepare()
            timed.claim(LocalTestLedger(root))
            timed.begin_attempt()
            self.now[0] = 2200.0
            with self.assertRaises(TrialLimit):
                timed.before_optimizer_step()
            self.assertEqual(timed.applied, 0)


if __name__ == "__main__":
    unittest.main()
