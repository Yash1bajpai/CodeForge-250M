"""Phone-safe regressions for CPU packaging/diagnostics and single-used-GPU admission."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from training.preflight_support import (capture_entry, single_used_t4, validate_gpu_admission,
                                        write_cpu_config)
from training.step185_trial import RUN_ID


class PreflightSupportTest(unittest.TestCase):
    def test_scientific_floats_survive_trainer_yaml_reader(self):
        context = {"model": {"rms_norm_eps": 1e-5},
                   "training": {"min_learning_rate": 6e-5, "adam_epsilon": 1e-8},
                   "data": {"seq_length": 2048}, "run": {"seed": 42}}
        with tempfile.TemporaryDirectory() as directory:
            path = write_cpu_config(context, Path(directory) / "config.yaml")
            loaded = yaml.safe_load(path.read_text())
        self.assertEqual({key: loaded[key] for key in context}, context)
        for section, key in (("model", "rms_norm_eps"), ("training", "min_learning_rate"),
                             ("training", "adam_epsilon")):
            self.assertIs(type(loaded[section][key]), float)

    def test_entry_system_exit_codes_and_argv_restoration(self):
        original = sys.argv
        for code, expected in ((None, 0), (0, 0), (2, 2), ("configuration rejected", 1)):
            with self.subTest(code=code):
                def entry():
                    raise SystemExit(code)
                record = capture_entry(entry, ["train.py", "--cpu-resume-preflight"], "waiver")
                self.assertEqual(record["exit_code"], expected)
                self.assertEqual(record["system_exit_code"], code)
                self.assertEqual(record["system_exit_message"], str(code))
                self.assertEqual(record["exception_type"], "SystemExit")
                self.assertIs(sys.argv, original)
        timeout = capture_entry(lambda: (_ for _ in ()).throw(TimeoutError("CPU deadline")), ["train.py"], "waiver")
        self.assertEqual((timeout["exit_code"], timeout["exception_type"]), (1, "TimeoutError"))

    def test_argparse_failure_records_exit_two_and_stderr(self):
        def entry():
            parser = argparse.ArgumentParser(prog="train.py")
            parser.add_argument("--checkpoint", required=True)
            parser.parse_args()
        record = capture_entry(entry, ["train.py"], "ordinary")
        self.assertEqual(record["exit_code"], 2)
        self.assertEqual(record["system_exit_code"], 2)
        self.assertIn("--checkpoint", record["stderr_tail"])
        self.assertEqual(record["stdout_tail"], "")

    def test_secrets_redacted_before_tail_and_status_survives_long_output(self):
        credential = "hf_" + "syntheticnotarealcredential" * 3
        signed_path = "private-secret-segment" * 600
        status = "METADATA_PREFLIGHT_ONLY_NOT_CLAIMED"
        def entry():
            print(json.dumps({"status": status}))
            print("benign diagnostic\n" * 400)
            print("https://example.invalid/" + signed_path + "?signature=synthetic")
            print("Authorization: Bearer " + credential, file=sys.stderr)
            print("final diagnostic")
            raise SystemExit("config rejected token=" + credential)
        record = capture_entry(entry, ["train.py"], "waiver")
        self.assertEqual(record["exit_code"], 1)
        self.assertEqual(record["reported_statuses"], [status])
        self.assertEqual(len(record["stdout_tail"]), 4000)
        self.assertLessEqual(len(record["stderr_tail"]), 4000)
        self.assertTrue(record["stdout_tail"].endswith("final diagnostic\n"))
        serialized = json.dumps(record)
        for secret in (credential, "private-secret-segment", "signature=synthetic"):
            self.assertNotIn(secret, serialized)
        self.assertIn("[REDACTED]", record["system_exit_message"])

    def test_allocated_two_needs_explicit_one_visible_gpu_zero(self):
        admission = {"owner": "yashbajpai2027", "free_compute": True, "run_id": RUN_ID,
                     "gpu_name": "Tesla T4", "allocated_gpu_count": 2, "visible_gpu_count": 1,
                     "cuda_visible_devices": "0", "remaining_quota_seconds": 3600,
                     "checked_at_epoch": 1000, "backend_timeout_seconds": 3600}
        self.assertEqual(validate_gpu_admission(admission, 1100, "0"), 2)
        self.assertEqual(validate_gpu_admission(dict(admission, allocated_gpu_count=1), 1100, "0"), 1)
        for patch in ({"visible_gpu_count": 2}, {"visible_gpu_count": None},
                      {"cuda_visible_devices": "0,1"}, {"cuda_visible_devices": None},
                      {"allocated_gpu_count": 3}, {"free_compute": False},
                      {"remaining_quota_seconds": 3599}, {"checked_at_epoch": 499},
                      {"backend_timeout_seconds": 7200}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_gpu_admission(dict(admission, **patch), 1100, "0")
        with self.assertRaises(ValueError):
            validate_gpu_admission(admission, 1100, "0,1")

    def test_runtime_rejects_second_visible_gpu_or_other_device(self):
        hardware = single_used_t4(1, "Tesla T4", 1, "0")
        self.assertEqual((hardware["visible_gpu_count"], hardware["used_gpu_count"],
                          hardware["cuda_visible_devices"]), (1, 1, "0"))
        for args in ((2, "Tesla T4", 1, "0"), (0, "Tesla T4", 1, "0"),
                     (1, "Tesla T4", 2, "0"), (1, "Tesla P100", 1, "0"),
                     (1, "Tesla T4", 1, "1"), (1, "Tesla T4", 1, None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                single_used_t4(*args)


if __name__ == "__main__":
    unittest.main()
