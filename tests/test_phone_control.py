"""Isolated security boundaries for the phone's live-check command."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import phone_control


class PhoneCheckSecurityTests(unittest.TestCase):
    def test_codex_gate_prevents_request(self):
        gates = (
            {"revocation_required": True,
             "provider_revocation_verified": True,
             "fresh_login_after_revocation_verified": True},
            {"revocation_required": False,
             "provider_revocation_verified": False,
             "fresh_login_after_revocation_verified": True},
            {"revocation_required": False,
             "provider_revocation_verified": True,
             "fresh_login_after_revocation_verified": False},
            {},
        )
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            receipt = home / "codeforge-storage/logs/provider-verification.json"
            receipt.parent.mkdir(parents=True)
            for gate in gates:
                with self.subTest(gate=gate):
                    receipt.write_text(json.dumps({"openai_codex": gate}), encoding="utf-8")
                    with patch.object(phone_control.Path, "home", return_value=home), \
                            patch.object(phone_control, "_model_probe") as request:
                        result = phone_control._codex_check()
                    request.assert_not_called()
                    self.assertFalse(result["ok"])
                    self.assertFalse(result["live_request_attempted"])
                    self.assertEqual(result["state"], "ACTION")

    def test_missing_receipt_does_not_trust_cached_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(phone_control.Path, "home", return_value=Path(directory)), \
                    patch.object(phone_control, "_model_probe") as request:
                result = phone_control._codex_check()
        request.assert_not_called()
        self.assertEqual(result["security_state"], "unknown")
        self.assertFalse(result["ok"])

    def test_provider_marker_must_be_a_complete_reply_line(self):
        marker = "CF_CHECK_unique"
        with patch.object(phone_control.uuid, "uuid4", return_value=SimpleNamespace(hex="unique")):
            for output in ("prompt: " + marker + "\n", "error mentioning " + marker + "\n", " " + marker + "\n"):
                with self.subTest(output=output), \
                        patch.object(phone_control.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
                    with self.assertRaises(ValueError):
                        phone_control._model_probe("google-antigravity/gemini-3.8-flash")
            with patch.object(phone_control.subprocess, "run",
                              return_value=SimpleNamespace(stdout=marker + "\n")):
                result = phone_control._model_probe("google-antigravity/gemini-3.8-flash")
        self.assertTrue(result["live_request_passed"])

    def test_failed_model_output_never_enters_report(self):
        secret = "credential-bearing-body-and-signed-url"
        error = subprocess.CalledProcessError(
            1, ["omp", "--private", secret], output=secret, stderr=secret)
        checks = {}
        with patch.object(phone_control.subprocess, "run", side_effect=error):
            phone_control.run_check(checks, "provider", lambda: phone_control._model_probe("provider/model"))
        self.assertEqual(checks["provider"]["error"],
                         {"exception": "CalledProcessError", "http_status": None})
        self.assertNotIn(secret, json.dumps(checks))

    def test_receipt_exception_body_is_not_published(self):
        secret = "private-provider-receipt-body"
        with patch.object(phone_control.Path, "read_text", side_effect=ValueError(secret)), \
                patch.object(phone_control, "_model_probe") as request:
            result = phone_control._codex_check()
        request.assert_not_called()
        self.assertNotIn(secret, json.dumps(result))
        self.assertEqual(result["error"], {"exception": "ValueError", "http_status": None})


if __name__ == "__main__":
    unittest.main()
