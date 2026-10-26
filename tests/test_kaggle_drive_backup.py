"""Backup safety boundaries; HTTP fixtures use only a local server."""
import importlib.util
from contextlib import nullcontext
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.kaggle_drive_backup import backup, download_verified, output_path


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/error"):
            self.send_error(403)
            return
        if self.path in ("/train", "/validation"):
            body = self.path.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Length", "2" if self.path == "/size" else "8")
        self.end_headers()
        self.wfile.write(b"ab")
        self.close_connection = True

    def log_message(self, *args):
        pass


class BackupTests(unittest.TestCase):
    def test_other_accounts_rejected_before_authentication(self):
        for kernel in ("utkarshhsinghhh/job", "tanishkkvermaa/job", "job", "yashbajpai2027/job/1"):
            with self.subTest(kernel=kernel), self.assertRaises(ValueError):
                backup(kernel, 1, Path("unused-backup-destination"))

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                backup("yashbajpai2027/job", 1, Path(directory))

    def test_paths_cannot_escape_or_replace_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../checkpoint.pt", "/checkpoint.pt", "C:/checkpoint.pt", "a\\b", "backup-manifest.json", ""):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    output_path(Path(directory), name)

    def test_incomplete_or_wrong_size_download_is_not_published(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                for endpoint, expected in (("/partial", 8), ("/size", 3)):
                    with self.subTest(endpoint=endpoint):
                        target = Path(directory) / endpoint.removeprefix("/")
                        with self.assertRaises((ValueError, RuntimeError)):
                            download_verified(f"http://127.0.0.1:{server.server_port}{endpoint}", target, expected)
                        self.assertFalse(target.exists())
                target = Path(directory) / "denied"
                with self.assertRaises(RuntimeError) as error:
                    download_verified(f"http://127.0.0.1:{server.server_port}/error?token=private-probe", target, 2)
                self.assertNotIn("private-probe", str(error.exception))
                self.assertFalse(target.exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    @unittest.skipUnless(importlib.util.find_spec("kagglesdk"), "Kaggle SDK is an optional PC dependency")
    def test_same_basename_in_distinct_output_directories_is_preserved(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            items = [SimpleNamespace(file_name=f"{split}/manifest.json",
                                     url=f"http://127.0.0.1:{server.server_port}/{split}")
                     for split in ("train", "validation")]
            service = SimpleNamespace(
                get_kernel=lambda request: SimpleNamespace(metadata=SimpleNamespace(current_version_number=1)),
                list_kernel_files=lambda request: SimpleNamespace(
                    files=[SimpleNamespace(name="manifest.json"), SimpleNamespace(name="manifest.json")],
                    next_page_token=""),
                list_kernel_session_output=lambda request: SimpleNamespace(files=items, next_page_token=""),
            )
            client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=service))
            for source_status in ("COMPLETE", "ERROR"):
                api = SimpleNamespace(
                    kernels_status=lambda kernel: SimpleNamespace(status=source_status),
                    build_kaggle_client=lambda: nullcontext(client),
                )
                with self.subTest(source_status=source_status), tempfile.TemporaryDirectory() as directory:
                    destination = Path(directory) / "backup"
                    with patch("scripts.kaggle_drive_backup.owner_api", return_value=api):
                        report = backup("yashbajpai2027/job", 1, destination)
                    self.assertEqual((destination / "train/manifest.json").read_bytes(), b"/train")
                    self.assertEqual((destination / "validation/manifest.json").read_bytes(), b"/validation")
                    self.assertEqual({item["path"] for item in report["files"]},
                                     {"train/manifest.json", "validation/manifest.json"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
