import unittest, tempfile
from pathlib import Path
from types import SimpleNamespace
from training.checkpoint_upload import SafeCheckpointUploader
class API:
    def list_repo_files(self, **kw): return ['latest_checkpoint.pt', 'stageA2/latest_checkpoint.pt']
    def upload_file(self, **kw): self.kw=kw; return SimpleNamespace(oid='revision')
    def get_paths_info(self, **kw): return [SimpleNamespace(size=3)]
class Tests(unittest.TestCase):
    def test_isolated_verified_upload(self):
        a=API();u=SafeCheckpointUploader(a,'repo','stageA3')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'latest_checkpoint.pt';p.write_bytes(b'abc')
            self.assertEqual(u.upload(str(p),1),'revision')
            self.assertEqual(a.kw['path_in_repo'],'stageA3/latest_checkpoint.pt')
    def test_existing_folder_refused(self):
        a=API();a.list_repo_files=lambda **k:['stageA3/latest_checkpoint.pt']
        with self.assertRaises(ValueError): SafeCheckpointUploader(a,'repo','stageA3')
    def test_old_prefix_refused(self):
        for p in ('stageA2','../stageA3','stageA3/../stageA2',''):
            with self.assertRaises(ValueError): SafeCheckpointUploader(API(),'repo',p)
    def test_missing_file_refused(self):
        with self.assertRaises(ValueError): SafeCheckpointUploader(API(),'repo','stageA3').upload('/no/file',1)
