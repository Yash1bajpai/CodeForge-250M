"""Synchronous fail-closed uploads of completed atomic local checkpoints."""
import os
from pathlib import PurePosixPath

class SafeCheckpointUploader:
    def __init__(self, api, repo_id, prefix):
        p = PurePosixPath(prefix)
        if len(p.parts) != 1 or not prefix.startswith('stageA3') or prefix in ('.', '..'):
            raise ValueError('Upload prefix must be a new stageA3-prefixed folder')
        self.api, self.repo_id, self.prefix = api, repo_id, prefix
        files = api.list_repo_files(repo_id=repo_id)
        if any(f.startswith(prefix + '/') and ('checkpoint' in f) for f in files):
            raise ValueError('Existing destination checkpoint: choose a fresh prefix')
    def upload(self, checkpoint, step):
        if not os.path.isfile(checkpoint) or os.path.getsize(checkpoint) == 0:
            raise ValueError('Missing completed checkpoint')
        result = self.api.upload_file(path_or_fileobj=checkpoint,
            path_in_repo=self.prefix + '/latest_checkpoint.pt', repo_id=self.repo_id,
            commit_message=f'Checkpoint step {step} in isolated continuation folder')
        revision = getattr(result, 'oid', None)
        if not revision:
            raise RuntimeError('Upload returned no commit revision')
        # Readback at the returned immutable revision. Never label upload complete on exception.
        info = self.api.get_paths_info(repo_id=self.repo_id,
            paths=[self.prefix + '/latest_checkpoint.pt'], revision=revision)
        if not info or info[0].size != os.path.getsize(checkpoint):
            raise RuntimeError('Remote checkpoint size readback failed')
        return revision
