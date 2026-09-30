import tempfile
import unittest
from pathlib import Path
import torch
from training.resume_safety import audit_shards, manifest_fingerprint, optimizer_step_counts, validate_resume_metadata

class ResumeSafetyTests(unittest.TestCase):
    def test_valid_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'shard_0001.pt'; tok=Path(d)/'tokenizer.json'
            torch.save(torch.zeros((2,8),dtype=torch.uint16),p); tok.write_text('{}')
            a=audit_shards([p],8,16)
            self.assertEqual(a[0]['rows'],2)
            f=manifest_fingerprint(a,tok); tok.write_text('{"changed":true}')
            self.assertNotEqual(f,manifest_fingerprint(a,tok))
    def test_short_shard_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ps=[Path(d)/f'shard_{i}.pt' for i in range(2)]
            for p,n in zip(ps,[2,1]): torch.save(torch.zeros((n,8),dtype=torch.uint16),p)
            with self.assertRaisesRegex(ValueError,'Unequal'): audit_shards(ps,8,16)
    def test_bad_shape_dtype_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.pt'
            for t in [torch.zeros((2,7),dtype=torch.uint16),torch.zeros((2,8)),torch.full((2,8),16,dtype=torch.uint16)]:
                torch.save(t,p)
                with self.assertRaises(ValueError): audit_shards([p],8,16)
    def test_optimizer_and_resume_checks(self):
        o={'state':{0:{'step':torch.tensor(7.)},1:{'step':torch.tensor(7.)}}}
        self.assertEqual(optimizer_step_counts(o),7)
        c={'optimizer_state_dict':o}
        with self.assertRaisesRegex(ValueError,'Legacy'): validate_resume_metadata(c,'f',256)
        c['resume_state']={'schema_version':1,'fingerprint':'f','seqs_per_step':256,'sample_cursor':1792,'completed_updates':7,'optimizer_step_count':7}
        self.assertEqual(validate_resume_metadata(c,'f',256)['sample_cursor'],1792)
        with self.assertRaises(ValueError): validate_resume_metadata(c,'wrong',256)
        o['state'][1]['step']=torch.tensor(8.)
        with self.assertRaises(ValueError): optimizer_step_counts(o)

if __name__=='__main__': unittest.main()

class LegacyCursorTest(unittest.TestCase):
    def test_wall_clock_accounting(self):
        from training.resume_safety import reconcile_legacy_cursor
        self.assertEqual(reconcile_legacy_cursor(2439,2080,187603456,256,2048),(2438,624128))
        for n in [187603457,0]:
            with self.assertRaises(ValueError):reconcile_legacy_cursor(2439,2080,n,256,2048)

class MigrationOrderTest(unittest.TestCase):
    def test_relative_order_and_cursor(self):
        from training.resume_safety import migrated_permutation
        rows=[4,2,4]
        perm, valid=migrated_permutation(rows,legacy_rows=4)
        self.assertEqual(sorted(perm),list(range(10)))
        old=torch.randperm(12,generator=torch.Generator().manual_seed(42)).tolist()
        expected=[]
        for idx in old:
            if idx in [6,7]:continue
            expected.append(idx if idx<6 else idx-2)
        self.assertEqual(perm,expected)
        self.assertEqual(int(valid.sum()),10)
