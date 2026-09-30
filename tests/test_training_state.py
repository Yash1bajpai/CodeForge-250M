"""CPU checkpoint regression, independent of CUDA training."""
import tempfile
import unittest
from pathlib import Path
import torch
from training.train import save_checkpoint

class CheckpointTest(unittest.TestCase):
    def test_state_roundtrip(self):
        m=torch.nn.Linear(2,2); o=torch.optim.AdamW(m.parameters()); s=torch.amp.GradScaler('cpu')
        m(torch.ones(1,2)).sum().backward(); o.step(); o.zero_grad()
        r={'schema_version':1,'fingerprint':'x','seqs_per_step':256,'sample_cursor':256,'completed_updates':1,'optimizer_step_count':1}
        with tempfile.TemporaryDirectory() as d:
            p=str(Path(d)/'latest.pt')
            save_checkpoint(m,o,1,.5,None,d,p,resume_state=r,scaler=s)
            c=torch.load(p,weights_only=True)
            self.assertEqual(c['step'],1)
            self.assertEqual(c['resume_state'],r)
            self.assertIn('rng_state',c)
            self.assertTrue(c['scaler_state_dict'])
            self.assertTrue(torch.equal(c['model_state_dict']['weight'],m.weight))

if __name__=='__main__':unittest.main()

class ActualDatasetMigrationTest(unittest.TestCase):
    def test_short_shards_no_uninitialized_rows(self):
        from training.train import LazyShardDataset
        from training.resume_safety import migrated_permutation
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i,rows in enumerate([4,2,4]):
                p=Path(d)/f'shard_{i:04d}.pt';paths.append(str(p))
                torch.save(torch.full((rows,8),i,dtype=torch.uint16),p)
            ds=LazyShardDataset(d,seq_length=8,shard_files=paths,vocab_size=16)
            self.assertEqual(len(ds),10)
            self.assertEqual(ds.data[:,0].long().tolist(),[0,0,0,0,1,1,2,2,2,2])
            perm,valid=migrated_permutation(ds.rows,legacy_rows=4)
            self.assertEqual(sorted(perm),list(range(10)))
            self.assertEqual(int(valid[:7].sum()),len([x for x in torch.randperm(12,generator=torch.Generator().manual_seed(42))[:7] if x.item() not in [6,7]]))

class PartialBatchGradientTest(unittest.TestCase):
    def test_last_batch_matches_mean_loss(self):
        a=torch.nn.Linear(2,1,bias=False);b=torch.nn.Linear(2,1,bias=False);b.load_state_dict(a.state_dict())
        x=torch.arange(10,dtype=torch.float32).reshape(5,2)
        a(x).square().mean().backward()
        planned=8;actual=5
        for part in [x[:3],x[3:]]:
            (b(part).square().mean()*len(part)/planned).backward()
        b.weight.grad.mul_(planned/actual)
        self.assertTrue(torch.allclose(a.weight.grad,b.weight.grad,atol=1e-5))

class WallClockGuardTest(unittest.TestCase):
    def test_embedded_guard_boundaries(self):
        import ast
        from types import SimpleNamespace
        tree = ast.parse((Path(__file__).parents[1] / 'training/train.py').read_text())
        expression = next(n.value for n in ast.walk(tree)
                          if isinstance(n, ast.Assign) and any(
                              isinstance(t, ast.Name) and t.id == 'wall_hit'
                              for t in n.targets))
        code = compile(ast.Expression(expression), '<wall-guard>', 'eval')
        for now, start, expected in [(999, 0, False), (1000, 0, True),
                                     (900, -22000, True)]:
            env = {'time': SimpleNamespace(time=lambda: now),
                   'start_time': start, 'args': SimpleNamespace(max_hours=6),
                   'os': SimpleNamespace(environ={'CF_ABSOLUTE_STOP_EPOCH': '1000'})}
            self.assertEqual(eval(code, env), expected)
