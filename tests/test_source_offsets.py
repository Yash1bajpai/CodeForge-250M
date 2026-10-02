import unittest,tempfile,json,sys,types
from pathlib import Path
from unittest.mock import patch
try: import datasets
except ImportError:
    sys.modules['datasets']=types.SimpleNamespace(load_dataset=lambda *a,**kw:None)
from data.download_stack import download_curated_stack
class Offsets(unittest.TestCase):
    def test_explicit_zero_keeps_finite_instruction_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cfg=p/'config.yaml'
            cfg.write_text('data:\n  target_tokens: 100\n  languages: [evol-codealpaca]\n  language_weights: [1]\n  skip_chars_by_source: {evol-codealpaca: 0}\n')
            with patch('data.download_stack.load_dataset',return_value=[{'instruction':'Describe a function','output':'Here is an answer'}]),patch.dict('os.environ',{'CF_SKIP_STAGE_A_DOCS':'1'}):
                download_curated_stack(str(cfg),str(p/'out'))
            stats=json.load(open(p/'out/source_stats.json'))
            self.assertEqual(stats['evol-codealpaca']['documents'],1)
    def test_empty_after_offset_is_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cfg=p/'c.yaml'
            cfg.write_text('data:\n  target_tokens: 100\n  languages: [evol-codealpaca]\n  language_weights: [1]\n  skip_chars_by_source: {evol-codealpaca: 1000000}\n')
            with patch('data.download_stack.load_dataset',return_value=[{'instruction':'a','output':'b'}]):
                with self.assertRaises(RuntimeError):download_curated_stack(str(cfg),str(p/'out'))
