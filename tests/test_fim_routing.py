import json, tempfile, unittest, random
from pathlib import Path
from unittest import mock
import sys, types
try:
    import torch  # noqa
    import transformers  # noqa
except ImportError:  # CPU-only CI without the ML stack: stub just what the module imports
    t = types.ModuleType("torch"); t.save = lambda *a, **k: None; t.tensor = lambda *a, **k: None; t.uint16 = None
    tr = types.ModuleType("transformers"); tr.PreTrainedTokenizerFast = object
    sys.modules.setdefault("torch", t); sys.modules.setdefault("transformers", tr)
import data.tokenize_dataset as td

CODE = "\n".join(f"x{i} = {i}" for i in range(30))

class FimRoutingTests(unittest.TestCase):
    def test_only_code_sources_allowed(self):
        for s in ["starcoder-python", "codeparrot-clean"]:
            self.assertTrue(td.fim_allowed(s))
        for s in ["fineweb-edu", "evol-codealpaca", "glaive-function-calling", "commitpackft-python", "tiny-textbooks"]:
            self.assertFalse(td.fim_allowed(s))

    def test_source_name_from_dedup_file(self):
        self.assertEqual(td.source_name("data/dedup/starcoder-python_dedup.jsonl"), "starcoder-python")

    def test_no_double_wrap(self):
        doc = "<|fim_prefix|>a<|fim_middle|>b" + "z" * 80
        self.assertEqual(td.apply_fim_transformation(doc, fim_rate=1.0), doc)

    def test_code_gets_fim_at_rate_one(self):
        random.seed(0)
        out = td.apply_fim_transformation(CODE, fim_rate=1.0)
        self.assertTrue(out.startswith("<|fim_prefix|>") and "<|fim_middle|>" in out)

    def test_build_dataset_routes_by_source(self):
        class Tok:
            eos_token_id = 0
            def encode(self, text):
                # one id per FIM token present, else one id for the doc
                return [1] * sum(text.count(t) for t in td.FIM_TOKENS) + [2]
            @classmethod
            def from_pretrained(cls, p): return cls()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d/"dedup").mkdir(); (d/"tok").mkdir()
            for name in ["starcoder-python", "fineweb-edu", "glaive-function-calling"]:
                with open(d/"dedup"/f"{name}_dedup.jsonl", "w") as f:
                    for _ in range(40): f.write(json.dumps({"text": CODE}) + "\n")
            seen = {}
            real = td.apply_fim_transformation
            def spy(code, fim_rate=0.5):
                seen["n"] = seen.get("n", 0) + 1
                return real(code, fim_rate=1.0)
            with mock.patch.object(td, "PreTrainedTokenizerFast", Tok), \
                 mock.patch.object(td, "apply_fim_transformation", spy), \
                 mock.patch.object(td.torch, "save"):
                td.build_tokenized_dataset(str(d/"dedup"), str(d/"tok"), str(d/"out"), seq_len=4)
            # FIM called only for the code source's 40 docs
            self.assertEqual(seen["n"], 40)

if __name__ == "__main__":
    unittest.main()
