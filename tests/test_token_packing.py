import hashlib
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:
    torch = None
try:
    import transformers  # noqa: F401
except ImportError:
    module = types.ModuleType("transformers")
    module.PreTrainedTokenizerFast = object
    sys.modules.setdefault("transformers", module)

if torch is not None:
    import data.tokenize_dataset as td


class CharacterTokenizer:
    eos_token_id = 0

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def get_vocab(self):
        return {"<eos>": 0, **{chr(i): i + 1 for i in range(256)}}

    def encode(self, text):
        return [ord(char) + 1 for char in text]


@unittest.skipUnless(torch is not None and hasattr(torch, "load"), "requires CPU PyTorch")
class TokenPackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dedup = self.root / "dedup"
        self.tokenizer = self.root / "tokenizer"
        self.dedup.mkdir()
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{\"fixture\": true}", encoding="utf-8")
        self.loader = mock.patch.object(td, "PreTrainedTokenizerFast", CharacterTokenizer)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def documents(self, source, texts):
        with (self.dedup / f"{source}_dedup.jsonl").open("w", encoding="utf-8") as handle:
            for text in texts:
                handle.write(json.dumps({"text": text}) + "\n")

    def build(self, name="out", seq_len=8):
        output = self.root / name
        manifest = td.build_tokenized_dataset(
            str(self.dedup), str(self.tokenizer), str(output), seq_len
        )
        self.assertEqual(manifest, json.loads((output / "manifest.json").read_text()))
        self.assertEqual(manifest["total_tokens"], manifest["valid_tokens"] + manifest["discarded_tokens"])
        self.assertEqual(manifest["sequences"] * seq_len, manifest["valid_tokens"] + manifest["padding_tokens"])
        self.assertEqual(manifest["total_tokens"], sum(source["tokens"] for source in manifest["sources"].values()))
        return output, manifest

    def packed(self, output, manifest):
        rows, lengths = [], []
        for shard in manifest["shards"]:
            path = output / shard["file"]
            valid_path = output / shard["valid_file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), shard["sha256"])
            self.assertEqual(hashlib.sha256(valid_path.read_bytes()).hexdigest(), shard["valid_sha256"])
            tensor = torch.load(path, weights_only=True)
            valid = torch.load(valid_path, weights_only=True)["valid_len"]
            self.assertEqual(tensor.dtype, torch.uint16)
            self.assertEqual(list(tensor.shape), [shard["rows"], manifest["seq_len"]])
            self.assertEqual(valid.numel(), shard["rows"])
            self.assertTrue(all(2 <= length <= manifest["seq_len"] for length in valid.tolist()))
            self.assertEqual(sum(valid.tolist()), shard["valid_tokens"])
            rows.extend(tensor.tolist())
            lengths.extend(valid.tolist())
        return rows, lengths

    def test_small_corpus_emits_real_tail_and_masks_only_padding(self):
        self.documents("fineweb-edu", ["ab"])
        output, manifest = self.build()
        rows, lengths = self.packed(output, manifest)
        self.assertEqual(lengths, [3])
        self.assertEqual(rows, [[98, 99, 0, 0, 0, 0, 0, 0]])
        # Exercise the sidecar's next-token mask contract, including real EOS ID 0.
        targets = torch.tensor(rows[0][1:])
        targets[lengths[0] - 1:] = -100
        self.assertEqual(targets.tolist(), [99, 0, -100, -100, -100, -100, -100])
        self.assertEqual(manifest["padding_tokens"], 5)
        self.assertEqual(manifest["discarded_tokens"], 0)
        self.assertEqual(manifest["tokenizer"]["eos_token_id"], 0)
        self.assertEqual(manifest["tokenizer"]["vocab_size"], 257)
        self.assertEqual(len(manifest["tokenizer"]["sha256"]), 64)

    def test_every_document_token_is_packed_in_sorted_file_order(self):
        self.documents("z-source", ["de"])
        self.documents("a-source", ["a", "bc"])
        output, manifest = self.build(seq_len=6)
        rows, lengths = self.packed(output, manifest)
        expected = [98, 0, 99, 100, 0, 101, 102, 0]
        self.assertEqual([token for row, valid in zip(rows, lengths) for token in row[:valid]], expected)
        self.assertEqual(lengths, [6, 2])
        self.assertEqual(manifest["sources"]["a-source"]["tokens"], 5)
        self.assertEqual(manifest["sources"]["z-source"]["tokens"], 3)

    def test_one_token_tail_is_explicitly_discarded(self):
        self.documents("fineweb-edu", ["abc", ""])
        output, manifest = self.build(seq_len=4)
        rows, lengths = self.packed(output, manifest)
        self.assertEqual(rows, [[98, 99, 100, 0]])
        self.assertEqual(lengths, [4])
        self.assertEqual(manifest["total_tokens"], 5)
        self.assertEqual(manifest["discarded_tokens"], 1)
        self.assertEqual(manifest["padding_tokens"], 0)

    def test_single_token_corpus_has_no_uninitialized_rows(self):
        self.documents("fineweb-edu", [""])
        output, manifest = self.build()
        self.assertEqual(self.packed(output, manifest), ([], []))
        self.assertEqual(manifest["valid_tokens"], 0)
        self.assertEqual(manifest["discarded_tokens"], 1)
        self.assertEqual(manifest["sequences"], 0)

    def test_full_and_partial_shards_have_matching_sidecars(self):
        self.documents("fineweb-edu", ["a" * 2004])
        output, manifest = self.build(seq_len=2)
        rows, lengths = self.packed(output, manifest)
        self.assertEqual([shard["rows"] for shard in manifest["shards"]], [1000, 2])
        self.assertEqual(len(rows), 1002)
        self.assertEqual(lengths, [2] * 1002)
        self.assertEqual([token for row in rows for token in row], [98] * 2004)
        self.assertEqual(manifest["discarded_tokens"], 1)

    def test_build_is_deterministic_without_changing_global_rng(self):
        code = "".join(f"value_{i} = {i}\r\n" for i in range(12))
        self.documents("starcoder-python", [code] * 12)
        random.seed(199)
        state = random.getstate()
        first_output, first = self.build("first", seq_len=32)
        self.assertEqual(random.getstate(), state)
        random.random()
        state = random.getstate()
        second_output, second = self.build("second", seq_len=32)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first, second)
        self.assertEqual(self.packed(first_output, first), self.packed(second_output, second))
        self.assertGreater(first["sources"]["starcoder-python"]["fim_documents"], 0)

    def test_source_routing_and_actual_post_fim_counts(self):
        code = "".join(f"value_{i} = {i}\n" for i in range(10))
        wrapped = td.apply_fim_transformation(code, fim_rate=1, rng=random.Random(7))
        sources = ["starcoder-python", "codeparrot-clean", "fineweb-edu", "glaive-function-calling", "commitpackft-python"]
        for source in sources:
            self.documents(source, [code, wrapped])
        output, manifest = self.build(seq_len=32)
        self.packed(output, manifest)
        for source in sources:
            stats = manifest["sources"][source]
            self.assertEqual(stats["documents"], 2)
            self.assertEqual(stats["preformatted_documents"], 1)
            self.assertLessEqual(stats["fim_documents"], 1)
            if source not in td.FIM_SOURCES:
                self.assertEqual(stats["fim_documents"], 0)
            self.assertEqual(stats["tokens"], len(code) + len(wrapped) + 2 + stats["fim_documents"] * sum(map(len, td.FIM_TOKENS)))

    def test_fim_round_trip_preserves_all_line_endings_and_characters(self):
        for ending in ["", "\n", "\r\n"]:
            code = "first = 1234567890\r\nsecond = 2345678901\nthird = 3456789012\rfourth = 4567890123" + ending
            transformed = td.apply_fim_transformation(code, fim_rate=1, rng=random.Random(17))
            self.assertTrue(transformed.startswith("<|fim_prefix|>"))
            prefix, remainder = transformed.removeprefix("<|fim_prefix|>").split("<|fim_suffix|>")
            suffix, middle = remainder.split("<|fim_middle|>")
            self.assertEqual(prefix + middle + suffix, code)
            self.assertEqual(td.apply_fim_transformation(transformed, fim_rate=1), transformed)

    def test_existing_output_is_untouched(self):
        self.documents("fineweb-edu", ["abc"])
        output = self.root / "out"
        output.mkdir()
        sentinel = output / "important.bin"
        sentinel.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(sentinel.read_bytes(), b"keep me")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_completed_build_cannot_be_overwritten(self):
        self.documents("fineweb-edu", ["abc"])
        output, _ = self.build()
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)

    def test_missing_inputs_and_empty_corpus_fail_before_output_creation(self):
        for dedup, tokenizer in [
            (self.root / "missing-dedup", self.tokenizer),
            (self.dedup, self.root / "missing-tokenizer"),
            (self.dedup, self.tokenizer),
        ]:
            with self.assertRaises(FileNotFoundError):
                td.build_tokenized_dataset(str(dedup), str(tokenizer), str(self.root / "out"))
            self.assertFalse((self.root / "out").exists())

    def test_tokenizer_identity_changes_with_saved_artifact(self):
        self.documents("fineweb-edu", ["abc"])
        _, first = self.build("first")
        (self.tokenizer / "tokenizer.json").write_text("{\"fixture\": false}", encoding="utf-8")
        _, second = self.build("second")
        self.assertNotEqual(first["tokenizer"]["sha256"], second["tokenizer"]["sha256"])


if __name__ == "__main__":
    unittest.main()
