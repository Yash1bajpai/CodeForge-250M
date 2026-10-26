import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data.prepare_fresh import lexical_windows, partition_documents, source_files


class FreshPreparationTests(unittest.TestCase):
    def test_long_reference_copy_detected_without_rejecting_short_idiom(self):
        reference = "def compute(values):\n" + "\n".join(
            f"    result_{i} = values[{i}] * {i + 3}" for i in range(12))
        index = set(lexical_windows(reference, 32, 12))
        copied = "# extra documentation\n" + reference.replace(" = ", "  =  ")
        self.assertTrue(any(window in index for window in lexical_windows(copied, 32, 12)))
        self.assertFalse(any(window in index for window in lexical_windows("return value + 1", 32, 12)))
        different = "def reverse_words(sentence):\n    return ' '.join(reversed(sentence.split()))"
        self.assertFalse(any(window in index for window in lexical_windows(different, 32, 12)))

    def test_document_groups_stay_in_one_reproducible_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dedup = root / "dedup"
            dedup.mkdir()
            records = [{"text": f"def function_{i}(value):\n    return value + {i}\n", "license": "mit", "id": i}
                       for i in range(80)]
            records.append({"text": records[0]["text"].replace("    ", "        "), "license": "mit", "id": "same-group"})
            (dedup / "codeparrot-clean_dedup.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            mappings = []
            for number in range(2):
                train, val = root / f"train{number}", root / f"val{number}"
                partition_documents(dedup, train, val, 0.2, 42)
                membership = {}
                for split, path in (("train", train), ("validation", val)):
                    for line in (path / "codeparrot-clean_dedup.jsonl").read_text(encoding="utf-8").splitlines():
                        record = json.loads(line)
                        self.assertEqual(record["license"], "mit")
                        membership[str(record["id"])] = split
                self.assertEqual(membership["0"], membership["same-group"])
                self.assertEqual(set(membership), {str(record["id"]) for record in records})
                mappings.append(membership)
            self.assertEqual(mappings[0], mappings[1])

    def test_unpinned_source_rejected_before_network(self):
        with self.assertRaises(ValueError):
            source_files({"repo": "unused/source", "revision": "main", "file_pattern": "*.parquet"})

    def test_fresh_training_requires_approval_before_any_runtime_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"model": {}, "training": {"require_explicit_approval": True}}))
            code = (
                "import sys; from pathlib import Path; import training.train as trainer; "
                "trainer.PROJECT_ROOT=sys.argv[1]; "
                "sys.argv=['train.py','--config',str(Path(sys.argv[1])/'config.json'),'--from_scratch']; "
                "trainer.train()"
            )
            env = os.environ.copy()
            env.pop("CF_TRAINING_APPROVED", None)
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            result = subprocess.run([sys.executable, "-B", "-c", code, directory], env=env,
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertFalse((root / "training.log").exists())
            self.assertFalse((root / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
