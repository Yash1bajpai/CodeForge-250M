import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from data.deduplicate import deduplicate_dataset
from data.filter_quality import filter_code_quality


def write_records(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class QualityFilterTests(unittest.TestCase):
    def test_python_ast_keeps_brackets_in_strings_and_comments(self):
        valid = {
            "text": 'def bracket_label():\n    text = "] unmatched string"\n    # ( unmatched comment\n    return text\n',
            "source": "original-source",
            "license": "MIT",
            "metadata": {"document_id": 17},
        }
        invalid = {
            "text": "def broken_function(:\n    return an_undefined_long_variable\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            write_records(raw / "starcoder-python_part_one_raw.jsonl", [valid, invalid])
            filter_code_quality(str(raw), str(root / "filtered"))
            self.assertEqual(
                read_records(root / "filtered" / "starcoder-python_part_one_filtered.jsonl"),
                [valid],
            )

    def test_non_python_formats_and_metadata_are_preserved(self):
        records = {
            "commitpackft-python": {
                "text": "<|fim_prefix|>old syntax is incomplete (\n<|fim_middle|>fixed syntax is incomplete ]",
                "format": "fim",
            },
            "starcoder-python": {
                "text": "<|fim_prefix|>def partial_function(\n<|fim_middle|>argument): return argument",
                "format": "fim",
            },
            "evol-codealpaca": {
                "text": "### Instruction:\nExplain what an unmatched bracket ] means in text.",
                "format": "instruction",
            },
            "glaive-function-calling": {
                "text": "SYSTEM: A helpful assistant\nUSER: Explain this bracket ] without running tools.",
                "format": "tool",
            },
            "fineweb-edu": {
                "text": "A sentence about unmatched brackets like ] in ordinary writing.\nAnother sentence for context.",
                "format": "prose",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for source, record in records.items():
                write_records(raw / f"{source}_raw.jsonl", [record])
            filter_code_quality(str(raw), str(root / "filtered"))
            for source, record in records.items():
                with self.subTest(source=source):
                    self.assertEqual(
                        read_records(root / "filtered" / f"{source}_filtered.jsonl"), [record]
                    )

    def test_empty_inputs_or_no_admitted_records_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            with self.assertRaises(ValueError):
                filter_code_quality(str(raw), str(root / "filtered"))
            (raw / "python_raw.jsonl").write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                filter_code_quality(str(raw), str(root / "filtered"))
            write_records(raw / "python_raw.jsonl", [
                {"text": "def invalid_function(:\n    return a_long_variable_name\n"}
            ])
            with self.assertRaises(ValueError):
                filter_code_quality(str(raw), str(root / "filtered"))


class DeduplicationTests(unittest.TestCase):
    def test_cross_source_exact_and_high_jaccard_near_duplicates(self):
        base = " ".join(f"word{i}" for i in range(100))
        near = base + " " + " ".join(f"extra{i}" for i in range(5))
        # The appended words add five five-grams to the original 96: J = 96/101.
        self.assertGreater(96 / 101, 0.95)
        first = {"text": base, "source": "first", "license": "MIT"}
        short_first = {"text": "return first", "document_id": 1}
        distinct = {"text": " ".join(f"different{i}" for i in range(100)), "source": "second"}
        short_second = {"text": "return second", "document_id": 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filtered = root / "filtered"
            filtered.mkdir()
            # Creation order is deliberately opposite the required source order.
            write_records(filtered / "z_source_filtered.jsonl", [
                {"text": base.replace(" ", "\n\t"), "source": "later-exact"},
                {"text": near, "source": "near"},
                distinct,
                short_second,
                {"text": "return\tfirst", "document_id": 3},
            ])
            write_records(filtered / "a_source_filtered.jsonl", [first, short_first])
            stats = deduplicate_dataset(str(filtered), str(root / "dedup"))
            self.assertEqual(
                read_records(root / "dedup" / "a_source_dedup.jsonl"), [first, short_first]
            )
            self.assertEqual(
                read_records(root / "dedup" / "z_source_dedup.jsonl"), [distinct, short_second]
            )
            self.assertEqual(stats["totals"], {
                "input_records": 7, "invalid_records": 0, "exact_removed": 2,
                "near_removed": 1, "kept": 4,
            })
            self.assertEqual(stats["parameters"]["threshold"], 0.85)
            self.assertEqual(stats["parameters"]["num_perm"], 128)
            self.assertEqual(stats["parameters"]["seed"], 1)
            self.assertEqual(stats["parameters"]["shingle_size_words"], 5)
            self.assertEqual(
                json.loads((root / "dedup" / "dedup_stats.json").read_text(encoding="utf-8")),
                stats,
            )
            repeat_stats = deduplicate_dataset(str(filtered), str(root / "repeat"))
            self.assertEqual(stats, repeat_stats)
            for source in ("a_source", "z_source"):
                self.assertEqual(
                    read_records(root / "dedup" / f"{source}_dedup.jsonl"),
                    read_records(root / "repeat" / f"{source}_dedup.jsonl"),
                )

    def test_lsh_candidates_require_actual_jaccard_similarity(self):
        records = [
            {"text": "alpha beta gamma delta epsilon zeta", "tag": "first"},
            {"text": "alpha beta gamma delta epsilon omega", "tag": "distinct"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filtered = root / "filtered"
            filtered.mkdir()
            write_records(filtered / "python_filtered.jsonl", records)
            with mock.patch("data.deduplicate.MinHashLSH.query", side_effect=[[], [0]]):
                stats = deduplicate_dataset(str(filtered), str(root / "dedup"))
            self.assertEqual(read_records(root / "dedup" / "python_dedup.jsonl"), records)
            self.assertEqual(stats["totals"]["near_removed"], 0)

    def test_empty_or_nontext_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filtered = root / "filtered"
            filtered.mkdir()
            with self.assertRaises(ValueError):
                deduplicate_dataset(str(filtered), str(root / "dedup"))
            (filtered / "empty_filtered.jsonl").write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                deduplicate_dataset(str(filtered), str(root / "dedup"))
            write_records(filtered / "empty_filtered.jsonl", [
                {"text": "  "}, {"text": None}, {"missing": "text"}, ["not", "a record"]
            ])
            with self.assertRaises(ValueError):
                deduplicate_dataset(str(filtered), str(root / "dedup"))


if __name__ == "__main__":
    unittest.main()
