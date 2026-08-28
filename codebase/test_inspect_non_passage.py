import json
import tempfile
import unittest
from pathlib import Path

from crawl_non_passage import (
    has_passage,
    is_bot_verification,
    normalize_passage,
    save_recovered_context,
)
from inspect_non_passage import inspect


class InspectNonPassageTest(unittest.TestCase):
    def test_finds_non_passages_and_joins_train_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts"
            contexts.mkdir()
            train = {
                "q1": {"question": "Question one?", "answer": [1, "3"]},
                "q2": {"question": "Question two?", "answer": None},
            }
            (root / "train.json").write_text(json.dumps(train), encoding="utf-8")
            samples = [
                {"id": 1, "passage": "", "link": "https://example/1"},
                {"id": 2, "passage": "usable"},
                {"id": 3, "link": "https://example/3"},
                {"id": 4, "passage": None},
            ]
            for sample in samples:
                (contexts / f"context_{sample['id']}.json").write_text(
                    json.dumps(sample), encoding="utf-8"
                )

            report = inspect(root / "train.json", contexts)

        self.assertEqual(
            report["summary"],
            {
                "selected_context_files": 4,
                "non_passage_contexts": 3,
                "referenced_by_train": 2,
                "train_references": 2,
            },
        )
        self.assertEqual(
            [item["passage_status"] for item in report["contexts"]],
            ["blank", "missing", "null"],
        )
        self.assertEqual(report["contexts"][1]["train_references"][0]["question_id"], "q1")

    def test_normalizes_and_saves_recovered_passage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            destination = root / "output" / "context_1.json"
            source.write_text(
                json.dumps({"id": 1, "link": "https://example/1", "passage": ""}),
                encoding="utf-8",
            )

            passage = normalize_passage("  Điều 1.\r\n\r\n\r\nNội dung.\xa0  ")
            save_recovered_context(source, destination, passage)

            self.assertEqual(passage, "Điều 1.\n\nNội dung.")
            self.assertTrue(has_passage(destination))
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8"))["passage"], passage
            )

    def test_detects_bot_verification_without_accepting_normal_content(self):
        self.assertTrue(
            is_bot_verification(
                "Just a moment...", "Performing security verification"
            )
        )
        self.assertFalse(is_bot_verification("Legal document", "Điều 1. Nội dung"))


if __name__ == "__main__":
    unittest.main()
