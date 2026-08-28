import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
except ImportError:  # Chunking/metric tests do not need the ML runtime.
    np = None

from legalir import (
    aggregate_chunk_hits,
    checkpointed_encode,
    chunk_document,
    macro_metrics,
    normalize_text,
    split_by_documents,
    weighted_rrf,
    word_windows,
)


class LegalIRTests(unittest.TestCase):
    @unittest.skipIf(np is None, "NumPy is not installed")
    def test_dense_encoding_resumes_from_last_checkpoint(self):
        class Encoder:
            def __init__(self, fail_on_call=None):
                self.calls = 0
                self.fail_on_call = fail_on_call

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, **_kwargs):
                self.calls += 1
                if self.calls == self.fail_on_call:
                    raise RuntimeError("interrupted")
                return np.array([[len(text), self.calls] for text in texts], dtype="float32")

        with TemporaryDirectory() as directory:
            data = Path(directory) / "dense.partial"
            state = Path(directory) / "dense.json"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                checkpointed_encode(
                    Encoder(fail_on_call=2),
                    ["a", "bb", "ccc"],
                    2,
                    2,
                    data,
                    state,
                    {"model": "test"},
                )

            resumed = Encoder()
            embeddings = checkpointed_encode(
                resumed, ["a", "bb", "ccc"], 2, 2, data, state, {"model": "test"}
            )
            self.assertEqual(resumed.calls, 1)
            self.assertEqual(embeddings[:, 0].tolist(), [1.0, 2.0, 3.0])
            del embeddings

    def test_normalization_preserves_legal_markers(self):
        value = normalize_text("Điều  12\r\nKhoản\u00a03: 15/2024/NĐ-CP")
        self.assertEqual(value, "Điều 12\nKhoản 3: 15/2024/NĐ-CP")

    def test_structural_chunking_keeps_short_articles_whole(self):
        document = {
            "id": 7,
            "name": "Luật mẫu",
            "passage": "Điều 1. Phạm vi\n1. Nội dung một.\n2. Nội dung hai.\nĐiều 2. Hiệu lực\nCó hiệu lực ngay.",
        }
        chunks = chunk_document(document, max_words=100, overlap_words=10)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk["clause"] for chunk in chunks], ["", ""])
        self.assertIn("1. Nội dung một", chunks[0]["text"])
        self.assertIn("2. Nội dung hai", chunks[0]["text"])
        self.assertTrue(all(chunk["document_id"] == "7" for chunk in chunks))

    def test_long_article_packs_consecutive_clauses(self):
        document = {
            "id": 8,
            "name": "Luật mẫu",
            "passage": (
                "Điều 1. Điều kiện\n"
                "1. một hai ba bốn năm sáu bảy.\n"
                "2. tám chín mười mười một mười hai.\n"
                "3. mười ba mười bốn mười lăm mười sáu.\n"
                "4. mười bảy mười tám mười chín hai mươi."
            ),
        }
        chunks = chunk_document(document, max_words=18, overlap_words=4)
        self.assertEqual([chunk["clause"] for chunk in chunks], ["Khoản 1–2", "Khoản 3–4"])
        self.assertTrue(all(len(chunk["text"].split()) <= 18 for chunk in chunks))

    def test_article_regex_rejects_wrapped_inline_reference(self):
        document = {
            "id": 9,
            "name": "Quyết định mẫu",
            "passage": (
                "Điều 1. Trách nhiệm\nNội dung chính.\n"
                "Điều 2 Quyết định này được gửi tới cơ quan liên quan.\n"
                "Điều 2. Hiệu lực\nCó hiệu lực từ ngày ký."
            ),
        }
        chunks = chunk_document(document, max_words=100, overlap_words=10)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Điều 2 Quyết định này", chunks[0]["text"])
        self.assertTrue(chunks[1]["article"].startswith("Điều 2."))

    def test_attachment_chapter_path_and_recipient_removal(self):
        document = {
            "id": 10,
            "name": "Quyết định mẫu",
            "passage": (
                "Điều 1. Ban hành\nNội dung chính.\nNơi nhận:\n- Cơ quan A\n"
                "QUY CHẾ\nTên quy chế mẫu\nChương 1.\nQUY ĐỊNH CHUNG\n"
                "Điều 1. Phạm vi\nNội dung quy chế."
            ),
        }
        chunks = chunk_document(document, max_words=100, overlap_words=10)
        self.assertEqual(len(chunks), 2)
        self.assertNotIn("Nơi nhận", chunks[0]["text"])
        self.assertIn("QUY CHẾ", chunks[1]["heading"])
        self.assertIn("Chương 1. QUY ĐỊNH CHUNG", chunks[1]["heading"])

    def test_word_windows_omits_overlap_only_tail(self):
        windows = word_windows(" ".join(map(str, range(545))), 320, 48)
        self.assertEqual([len(window.split()) for window in windows], [320, 273])
        self.assertEqual(windows[-1].split()[-1], "544")

    def test_top_n_average_and_rrf(self):
        chunks = [
            {"document_id": "a"},
            {"document_id": "a"},
            {"document_id": "b"},
        ]
        ranking = aggregate_chunk_hits([0, 1, 2], [0.9, 0.7, 0.75], chunks, top_n=2)
        self.assertEqual(ranking, [("a", 0.8), ("b", 0.75)])
        fused = weighted_rrf(
            [(1.0, ranking), (1.0, [("b", 1.0), ("a", 0.5)])], k=60
        )
        self.assertEqual({item[0] for item in fused}, {"a", "b"})

    def test_split_preserves_duplicates_and_keeps_their_documents_together(self):
        data = {
            "1": {"question": "Câu hỏi?", "answer": ["a"]},
            "2": {"question": " Câu   hỏi? ", "answer": ["b"]},
            "3": {"question": "Khác", "answer": ["c"]},
            "4": {"question": "Khác nữa", "answer": ["d"]},
        }
        train, validation, manifest = split_by_documents(data, 0.5, seed=1)
        split = {**train, **validation}
        self.assertEqual(split, data)
        self.assertEqual("1" in train, "2" in train)
        self.assertFalse(
            {answer for item in train.values() for answer in item["answer"]}
            & {answer for item in validation.values() for answer in item["answer"]}
        )
        self.assertEqual(len(validation), 2)
        self.assertEqual(manifest["question_groups"], 3)
        self.assertEqual(manifest["actual_validation_ratio"], 0.5)

    def test_official_metrics_and_over_five_rule(self):
        gold = {
            "q1": {"answer": ["a", "b"]},
            "q2": {"answer": ["x"]},
        }
        result = macro_metrics(
            gold,
            {"q1": ["a", "c"], "q2": ["x", "1", "2", "3", "4", "5"]},
        )
        self.assertEqual(result["macro_recall"], 0.25)
        self.assertEqual(result["macro_precision"], 0.25)


if __name__ == "__main__":
    unittest.main()
