import json
import tempfile
import unittest
import zipfile
from collections import UserDict
from pathlib import Path

from train_qlora import (
    DEFAULT_MODEL,
    build_parser,
    build_pair_rows,
    candidate_ids,
    candidate_records,
    fused_document_ranking,
    pair_ranking_metrics,
    prompt_token_ids,
    write_submission,
)
from legalir import build_parser as build_legalir_parser, select_document_chunks


class QLoRARerankerTests(unittest.TestCase):
    def test_chunk_candidate_cli_defaults(self):
        args = build_legalir_parser().parse_args(["candidates"])
        self.assertEqual(
            (
                args.chunk_heading_weight,
                args.chunk_body_weight,
                args.chunk_dense_weight,
                args.chunk_rrf_k,
                args.chunk_threshold,
                args.chunks_per_document,
            ),
            (0.5, 2.0, 4.0, 4, 0.4, 2),
        )

    def test_candidates_select_at_most_two_chunks_above_threshold(self):
        chunks = [
            {"row_id": index, "chunk_id": str(index), "text": str(index)}
            for index in range(14)
        ]
        scores = {index: 14 - index for index in range(14)}
        selected = select_document_chunks(
            chunks,
            scores,
            scores,
            scores,
        )
        above_threshold = select_document_chunks(
            chunks, scores, scores, scores, limit=14
        )
        fallback = select_document_chunks(
            chunks, scores, scores, scores, threshold=2.0
        )
        self.assertEqual([item["chunk_id"] for item in selected], ["0", "1"])
        self.assertEqual(len(above_threshold), 12)
        self.assertTrue(all(item["score"] >= 0.4 for item in above_threshold))
        self.assertEqual([item["chunk_id"] for item in fallback], ["0"])

    def test_distributed_launcher_rank_argument_is_accepted(self):
        args = build_parser().parse_args(
            ["--local-rank=1", "train", "--train-pairs", "train.jsonl"]
        )
        self.assertEqual(args.local_rank, 1)

    def test_fusion_cli_defaults_and_overrides(self):
        train = build_parser().parse_args(["train", "--train-pairs", "train.jsonl"])
        rerank = build_parser().parse_args(
            [
                "rerank",
                "--questions",
                "questions.json",
                "--candidates",
                "candidates.json",
                "--output",
                "output.json",
                "--qwen-weight",
                "7",
                "--retrieval-weight",
                "3",
            ]
        )
        self.assertEqual((train.qwen_weight, train.retrieval_weight), (0.6, 0.4))
        self.assertEqual((rerank.qwen_weight, rerank.retrieval_weight), (7.0, 3.0))

    def test_chat_template_batch_encoding_is_unwrapped(self):
        class Tokenizer:
            def __call__(self, *_args, **_kwargs):
                return {"input_ids": [1]}

            def decode(self, *_args, **_kwargs):
                return "document"

            def apply_chat_template(self, *_args, **_kwargs):
                return UserDict({"input_ids": [10, 11]})

        self.assertEqual(prompt_token_ids(Tokenizer(), "question", "text", 256, 1), [10, 11])

    def test_qwen3_06b_submission_zip(self):
        self.assertEqual(DEFAULT_MODEL, "Qwen/Qwen3-0.6B")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.json"
            zip_path = write_submission(path, {"q1": ["a"]})
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])
                self.assertEqual(
                    json.loads(archive.read("submission.json")),
                    {"q1": {"answer": ["a"]}},
                )

    def test_candidate_formats_and_pair_labels(self):
        self.assertEqual(candidate_ids({"answer": [1, "2", 1]}), ["1", "2"])
        questions = {
            "q1": {"question": "Mức kỷ luật được giảm nhẹ thế nào?", "answer": ["a"]}
        }
        candidates = [
            {
                "document_id": "a",
                "retrieved": False,
                "document_score": None,
                "chunks": [
                    {
                        "chunk_id": "a:0",
                        "text": "Mức kỷ luật được xem xét giảm nhẹ.",
                        "score": 0.5,
                    }
                ],
            },
            {
                "document_id": "b",
                "retrieved": True,
                "document_score": 0.8,
                "chunks": [
                    {"chunk_id": "b:0", "text": "Quy định không liên quan.", "score": 0.5}
                ],
            },
            {
                "document_id": "c",
                "retrieved": True,
                "document_score": 0.6,
                "chunks": [
                    {"chunk_id": "c:0", "text": "Nội dung khác.", "score": 0.41}
                ],
            },
        ]
        rows, manifest = build_pair_rows(
            questions, {"q1": candidate_records({"candidates": candidates})}
        )

        self.assertEqual({row["label"] for row in rows}, {0, 5})
        positive = next(row for row in rows if row["label"] == 5)
        self.assertEqual(positive["chunk_id"], "a:0")
        self.assertFalse(positive["retrieved"])
        self.assertTrue(next(row for row in rows if row["label"] == 0)["retrieved"])
        self.assertEqual(manifest["positive_pairs"], 1)
        self.assertEqual(manifest["negative_pairs"], 2)

    def test_candidate_aware_recall_ignores_injected_gold(self):
        rows = [
            {"question_id": "q1", "document_id": "a", "label": 5, "retrieved": True, "retrieval_score": 0.8},
            {"question_id": "q1", "document_id": "b", "label": 0, "retrieved": True, "retrieval_score": 0.2},
            {"question_id": "q2", "document_id": "c", "label": 5, "retrieved": False, "retrieval_score": None},
            {"question_id": "q2", "document_id": "d", "label": 0, "retrieved": True, "retrieval_score": 0.7},
        ]
        self.assertEqual(
            pair_ranking_metrics(rows, [0.9, 0.1, 1.0, 0.8], top_k=1),
            {"macro_recall_at_1": 0.5, "macro_precision_at_1": 0.5},
        )

    def test_final_score_fuses_normalized_qwen_and_retrieval(self):
        items = [
            {"document_id": "a", "chunk_id": "a:0", "score": 4.5, "retrieved": True, "retrieval_score": 0.1},
            {"document_id": "a", "chunk_id": "a:1", "score": 4.0, "retrieved": True, "retrieval_score": 0.1},
            {"document_id": "b", "chunk_id": "b:0", "score": 3.5, "retrieved": True, "retrieval_score": 0.9},
        ]
        ranking = fused_document_ranking(items)
        self.assertEqual([item["document_id"] for item in ranking], ["b", "a"])
        self.assertAlmostEqual(ranking[0]["qwen_score_normalized"], 0.7)
        self.assertAlmostEqual(ranking[0]["retrieval_score_normalized"], 1.0)
        self.assertAlmostEqual(ranking[0]["score"], 0.82)
        self.assertEqual(ranking[1]["chunk_id"], "a:0")
        self.assertAlmostEqual(fused_document_ranking(items, 6, 4)[0]["score"], 0.82)
        self.assertEqual(fused_document_ranking(items, 1, 0)[0]["document_id"], "a")


if __name__ == "__main__":
    unittest.main()
