"""QLoRA pointwise reranker for Vietnamese LegalIR candidates."""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
QWEN_WEIGHT = 0.6
RETRIEVAL_WEIGHT = 0.4
SYSTEM_PROMPT = """Bạn là bộ xếp hạng văn bản pháp luật tiếng Việt.
Đánh giá mức độ đoạn văn chứa thông tin cần thiết để trả lời câu hỏi:
0 = hoàn toàn không liên quan; 1 = liên quan rất ít; 2 = có liên quan nhưng không trả lời;
3 = trả lời một phần; 4 = gần đầy đủ; 5 = trả lời trực tiếp và đầy đủ.
Chỉ trả về đúng một chữ số từ 0 đến 5, không giải thích."""


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain an object")
                rows.append(value)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_ids(value: object) -> list[str]:
    if isinstance(value, dict):
        for key in ("candidates", "answer", "documents"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError("Each candidate entry must be a list or contain candidates/answer")
    result = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("document_id", item.get("id"))
        if item is not None and str(item) not in result:
            result.append(str(item))
    return result


def candidate_records(value: object) -> list[dict]:
    if isinstance(value, dict):
        value = value.get("candidates", value.get("documents", value.get("answer")))
    if not isinstance(value, list):
        raise ValueError("Each candidate entry must contain a candidates list")
    records = []
    for item in value:
        if not isinstance(item, dict) or not item.get("document_id"):
            raise ValueError(
                "Candidates must contain document/chunk objects; regenerate with legalir.py candidates"
            )
        chunks = item.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError(f"Candidate document {item['document_id']} has no chunks list")
        if any(
            not isinstance(chunk, dict)
            or not chunk.get("chunk_id")
            or not isinstance(chunk.get("text"), str)
            for chunk in chunks
        ):
            raise ValueError(f"Candidate document {item['document_id']} has an invalid chunk")
        records.append(
            {
                **item,
                "document_id": str(item["document_id"]),
                "retrieved": bool(item.get("retrieved", True)),
                "chunks": chunks,
            }
        )
    return records


def load_candidate_map(path: Path) -> dict[str, list[dict]]:
    return {str(key): candidate_records(value) for key, value in read_json(path).items()}


def build_pair_rows(
    questions: dict,
    candidates: dict[str, list[dict]],
) -> tuple[list[dict], dict]:
    question_map = {str(key): value for key, value in questions.items()}
    missing = sorted(set(question_map) - set(candidates))
    if missing:
        raise ValueError(f"Candidates are missing {len(missing)} questions; first: {missing[0]}")
    rows = []
    documents_without_chunks = 0
    for question_id, item in question_map.items():
        positives = {str(value) for value in item.get("answer") or []}
        if not positives:
            raise ValueError(f"Question {question_id} has no positive answer")
        document_ids = {candidate["document_id"] for candidate in candidates[question_id]}
        missing_positives = sorted(positives - document_ids)
        if missing_positives:
            raise ValueError(
                f"Candidates for {question_id} are missing gold documents: "
                f"{', '.join(missing_positives)}; regenerate with legalir.py candidates"
            )
        for candidate in candidates[question_id]:
            document_id = candidate["document_id"]
            label = 5 if document_id in positives else 0
            if not candidate["chunks"]:
                documents_without_chunks += 1
            for chunk in candidate["chunks"]:
                rows.append(
                    {
                        "question_id": question_id,
                        "document_id": document_id,
                        "chunk_id": chunk["chunk_id"],
                        "question": str(question_map[question_id].get("question") or ""),
                        "text": chunk["text"],
                        "label": label,
                        "retrieved": candidate["retrieved"],
                        "retrieval_score": candidate.get("document_score"),
                        "selection_score": chunk.get("score"),
                    }
                )
    manifest = {
        "questions": len(question_map),
        "pairs": len(rows),
        "positive_pairs": sum(row["label"] == 5 for row in rows),
        "negative_pairs": sum(row["label"] == 0 for row in rows),
        "documents_without_chunks": documents_without_chunks,
        "chunk_selection": "provided_by_legalir_candidates",
        "labels": {"not_relevant": 0, "relevant": 5},
        "note": "Grades 1-4 are unsupervised because train.json only has document-level positives.",
    }
    return rows, manifest


def command_prepare(args: argparse.Namespace) -> None:
    rows, manifest = build_pair_rows(
        read_json(args.questions),
        load_candidate_map(args.candidates),
    )
    random.Random(args.seed).shuffle(rows)
    write_jsonl(args.output, rows)
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def user_prompt(question: str, text: str) -> str:
    return f"Câu hỏi:\n{question.strip()}\n\nĐoạn văn ứng viên:\n{text.strip()}\n\nĐiểm:"


def prompt_token_ids(tokenizer, question: str, text: str, max_length: int, reserve: int) -> list[int]:
    document_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while True:
        clipped = tokenizer.decode(document_ids, skip_special_tokens=True)
        ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, clipped)},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        overflow = len(ids) + reserve - max_length
        if overflow <= 0:
            return ids
        if not document_ids:
            raise ValueError(f"max_length={max_length} is too small for the prompt")
        document_ids = document_ids[: max(0, len(document_ids) - overflow - 8)]


class PairDataset:
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        target = self.tokenizer(str(int(row["label"])), add_special_tokens=False)["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            target = target + [self.tokenizer.eos_token_id]
        prompt = prompt_token_ids(
            self.tokenizer, row["question"], row["text"], self.max_length, len(target)
        )
        return {
            "input_ids": prompt + target,
            "attention_mask": [1] * (len(prompt) + len(target)),
            "labels": [-100] * len(prompt) + target,
        }


class PairCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        length = max(len(item["input_ids"]) for item in features)
        return {
            "input_ids": torch.tensor(
                [item["input_ids"] + [self.pad_token_id] * (length - len(item["input_ids"])) for item in features]
            ),
            "attention_mask": torch.tensor(
                [item["attention_mask"] + [0] * (length - len(item["attention_mask"])) for item in features]
            ),
            "labels": torch.tensor(
                [item["labels"] + [-100] * (length - len(item["labels"])) for item in features]
            ),
        }


def load_tokenizer(source: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def inference_model(model_name: str, adapter: Path | None, max_length: int):
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter) if adapter else model_name,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=torch.cuda.is_available(),
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def fused_document_ranking(
    items: list[dict],
    qwen_weight: float = QWEN_WEIGHT,
    retrieval_weight: float = RETRIEVAL_WEIGHT,
) -> list[dict]:
    """Max-pool Qwen chunks, min-max retrieval scores, then combine both signals."""
    if (
        not math.isfinite(qwen_weight)
        or not math.isfinite(retrieval_weight)
        or qwen_weight < 0.0
        or retrieval_weight < 0.0
        or qwen_weight + retrieval_weight <= 0.0
    ):
        raise ValueError("Fusion weights must be finite, non-negative, and not both zero")
    weight_sum = qwen_weight + retrieval_weight
    qwen_weight /= weight_sum
    retrieval_weight /= weight_sum
    best_qwen: dict[str, dict] = {}
    retrieval_scores: dict[str, float] = {}
    for item in items:
        if not item.get("retrieved", True):
            continue
        document_id = str(item["document_id"])
        qwen_score = float(item["score"])
        retrieval_score = item.get("retrieval_score")
        if retrieval_score is None:
            raise ValueError(
                "Retrieved candidates need document_score; regenerate candidates/pairs"
            )
        retrieval_score = float(retrieval_score)
        if not math.isfinite(qwen_score) or not 0.0 <= qwen_score <= 5.0:
            raise ValueError(f"Invalid Qwen score for document {document_id}: {qwen_score}")
        if not math.isfinite(retrieval_score):
            raise ValueError(
                f"Invalid retrieval score for document {document_id}: {retrieval_score}"
            )
        previous_retrieval = retrieval_scores.get(document_id)
        if previous_retrieval is not None and not math.isclose(
            previous_retrieval, retrieval_score
        ):
            raise ValueError(f"Inconsistent retrieval scores for document {document_id}")
        retrieval_scores[document_id] = retrieval_score
        if document_id not in best_qwen or qwen_score > best_qwen[document_id]["score"]:
            best_qwen[document_id] = {**item, "document_id": document_id, "score": qwen_score}

    if not best_qwen:
        return []
    low, high = min(retrieval_scores.values()), max(retrieval_scores.values())
    span = high - low
    ranking = []
    for document_id, item in best_qwen.items():
        qwen_score = float(item["score"])
        retrieval_score = retrieval_scores[document_id]
        qwen_normalized = qwen_score / 5.0
        retrieval_normalized = (
            (retrieval_score - low) / span if span > 0.0 else 1.0
        )
        ranking.append(
            {
                **item,
                "qwen_score": qwen_score,
                "qwen_score_normalized": qwen_normalized,
                "retrieval_score": retrieval_score,
                "retrieval_score_normalized": retrieval_normalized,
                "score": (
                    qwen_weight * qwen_normalized
                    + retrieval_weight * retrieval_normalized
                ),
            }
        )
    return sorted(ranking, key=lambda item: (-item["score"], item["document_id"]))


def pair_ranking_metrics(
    rows: list[dict],
    scores: list[float],
    top_k: int,
    qwen_weight: float = QWEN_WEIGHT,
    retrieval_weight: float = RETRIEVAL_WEIGHT,
) -> dict[str, float]:
    if len(rows) != len(scores):
        raise ValueError("Validation rows and scores must have the same length")
    relevant, candidates = defaultdict(set), defaultdict(list)
    for row, score in zip(rows, scores):
        question_id = str(row["question_id"])
        document_id = str(row["document_id"])
        if int(row["label"]) == 5:
            relevant[question_id].add(document_id)
        candidates[question_id].append({**row, "document_id": document_id, "score": score})

    recalls, precisions = [], []
    for question_id, gold in relevant.items():
        prediction = {
            item["document_id"]
            for item in fused_document_ranking(
                candidates[question_id], qwen_weight, retrieval_weight
            )[:top_k]
        }
        hits = len(gold & prediction)
        recalls.append(hits / len(gold))
        precisions.append(hits / len(prediction) if prediction else 0.0)
    return {
        f"macro_recall_at_{top_k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"macro_precision_at_{top_k}": sum(precisions) / len(precisions) if precisions else 0.0,
    }


def metric_functions(
    rows: list[dict],
    tokenizer,
    top_k: int,
    qwen_weight: float,
    retrieval_weight: float,
):
    import torch

    grade_ids = [tokenizer(str(grade), add_special_tokens=False)["input_ids"] for grade in range(6)]
    if any(len(ids) != 1 for ids in grade_ids):
        raise ValueError("This tokenizer does not encode grades 0-5 as single tokens")
    grade_token_ids = [ids[0] for ids in grade_ids]

    def preprocess_logits(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        grade_positions = (labels != -100).int().argmax(dim=-1) - 1
        batch_indices = torch.arange(labels.shape[0], device=labels.device)
        grade_logits = logits[batch_indices, grade_positions][:, grade_token_ids].float()
        grades = torch.arange(6, device=grade_logits.device, dtype=grade_logits.dtype)
        return grade_logits.softmax(dim=-1) @ grades

    def compute_metrics(prediction):
        return pair_ranking_metrics(
            rows,
            prediction.predictions.reshape(-1).tolist(),
            top_k,
            qwen_weight,
            retrieval_weight,
        )

    return preprocess_logits, compute_metrics


def command_train(args: argparse.Namespace) -> None:
    import os

    # Recall@k evaluation needs grade logits; Unsloth suppresses them by default.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    import torch
    from unsloth import FastLanguageModel
    from unsloth.models.loader_utils import prepare_device_map
    from transformers import Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("Unsloth QLoRA training requires a CUDA GPU")
    set_seed(args.seed)
    device_map, _ = prepare_device_map()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
        device_map=device_map,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.print_trainable_parameters()

    train_dataset = PairDataset(read_jsonl(args.train_pairs), tokenizer, args.max_length)
    validation_rows = (
        read_jsonl(args.validation_pairs)
        if args.validation_pairs and args.validation_pairs.exists()
        else None
    )
    if validation_rows and any(
        "retrieved" not in row or "retrieval_score" not in row
        for row in validation_rows
    ):
        raise ValueError(
            "Regenerate validation pairs to include retrieval status and score"
        )
    eval_dataset = (
        PairDataset(validation_rows, tokenizer, args.max_length) if validation_rows else None
    )
    preprocess_logits, compute_metrics = (
        metric_functions(
            validation_rows,
            tokenizer,
            args.metric_top_k,
            args.qwen_weight,
            args.retrieval_weight,
        )
        if validation_rows
        else (None, None)
    )
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model=(
            f"eval_macro_recall_at_{args.metric_top_k}" if eval_dataset else None
        ),
        greater_is_better=True if eval_dataset else None,
        save_total_limit=2,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PairCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits,
        compute_metrics=compute_metrics,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_adapter = args.output / "final_adapter"
    trainer.save_model(str(final_adapter))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_adapter)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        write_json(
            args.output / "run_config.json",
            {key: value for key, value in vars(args).items() if key != "run"},
        )


def label_probabilities(model, tokenizer, rows: list[dict], max_length: int, batch_size: int):
    import torch

    label_ids = [tokenizer(str(grade), add_special_tokens=False)["input_ids"] for grade in range(6)]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("This tokenizer does not encode grades 0-5 as single tokens")
    grade_token_ids = [ids[0] for ids in label_ids]
    device = model.get_input_embeddings().weight.device
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        sequences = []
        for row in batch:
            sequences.append(
                prompt_token_ids(tokenizer, row["question"], row["text"], max_length, 1)
            )
        width = max(map(len, sequences))
        padded = [
            [tokenizer.pad_token_id] * (width - len(sequence)) + sequence
            for sequence in sequences
        ]
        input_ids = torch.tensor(padded, device=device)
        attention_mask = torch.tensor(
            [[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences],
            device=device,
        )
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                logits_to_keep=1,
                use_cache=False,
            ).logits[:, -1, grade_token_ids].float()
        probabilities = logits.softmax(dim=-1).cpu()
        for row, probs in zip(batch, probabilities.tolist()):
            yield sum(grade * probability for grade, probability in enumerate(probs)), probs


def macro_metrics(gold: dict, predictions: dict[str, list[str]]) -> dict[str, float]:
    recalls, precisions = [], []
    for question_id, item in gold.items():
        relevant = {str(value) for value in item.get("answer") or []}
        predicted = set(predictions.get(str(question_id), []))
        hits = len(relevant & predicted)
        recalls.append(hits / len(relevant) if relevant else 0.0)
        precisions.append(hits / len(predicted) if predicted else 0.0)
    return {
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
    }


def write_submission(path: Path, predictions: dict[str, list[str]]) -> Path:
    write_json(
        path,
        {question_id: {"answer": answer} for question_id, answer in predictions.items()},
    )
    zip_path = path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, "submission.json")
    return zip_path


def command_rerank(args: argparse.Namespace) -> None:
    from tqdm.auto import tqdm

    questions = {str(key): value for key, value in read_json(args.questions).items()}
    candidates = load_candidate_map(args.candidates)
    missing = sorted(set(questions) - set(candidates))
    if missing:
        raise ValueError(f"Candidates are missing {len(missing)} questions; first: {missing[0]}")
    rows = []
    for question_id, document_candidates in candidates.items():
        question = str(questions[question_id].get("question") or "")
        for candidate in document_candidates:
            for chunk in candidate["chunks"]:
                rows.append(
                    {
                        "question_id": question_id,
                        "document_id": candidate["document_id"],
                        "chunk_id": chunk["chunk_id"],
                        "question": question,
                        "text": chunk["text"],
                        "retrieved": candidate["retrieved"],
                        "retrieval_score": candidate.get("document_score"),
                        "selection_score": chunk.get("score"),
                    }
                )

    model, tokenizer = inference_model(args.model, args.adapter, args.max_length)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    scored = defaultdict(list)
    for row, (score, probabilities) in zip(
        rows,
        tqdm(
            label_probabilities(model, tokenizer, rows, args.max_length, args.batch_size),
            total=len(rows),
            desc="Reranking",
            unit="pair",
        ),
    ):
        scored[row["question_id"]].append(
            {
                "document_id": row["document_id"],
                "chunk_id": row["chunk_id"],
                "score": score,
                "retrieved": row["retrieved"],
                "retrieval_score": row["retrieval_score"],
                "selection_score": row["selection_score"],
                "grade_probabilities": probabilities,
            }
        )

    output, predictions = {}, {}
    for question_id in candidates:
        ranking = fused_document_ranking(
            scored[question_id], args.qwen_weight, args.retrieval_weight
        )
        answer = [item["document_id"] for item in ranking[: args.top_k]]
        predictions[question_id] = answer
        output[question_id] = {"answer": answer, "candidates": ranking}
    write_json(args.output, output)
    if args.submission:
        zip_path = write_submission(args.submission, predictions)
        print(f"Wrote submission -> {args.submission} and {zip_path}")

    if all(item.get("answer") is not None for item in questions.values()):
        metrics = {
            f"top_{k}": macro_metrics(
                questions,
                {question_id: [item["document_id"] for item in output[question_id]["candidates"][:k]] for question_id in questions},
            )
            for k in range(1, 6)
        }
        write_json(args.output.with_suffix(".metrics.json"), metrics)
        print(json.dumps(metrics, indent=2))
    print(f"Wrote reranked scores for {len(output)} questions -> {args.output}")


def command_merge(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    load_tokenizer(str(args.adapter)).save_pretrained(args.output)
    print(f"Wrote merged FP16 model -> {args.output}")


def add_fusion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qwen-weight", type=float, default=QWEN_WEIGHT)
    parser.add_argument("--retrieval-weight", type=float, default=RETRIEVAL_WEIGHT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build 0/5 pointwise pairs from fused candidates")
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--candidates", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.set_defaults(run=command_prepare)

    train = commands.add_parser("train", help="Fine-tune a 4-bit QLoRA adapter")
    train.add_argument("--train-pairs", type=Path, required=True)
    train.add_argument("--validation-pairs", type=Path)
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument("--output", type=Path, default=ARTIFACTS / "qlora")
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--epochs", type=float, default=2.0)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--warmup-ratio", type=float, default=0.03)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--lora-rank", type=int, default=64)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.0)
    train.add_argument("--metric-top-k", type=int, default=5, choices=range(1, 6))
    add_fusion_arguments(train)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--resume-from-checkpoint")
    train.set_defaults(run=command_train)

    rerank = commands.add_parser("rerank", help="Zero-shot or adapter-based reranking")
    rerank.add_argument("--questions", type=Path, required=True)
    rerank.add_argument("--candidates", type=Path, required=True)
    rerank.add_argument("--model", default=DEFAULT_MODEL)
    rerank.add_argument("--adapter", type=Path)
    rerank.add_argument("--output", type=Path, required=True)
    rerank.add_argument("--submission", type=Path)
    rerank.add_argument("--top-k", type=int, default=5, choices=range(1, 6))
    rerank.add_argument("--max-length", type=int, default=1024)
    rerank.add_argument("--batch-size", type=int, default=4)
    add_fusion_arguments(rerank)
    rerank.set_defaults(run=command_rerank)

    merge = commands.add_parser("merge", help="Merge an adapter into an FP16 model")
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--model", default=DEFAULT_MODEL)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(run=command_merge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "max_length") and args.max_length < 256:
        raise SystemExit("--max-length must be >= 256")
    if hasattr(args, "qwen_weight") and (
        not math.isfinite(args.qwen_weight)
        or not math.isfinite(args.retrieval_weight)
        or args.qwen_weight < 0
        or args.retrieval_weight < 0
        or args.qwen_weight + args.retrieval_weight <= 0
    ):
        raise SystemExit("Fusion weights must be finite, non-negative, and not both zero")
    args.run(args)


if __name__ == "__main__":
    main()
