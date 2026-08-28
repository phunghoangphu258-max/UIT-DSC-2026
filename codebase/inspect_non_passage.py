"""Inspect selected contexts with no usable passage and their train references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_non_passage_contexts(contexts_dir: Path) -> tuple[int, list[dict]]:
    if not contexts_dir.is_dir():
        raise ValueError(f"context directory not found: {contexts_dir}")

    contexts = []
    files = sorted(contexts_dir.glob("*.json"))
    for path in files:
        context = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(context, dict):
            raise ValueError(f"{path} must contain a JSON object")
        passage = context.get("passage")
        if isinstance(passage, str) and passage.strip():
            continue

        status = (
            "missing"
            if "passage" not in context
            else "null"
            if passage is None
            else "blank"
            if isinstance(passage, str)
            else f"non-string ({type(passage).__name__})"
        )
        context_id = context.get("id")
        contexts.append(
            {
                "document_id": None if context_id is None else str(context_id),
                "file": path.name,
                "passage_status": status,
                "name": context.get("name"),
                "link": context.get("link"),
                "train_references": [],
            }
        )

    return len(files), contexts


def inspect(train_path: Path, contexts_dir: Path) -> dict:
    train = json.loads(train_path.read_text(encoding="utf-8-sig"))
    if not isinstance(train, dict):
        raise ValueError(f"{train_path} must contain a JSON object")

    file_count, contexts = find_non_passage_contexts(contexts_dir)

    by_id = {item["document_id"]: item for item in contexts}
    for question_id, item in train.items():
        for answer_id in item.get("answer") or []:
            context = by_id.get(str(answer_id))
            if context is not None:
                context["train_references"].append(
                    {
                        "question_id": str(question_id),
                        "question": item.get("question"),
                        "answer": [str(value) for value in item.get("answer") or []],
                    }
                )

    return {
        "summary": {
            "selected_context_files": file_count,
            "non_passage_contexts": len(contexts),
            "referenced_by_train": sum(
                bool(item["train_references"]) for item in contexts
            ),
            "train_references": sum(len(item["train_references"]) for item in contexts),
        },
        "contexts": contexts,
    }


def print_report(report: dict) -> None:
    summary = report["summary"]
    print(f"Scanned {summary['selected_context_files']} selected contexts.")
    print(
        f"Found {summary['non_passage_contexts']} non-passage contexts; "
        f"{summary['referenced_by_train']} are referenced by "
        f"{summary['train_references']} train rows."
    )
    for context in report["contexts"]:
        print(
            f"\n[{context['document_id'] or 'unknown ID'}] "
            f"{context['passage_status']} — {context['file']}"
        )
        if context["name"]:
            print(f"  name: {context['name']}")
        if context["link"]:
            print(f"  link: {context['link']}")
        if not context["train_references"]:
            print("  train: not referenced")
        for reference in context["train_references"]:
            print(
                f"  train {reference['question_id']} "
                f"(answer: {', '.join(reference['answer'])}): "
                f"{reference['question']}"
            )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "train.json")
    parser.add_argument(
        "--contexts",
        type=Path,
        default=PROJECT_ROOT / "selected-contexts" / "selected-contexts",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    try:
        report = inspect(args.train, args.contexts)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
