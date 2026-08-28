"""Minimal offline LegalIR pipeline: split, chunk, index, evaluate, and submit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
CONTEXTS = PROJECT_ROOT / "selected-contexts" / "selected-contexts"
TOKEN_PATTERN = r"(?u)\b[\wĐđ]+(?:[./-][\wĐđ]+)*\b"
ARTICLE_RE = re.compile(
    r"(?im)^[ \t]*Điều[ \t]+(?P<number>\d{1,3}[a-zA-Z]?)"
    r"(?:[ \t]*[.:][ \t]*|[ \t]*(?=\r?$))"
)
CLAUSE_RE = re.compile(r"(?m)^\s*(?P<number>\d{1,3})[.)]\s+(?=\S)")
POINT_RE = re.compile(r"(?im)^\s*(?P<number>[a-zđ])[.)]\s+(?=\S)")
CHAPTER_RE = re.compile(
    r"(?im)^[ \t]*Chương[ \t]+(?P<number>[IVXLCDM]+|\d{1,3})[.)]?[^\r\n]*$"
)
SECTION_RE = re.compile(
    r"(?im)^[ \t]*Mục[ \t]+(?P<number>[IVXLCDM]+|\d{1,3})[.)]?[^\r\n]*$"
)
ATTACHMENT_RE = re.compile(
    r"(?im)^(?P<title>[ \t]*(?:QUY[ \t]+CHẾ|QUY[ \t]+ĐỊNH|"
    r"VĂN[ \t]+BẢN[ \t]+BAN[ \t]+HÀNH[ \t]+KÈM[ \t]+THEO)[ \t]*)$"
    r"|^(?P<appendix>[ \t]*PHỤ[ \t]+LỤC[^\r\n]*)$"
)
RECIPIENTS_RE = re.compile(r"(?im)^[ \t]*Nơi[ \t]+nhận[ \t]*:")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalized_question(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).casefold()


def fallback_text(name: str | None, link: str | None) -> str:
    if name and name.strip():
        return normalize_text(name)
    path = unquote(urlparse(link or "").path).rsplit("/", 1)[-1]
    return normalize_text(re.sub(r"[-_]+", " ", path)) or "văn bản không có nội dung"


def word_windows(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text] if text else []
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    step = max_words - overlap_words
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, len(words) - overlap_words, step)
    ]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_document_parts(text: str) -> list[tuple[str, str]]:
    """Split attached regulations/appendices from the main instrument."""
    first_article = ARTICLE_RE.search(text)
    attachments = []
    for match in ATTACHMENT_RE.finditer(text):
        is_appendix = bool(match.group("appendix"))
        if is_appendix or (first_article and match.start() > first_article.start()):
            attachments.append(match)
    if not attachments:
        return [("", text)]

    parts: list[tuple[str, str]] = []
    before = text[: attachments[0].start()].strip()
    if before:
        parts.append(("", before))
    for index, match in enumerate(attachments):
        end = attachments[index + 1].start() if index + 1 < len(attachments) else len(text)
        parts.append((compact_text(match.group(0)), text[match.end() : end].strip()))
    return parts


def strip_recipient_tail(text: str) -> str:
    """Remove the administrative recipient/signature tail from one instrument part."""
    match = RECIPIENTS_RE.search(text)
    return text[: match.start()].strip() if match else text.strip()


def structural_units(text: str) -> list[dict]:
    """Return preambles and whole articles with their legal heading path."""
    units: list[dict] = []
    for original_part, original_body in split_document_parts(text):
        part = original_part
        body = strip_recipient_tail(original_body)
        events = []
        for kind, pattern in (
            ("chapter", CHAPTER_RE),
            ("section", SECTION_RE),
            ("article", ARTICLE_RE),
        ):
            events.extend(
                {
                    "kind": kind,
                    "match": match,
                    "start": match.start(),
                    "end": match.end(),
                }
                for match in pattern.finditer(body)
            )
        events.sort(key=lambda item: (item["start"], item["end"]))

        if not events:
            if body:
                units.append(
                    {
                        "document_part": part,
                        "chapter": "",
                        "section": "",
                        "article": "" if part else "Mở đầu",
                        "text": body,
                    }
                )
            continue

        first_boundary = events[0]["start"]
        introduction = body[:first_boundary].strip()
        if part and introduction and len(introduction.split()) <= 40:
            part = " | ".join((part, compact_text(introduction)))
            introduction = ""
        if introduction:
            units.append(
                {
                    "document_part": part,
                    "chapter": "",
                    "section": "",
                    "article": "Mở đầu",
                    "text": introduction,
                }
            )

        chapter = section = ""
        for index, event in enumerate(events):
            match = event["match"]
            next_start = events[index + 1]["start"] if index + 1 < len(events) else len(body)
            gap = body[event["end"] : next_start].strip()
            if event["kind"] == "chapter":
                chapter = compact_text(match.group(0))
                section = ""
                if gap and len(gap.split()) <= 30:
                    chapter = " ".join((chapter, compact_text(gap)))
                elif gap:
                    units.append(
                        {
                            "document_part": part,
                            "chapter": chapter,
                            "section": "",
                            "article": "Mở đầu chương",
                            "text": gap,
                        }
                    )
                continue
            if event["kind"] == "section":
                section = compact_text(match.group(0))
                if gap and len(gap.split()) <= 30:
                    section = " ".join((section, compact_text(gap)))
                elif gap:
                    units.append(
                        {
                            "document_part": part,
                            "chapter": chapter,
                            "section": section,
                            "article": "Mở đầu mục",
                            "text": gap,
                        }
                    )
                continue

            article_label = f"Điều {match.group('number')}."
            article_body = gap
            clauses = list(CLAUSE_RE.finditer(article_body))
            if clauses:
                possible_title = article_body[: clauses[0].start()].strip()
                if possible_title and len(possible_title.split()) <= 30:
                    article_label = " ".join((article_label, compact_text(possible_title)))
                    article_body = article_body[clauses[0].start() :].strip()
            units.append(
                {
                    "document_part": part,
                    "chapter": chapter,
                    "section": section,
                    "article": article_label,
                    "text": article_body or article_label,
                }
            )

    return units


def labeled_units(text: str, pattern: re.Pattern, prefix: str) -> list[tuple[str, str]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]
    units: list[tuple[str, str]] = []
    if text[: matches[0].start()].strip():
        units.append(("", text[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        units.append(
            (
                f"{prefix} {match.group('number')}",
                text[match.start() : end].strip(),
            )
        )
    return units


def label_range(labels: list[str]) -> str:
    labels = list(dict.fromkeys(label for label in labels if label))
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]}–{labels[-1].rsplit(' ', 1)[-1]}"


def bounded_text_parts(
    text: str, max_words: int, overlap_words: int, stage: int = 0
) -> list[str]:
    if len(text.split()) <= max_words:
        return [text] if text else []

    if stage == 0:
        segments = [value for _, value in labeled_units(text, POINT_RE, "Điểm")]
    elif stage == 1:
        segments = [value.strip() for value in re.split(r"\n{2,}", text) if value.strip()]
    else:
        return word_windows(text, max_words, overlap_words)
    if len(segments) <= 1:
        return bounded_text_parts(text, max_words, overlap_words, stage + 1)

    result: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    for segment in segments:
        count = len(segment.split())
        if count > max_words:
            if buffer:
                result.append("\n".join(buffer))
                buffer, buffer_words = [], 0
            result.extend(bounded_text_parts(segment, max_words, overlap_words, stage + 1))
        elif buffer and buffer_words + count > max_words:
            result.append("\n".join(buffer))
            buffer, buffer_words = [segment], count
        else:
            buffer.append(segment)
            buffer_words += count
    if buffer:
        result.append("\n".join(buffer))
    return result


def article_parts(
    text: str, max_words: int, overlap_words: int
) -> list[tuple[str, str]]:
    if len(text.split()) <= max_words:
        return [("", text)]

    clauses = labeled_units(text, CLAUSE_RE, "Khoản")
    if len(clauses) == 1 and not clauses[0][0]:
        return [("", value) for value in bounded_text_parts(text, max_words, overlap_words)]

    result: list[tuple[str, str]] = []
    labels: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    def flush() -> None:
        nonlocal labels, buffer, buffer_words
        if buffer:
            result.append((label_range(labels), "\n".join(buffer)))
        labels, buffer, buffer_words = [], [], 0

    for label, clause in clauses:
        count = len(clause.split())
        if count > max_words:
            flush()
            result.extend(
                (label, value)
                for value in bounded_text_parts(clause, max_words, overlap_words)
            )
        elif buffer and buffer_words + count > max_words:
            flush()
            labels, buffer, buffer_words = [label], [clause], count
        else:
            labels.append(label)
            buffer.append(clause)
            buffer_words += count
    flush()
    return result


def chunk_document(
    document: dict, max_words: int = 240, overlap_words: int = 40
) -> list[dict]:
    document_id = str(document["id"])
    name = normalize_text(str(document.get("name") or ""))
    link = str(document.get("link") or "")
    passage = normalize_text(str(document.get("passage") or ""))
    if not passage:
        passage = fallback_text(name, link)

    chunks: list[dict] = []
    for unit in structural_units(passage):
        for part, (clause, window) in enumerate(
            article_parts(unit["text"], max_words, overlap_words), 1
        ):
            heading = " | ".join(
                value
                for value in (
                    name,
                    unit["document_part"],
                    unit["chapter"],
                    unit["section"],
                    unit["article"],
                    clause,
                )
                if value
            )
            chunks.append(
                {
                    "chunk_id": f"{document_id}:{len(chunks):05d}",
                    "document_id": document_id,
                    "name": name,
                    "document_part": unit["document_part"],
                    "chapter": unit["chapter"],
                    "section": unit["section"],
                    "article": unit["article"],
                    "clause": clause,
                    "part": part,
                    "heading": heading,
                    "text": window,
                    "link": link,
                }
            )
    return chunks


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                item = json.loads(line)
                item["row_id"] = len(chunks)
                if not item.get("document_id"):
                    raise ValueError(f"Missing document_id at {path}:{line_number}")
                chunks.append(item)
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_prepare(args: argparse.Namespace) -> None:
    files = sorted(args.contexts.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON contexts in {args.contexts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document_count = chunk_count = fallback_count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for path in files:
            document = read_json(path)
            if not str(document.get("passage") or "").strip():
                fallback_count += 1
            for chunk in chunk_document(document, args.max_words, args.overlap_words):
                chunk["row_id"] = chunk_count
                output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1
            document_count += 1
    print(
        f"Prepared {chunk_count} chunks from {document_count} documents "
        f"({fallback_count} metadata fallbacks) -> {args.output}"
    )


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def group_duplicate_questions(data: dict) -> tuple[dict, dict[str, list[str]]]:
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for question_id, item in data.items():
        groups[normalized_question(str(item.get("question") or ""))].append(
            (str(question_id), item)
        )

    grouped, sources = {}, {}
    for rows in groups.values():
        rows.sort(key=lambda row: row[0])
        question_id = rows[0][0]
        answers = sorted({str(value) for _, item in rows for value in item.get("answer") or []})
        grouped[question_id] = {"question": rows[0][1].get("question"), "answer": answers}
        sources[question_id] = [row[0] for row in rows]
    return grouped, sources


def split_by_documents(data: dict, validation_ratio: float, seed: int) -> tuple[dict, dict, dict]:
    grouped, sources = group_duplicate_questions(data)
    union_find = UnionFind()
    for item in grouped.values():
        answers = [str(value) for value in item.get("answer") or []]
        for document_id in answers:
            union_find.find(document_id)
        for document_id in answers[1:]:
            union_find.union(answers[0], document_id)

    components: dict[str, set[str]] = defaultdict(set)
    for document_id in union_find.parent:
        components[union_find.find(document_id)].add(document_id)

    component_questions: dict[str, int] = defaultdict(int)
    for question_id, item in grouped.items():
        answers = [str(value) for value in item.get("answer") or []]
        if answers:
            component_questions[union_find.find(answers[0])] += len(sources[question_id])

    component_list = [
        (documents, component_questions[root]) for root, documents in components.items()
    ]
    random.Random(seed).shuffle(component_list)
    component_list.sort(key=lambda component: component[1], reverse=True)
    target = round(len(data) * validation_ratio)
    validation_documents: set[str] = set()
    validation_questions = 0
    for documents, question_count in component_list:
        before = abs(target - validation_questions)
        after = abs(target - validation_questions - question_count)
        if after < before:
            validation_documents.update(documents)
            validation_questions += question_count

    train, validation = {}, {}
    for question_id, item in data.items():
        answers = {str(value) for value in item.get("answer") or []}
        destination = validation if answers and answers <= validation_documents else train
        destination[str(question_id)] = item

    manifest = {
        "seed": seed,
        "validation_ratio": validation_ratio,
        "question_groups": len(grouped),
        "source_questions": len(data),
        "train_questions": len(train),
        "validation_questions": len(validation),
        "actual_validation_ratio": len(validation) / len(data) if data else 0.0,
        "train_documents": len({a for item in train.values() for a in item["answer"]}),
        "validation_documents": len(validation_documents),
        "duplicate_sources": {key: value for key, value in sources.items() if len(value) > 1},
    }
    return train, validation, manifest


def command_split(args: argparse.Namespace) -> None:
    train, validation, manifest = split_by_documents(
        read_json(args.input), args.validation_ratio, args.seed
    )
    write_json(args.output_dir / "train.json", train)
    write_json(args.output_dir / "validation.json", validation)
    write_json(args.output_dir / "split_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def update_manifest(directory: Path, section: str, value: dict) -> None:
    path = directory / "index_manifest.json"
    manifest = read_json(path) if path.exists() else {}
    manifest[section] = value
    write_json(path, manifest)


def build_bm25_index(texts: list[str], output: Path) -> None:
    try:
        import bm25s
        from bm25s.tokenization import Tokenizer
    except ImportError as error:
        raise SystemExit("Install dependencies: pip install -r codebase/requirements-legalir.txt") from error

    tokenizer = Tokenizer(lower=True, splitter=TOKEN_PATTERN, stopwords=[])
    token_ids = tokenizer.tokenize(texts, update_vocab=True, show_progress=True)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(tokenizer.to_tokenized_tuple(token_ids))
    retriever.save(output)
    tokenizer.save_vocab(output)


def command_build_bm25(args: argparse.Namespace) -> None:
    chunks = load_chunks(args.chunks)
    build_bm25_index([chunk["heading"] or chunk["text"] for chunk in chunks], args.output / "bm25_heading")
    build_bm25_index([chunk["text"] for chunk in chunks], args.output / "bm25_body")
    update_manifest(
        args.output,
        "bm25",
        {
            "chunks_sha256": sha256_file(args.chunks),
            "chunk_count": len(chunks),
            "method": "lucene",
            "fields": ["heading", "body"],
            "token_pattern": TOKEN_PATTERN,
        },
    )
    print(f"Built heading/body BM25 indexes -> {args.output}")


def load_encoder(model_name: str, revision: str | None, device: str, max_length: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Install dependencies: pip install -r codebase/requirements-legalir.txt") from error
    kwargs = {} if device == "auto" else {"device": device}
    if revision:
        kwargs["revision"] = revision
    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = max_length

    if str(model.device).startswith("cuda"):
        model.half()

    return model


def checkpointed_encode(
    model,
    texts: list[str],
    batch_size: int,
    checkpoint_size: int,
    data_path: Path,
    state_path: Path,
    metadata: dict,
):
    import numpy as np

    dimension = int(model.get_sentence_embedding_dimension())
    expected = {**metadata, "row_count": len(texts), "dimension": dimension}
    completed = 0
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.exists() != data_path.exists():
        raise ValueError(
            f"Incomplete dense checkpoint; remove both {state_path} and {data_path}"
        )
    if state_path.exists():
        state = read_json(state_path)
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Dense checkpoint does not match this build: {state_path}")
        completed = int(state.get("completed", 0))
        if not 0 <= completed <= len(texts):
            raise ValueError(f"Invalid dense checkpoint progress: {completed}")
        expected_bytes = len(texts) * dimension * np.dtype("float32").itemsize
        if data_path.stat().st_size != expected_bytes:
            raise ValueError(f"Invalid dense checkpoint size: {data_path}")
        print(f"Resuming dense encoding at row {completed:,}/{len(texts):,}")
        mode = "r+"
    else:
        mode = "w+"

    embeddings = np.memmap(
        data_path, dtype="float32", mode=mode, shape=(len(texts), dimension)
    )
    if mode == "w+":
        embeddings.flush()
        temporary_state = state_path.with_suffix(state_path.suffix + ".tmp")
        write_json(temporary_state, {**expected, "completed": 0})
        temporary_state.replace(state_path)
    for start in range(completed, len(texts), checkpoint_size):
        end = min(start + checkpoint_size, len(texts))
        embeddings[start:end] = model.encode(
            texts[start:end],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.flush()
        temporary_state = state_path.with_suffix(state_path.suffix + ".tmp")
        write_json(temporary_state, {**expected, "completed": end})
        temporary_state.replace(state_path)
        print(f"Dense checkpoint: {end:,}/{len(texts):,} rows")
    return embeddings


def command_build_dense(args: argparse.Namespace) -> None:
    try:
        import faiss
        import numpy as np
    except ImportError as error:
        raise SystemExit("Install dependencies: pip install -r codebase/requirements-legalir.txt") from error

    chunks = load_chunks(args.chunks)
    model = load_encoder(args.model, args.revision, args.device, args.max_length)
    texts = ["\n".join(filter(None, (chunk["heading"], chunk["text"]))) for chunk in chunks]
    chunks_checksum = sha256_file(args.chunks)
    checkpoint_data = args.output / "bge_dense.embeddings.partial"
    checkpoint_state = args.output / "bge_dense.checkpoint.json"
    embeddings = checkpointed_encode(
        model,
        texts,
        args.batch_size,
        args.checkpoint_size,
        checkpoint_data,
        checkpoint_state,
        {
            "chunks_sha256": chunks_checksum,
            "model": args.model,
            "revision": args.revision,
            "max_length": args.max_length,
            "dtype": str(next(model.parameters()).dtype),
            "normalized": True,
        },
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    args.output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output / "bge_dense.faiss"))
    update_manifest(
        args.output,
        "dense",
        {
            "chunks_sha256": chunks_checksum,
            "chunk_count": len(chunks),
            "model": args.model,
            "revision": args.revision,
            "dimension": int(embeddings.shape[1]),
            "max_length": args.max_length,
            "normalized": True,
            "index": "IndexFlatIP",
        },
    )
    del embeddings
    checkpoint_data.unlink(missing_ok=True)
    checkpoint_state.unlink(missing_ok=True)
    print(f"Built {index.ntotal} dense vectors -> {args.output / 'bge_dense.faiss'}")


def aggregate_chunk_hits(
    indices: list[int], scores: list[float], chunks: list[dict], top_n: int
) -> list[tuple[str, float]]:
    by_document: dict[str, list[float]] = defaultdict(list)
    for index, score in zip(indices, scores):
        if 0 <= index < len(chunks) and math.isfinite(float(score)):
            by_document[chunks[index]["document_id"]].append(float(score))
    aggregated = []
    for document_id, values in by_document.items():
        best = sorted(values, reverse=True)[:top_n]
        aggregated.append((document_id, sum(best) / len(best)))
    return sorted(aggregated, key=lambda item: (-item[1], item[0]))


def weighted_rrf(
    rankings: list[tuple[float, list[tuple[str, float]]]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for weight, ranking in rankings:
        for rank, (document_id, _) in enumerate(ranking, 1):
            scores[document_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def select_document_chunks(
    chunks: list[dict],
    heading_scores,
    body_scores,
    dense_scores,
    heading_weight: float = 0.5,
    body_weight: float = 2.0,
    dense_weight: float = 4.0,
    rrf_k: int = 4,
    threshold: float = 0.4,
    limit: int = 2,
) -> list[dict]:
    def ranking(scores) -> list[tuple[str, float]]:
        return [
            (chunk["chunk_id"], float(scores[chunk["row_id"]]))
            for chunk in sorted(
                chunks,
                key=lambda item: (
                    -float(scores[item["row_id"]]),
                    item["chunk_id"],
                ),
            )
        ]

    fused = weighted_rrf(
        [
            (heading_weight, ranking(heading_scores)),
            (body_weight, ranking(body_scores)),
            (dense_weight, ranking(dense_scores)),
        ],
        rrf_k,
    )
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    selected = [item for item in fused if item[1] >= threshold][:limit]
    if not selected and fused:
        selected = fused[:1]
    return [
        {
            "chunk_id": chunk_id,
            "text": "\n".join(
                str(by_id[chunk_id].get(key) or "").strip()
                for key in ("heading", "text")
                if str(by_id[chunk_id].get(key) or "").strip()
            ),
            "score": score,
        }
        for chunk_id, score in selected
    ]


def load_bm25(directory: Path):
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        import bm25s
        from bm25s.tokenization import Tokenizer
    except ImportError as error:
        raise SystemExit("Install dependencies: pip install -r codebase/requirements-legalir.txt") from error
    tokenizer = Tokenizer(lower=True, splitter=TOKEN_PATTERN, stopwords=[])
    tokenizer.load_vocab(directory)
    return bm25s.BM25.load(directory, mmap=True), tokenizer


def retrieve_questions(
    args: argparse.Namespace, questions: dict, include_chunks: bool = False
) -> dict:
    try:
        import faiss
        import numpy as np
    except ImportError as error:
        raise SystemExit("Install dependencies: pip install -r codebase/requirements-legalir.txt") from error

    chunks = load_chunks(args.chunks)
    manifest = read_json(args.index_dir / "index_manifest.json")
    checksum = sha256_file(args.chunks)
    for section in ("bm25", "dense"):
        if manifest.get(section, {}).get("chunks_sha256") != checksum:
            raise ValueError(f"{section} index was built from different chunks")

    heading_index, heading_tokenizer = load_bm25(args.index_dir / "bm25_heading")
    body_index, body_tokenizer = load_bm25(args.index_dir / "bm25_body")
    dense_index = faiss.read_index(str(args.index_dir / "bge_dense.faiss"))
    dense_config = manifest["dense"]
    model = load_encoder(
        args.model or dense_config["model"],
        args.revision if args.revision is not None else dense_config.get("revision"),
        args.device,
        int(dense_config["max_length"]),
    )

    question_ids = [str(value) for value in questions]
    texts = [str(questions[question_id].get("question") or "") for question_id in question_ids]
    heading_tokens = heading_tokenizer.tokenize(texts, update_vocab=False, show_progress=False)
    body_tokens = body_tokenizer.tokenize(texts, update_vocab=False, show_progress=False)
    top_chunks = min(args.top_chunks, len(chunks))
    heading_ids, heading_scores = heading_index.retrieve(
        heading_tokens, k=top_chunks, backend_selection="numpy"
    )
    body_ids, body_scores = body_index.retrieve(
        body_tokens, k=top_chunks, backend_selection="numpy"
    )
    query_vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dense_scores, dense_ids = dense_index.search(
        np.ascontiguousarray(query_vectors, dtype="float32"), top_chunks
    )

    chunks_by_document: dict[str, list[dict]] = defaultdict(list)
    if include_chunks:
        for chunk in chunks:
            chunks_by_document[chunk["document_id"]].append(chunk)

    predictions = {}
    for row, question_id in enumerate(question_ids):
        heading = aggregate_chunk_hits(
            [int(value) for value in heading_ids[row]],
            [float(value) for value in heading_scores[row]],
            chunks,
            args.top_n,
        )
        body = aggregate_chunk_hits(
            [int(value) for value in body_ids[row]],
            [float(value) for value in body_scores[row]],
            chunks,
            args.top_n,
        )
        dense = aggregate_chunk_hits(
            [int(value) for value in dense_ids[row]],
            [float(value) for value in dense_scores[row]],
            chunks,
            args.top_n,
        )
        ranking = weighted_rrf(
            [
                (args.heading_weight, heading),
                (args.body_weight, body),
                (args.dense_weight, dense),
            ],
            args.rrf_k,
        )
        top_documents = ranking[: args.top_documents]
        if not include_chunks:
            predictions[question_id] = [document_id for document_id, _ in top_documents]
            continue

        retrieved_ids = {document_id for document_id, _ in top_documents}
        document_scores = dict(top_documents)
        document_ids = list(retrieved_ids)
        document_ids.sort(key=lambda value: (-document_scores[value], value))
        for document_id in map(str, questions[question_id].get("answer") or []):
            if document_id not in retrieved_ids and document_id not in document_ids:
                document_ids.append(document_id)

        heading_all_scores = heading_index.get_scores_from_ids(heading_tokens[row])
        body_all_scores = body_index.get_scores_from_ids(body_tokens[row])
        candidate_rows = [
            chunk["row_id"]
            for document_id in document_ids
            for chunk in chunks_by_document.get(document_id, [])
        ]
        dense_by_row = {}
        if candidate_rows:
            row_ids = np.asarray(candidate_rows, dtype="int64")
            dense_vectors = dense_index.reconstruct_batch(row_ids)
            candidate_dense_scores = dense_vectors @ np.asarray(
                query_vectors[row], dtype="float32"
            )
            dense_by_row = dict(zip(row_ids.tolist(), candidate_dense_scores.tolist()))

        candidates = []
        for document_id in document_ids:
            document_chunks = chunks_by_document.get(document_id, [])
            if not document_chunks:
                raise ValueError(f"No chunks found for candidate document {document_id}")
            candidates.append(
                {
                    "document_id": document_id,
                    "document_score": document_scores.get(document_id),
                    "retrieved": document_id in retrieved_ids,
                    "chunks": select_document_chunks(
                        document_chunks,
                        heading_all_scores,
                        body_all_scores,
                        dense_by_row,
                        args.chunk_heading_weight,
                        args.chunk_body_weight,
                        args.chunk_dense_weight,
                        args.chunk_rrf_k,
                        args.chunk_threshold,
                        args.chunks_per_document,
                    ),
                }
            )
        predictions[question_id] = {"candidates": candidates}
    return predictions


def macro_metrics(gold: dict, predictions: dict[str, list[str]]) -> dict[str, float]:
    recalls, precisions = [], []
    for question_id, item in gold.items():
        relevant = {str(value) for value in item.get("answer") or []}
        predicted_list = [str(value) for value in predictions.get(str(question_id), [])]
        if len(predicted_list) > 5:
            recalls.append(0.0)
            precisions.append(0.0)
            continue
        predicted = set(predicted_list)
        hits = len(relevant & predicted)
        recalls.append(hits / len(relevant) if relevant else 0.0)
        precisions.append(hits / len(predicted) if predicted else 0.0)
    return {
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "questions": len(recalls),
    }


def command_evaluate(args: argparse.Namespace) -> None:
    gold = read_json(args.questions)
    predictions = retrieve_questions(args, gold)
    if args.predictions:
        write_json(args.predictions, predictions)
    print(json.dumps(macro_metrics(gold, predictions), indent=2))


def command_candidates(args: argparse.Namespace) -> None:
    questions = read_json(args.questions)
    predictions = retrieve_questions(args, questions, include_chunks=True)
    write_json(args.output, predictions)
    print(f"Wrote {len(predictions)} document/chunk candidate sets -> {args.output}")


def command_predict(args: argparse.Namespace) -> None:
    questions = read_json(args.questions)
    predictions = retrieve_questions(args, questions)
    submission = {
        question_id: {"answer": answers} for question_id, answers in predictions.items()
    }
    write_json(args.output, submission)
    zip_path = args.output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.output, "submission.json")
    print(f"Wrote {len(submission)} predictions -> {args.output} and {zip_path}")


def add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunks", type=Path, default=ARTIFACTS / "chunks.jsonl")
    parser.add_argument("--index-dir", type=Path, default=ARTIFACTS / "index")
    parser.add_argument("--model", help="Override model path/ID stored in manifest")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-chunks", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--top-documents", type=int, default=5)
    parser.add_argument("--heading-weight", type=float, default=1.5)
    parser.add_argument("--body-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)


def add_chunk_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-heading-weight", type=float, default=0.5)
    parser.add_argument("--chunk-body-weight", type=float, default=2.0)
    parser.add_argument("--chunk-dense-weight", type=float, default=4.0)
    parser.add_argument("--chunk-rrf-k", type=int, default=4)
    parser.add_argument("--chunk-threshold", type=float, default=0.4)
    parser.add_argument("--chunks-per-document", type=int, default=2, choices=(1, 2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Normalize and structurally chunk corpus")
    prepare.add_argument("--contexts", type=Path, default=CONTEXTS)
    prepare.add_argument("--output", type=Path, default=ARTIFACTS / "chunks.jsonl")
    prepare.add_argument("--max-words", type=int, default=240)
    prepare.add_argument("--overlap-words", type=int, default=40)
    prepare.set_defaults(run=command_prepare)

    split = commands.add_parser("split", help="Create a document-disjoint 70/30 split")
    split.add_argument("--input", type=Path, default=PROJECT_ROOT / "train.json")
    split.add_argument("--output-dir", type=Path, default=ARTIFACTS / "split")
    split.add_argument("--validation-ratio", type=float, default=0.3)
    split.add_argument("--seed", type=int, default=2026)
    split.set_defaults(run=command_split)

    bm25 = commands.add_parser("build-bm25", help="Build heading/body BM25 indexes")
    bm25.add_argument("--chunks", type=Path, default=ARTIFACTS / "chunks.jsonl")
    bm25.add_argument("--output", type=Path, default=ARTIFACTS / "index")
    bm25.set_defaults(run=command_build_bm25)

    dense = commands.add_parser("build-dense", help="Build exact Vietnamese embedding FAISS index")
    dense.add_argument("--chunks", type=Path, default=ARTIFACTS / "chunks.jsonl")
    dense.add_argument("--output", type=Path, default=ARTIFACTS / "index")
    dense.add_argument("--model", default="AITeamVN/Vietnamese_Embedding_v2")
    dense.add_argument("--revision")
    dense.add_argument("--device", default="auto")
    dense.add_argument("--batch-size", type=int, default=16)
    dense.add_argument("--checkpoint-size", type=int, default=2048)
    dense.add_argument("--max-length", type=int, default=512)
    dense.set_defaults(run=command_build_dense)

    evaluate = commands.add_parser("evaluate", help="Retrieve and compute official macro metrics")
    evaluate.add_argument("--questions", type=Path, default=ARTIFACTS / "split" / "validation.json")
    evaluate.add_argument("--predictions", type=Path)
    add_retrieval_arguments(evaluate)
    evaluate.set_defaults(run=command_evaluate)

    candidates = commands.add_parser(
        "candidates", help="Export a larger fused candidate pool for reranking"
    )
    candidates.add_argument(
        "--questions", type=Path, default=ARTIFACTS / "split" / "train.json"
    )
    candidates.add_argument(
        "--output", type=Path, default=ARTIFACTS / "train_candidates.json"
    )
    add_retrieval_arguments(candidates)
    add_chunk_candidate_arguments(candidates)
    candidates.set_defaults(top_documents=20, run=command_candidates)

    predict = commands.add_parser("predict", help="Create submission.json and submission.zip")
    predict.add_argument("--questions", type=Path, default=PROJECT_ROOT / "public-official.json")
    predict.add_argument("--output", type=Path, default=ARTIFACTS / "submission.json")
    add_retrieval_arguments(predict)
    predict.set_defaults(run=command_predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "top_n") and args.top_n < 1:
        raise SystemExit("--top-n must be >= 1")
    if hasattr(args, "rrf_k") and args.rrf_k < 1:
        raise SystemExit("--rrf-k must be >= 1")
    if hasattr(args, "chunk_rrf_k") and args.chunk_rrf_k < 1:
        raise SystemExit("--chunk-rrf-k must be >= 1")
    if hasattr(args, "chunk_threshold") and args.chunk_threshold < 0:
        raise SystemExit("--chunk-threshold must be >= 0")
    for name in ("chunk_heading_weight", "chunk_body_weight", "chunk_dense_weight"):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")
    if hasattr(args, "top_documents") and args.top_documents < 1:
        raise SystemExit("--top-documents must be >= 1")
    if args.command == "predict" and args.top_documents > 5:
        raise SystemExit("predict allows at most 5 documents; use candidates for a larger pool")
    if hasattr(args, "validation_ratio") and not 0 < args.validation_ratio < 1:
        raise SystemExit("--validation-ratio must be between 0 and 1")
    if hasattr(args, "checkpoint_size") and args.checkpoint_size < 1:
        raise SystemExit("--checkpoint-size must be >= 1")
    if hasattr(args, "max_words") and args.max_words < 1:
        raise SystemExit("--max-words must be >= 1")
    if hasattr(args, "overlap_words") and not 0 <= args.overlap_words < args.max_words:
        raise SystemExit("--overlap-words must be between 0 and --max-words - 1")
    args.run(args)


if __name__ == "__main__":
    main()
