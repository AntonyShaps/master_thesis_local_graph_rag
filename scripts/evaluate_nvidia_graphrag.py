from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_QUESTIONS_FILE = Path("data/eval/nvidia_gold_questions.jsonl")
REQUIRED_QUESTION_FIELDS = ("id", "type", "question", "gold_answer")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TOKEN_RE = re.compile(r"[a-z0-9]+")
NO_ANSWER_RE = re.compile(
    r"\b("
    r"not\s+(?:(?:explicitly|directly)\s+)?"
    r"(?:reported|available|provided|found|included|specified)"
    r"|does\s+not\s+explicitly\s+state"
    r"|does\s+not\s+state"
    r"|cannot\s+find"
    r"|exact\s+(?:numerical\s+)?value\s+is\s+not\s+specified"
    r"|available\s+information\s+does\s+not\s+include"
    r"|information\s+provided\s+does\s+not\s+include"
    r")\b",
    flags=re.IGNORECASE,
)
VALID_LABELS = {"correct", "partial", "incorrect", "no_answer"}

RULE_CHECKS: dict[str, list[tuple[str, list[str]]]] = {
    "NVID-F01": [
        (
            "Nominating and Corporate Governance Committee",
            [r"nominating\s+and\s+corporate\s+governance\s+committee"],
        ),
    ],
    "NVID-F02": [
        ("GRI", [r"\bgri\b", r"global\s+reporting\s+initiative"]),
        ("SASB", [r"\bsasb\b"]),
        ("TCFD", [r"\btcfd\b"]),
        (
            "U.N. Sustainable Development Goals",
            [
                r"u\.?\s*n\.?\s+sustainable\s+development\s+goals",
                r"\bsdgs?\b",
                r"united\s+nations\s+sustainable\s+development\s+goals",
            ],
        ),
    ],
    "NVID-F03": [
        ("simulation or forecasting", [r"simulat", r"forecast", r"predict"]),
        ("weather or climate simulation", [r"weather", r"climate"]),
        (
            "extreme weather, resilience, or mitigation",
            [r"extreme\s+weather", r"resilien", r"mitigation", r"disaster"],
        ),
    ],
    "NVID-F04": [
        ("conflict-free", [r"conflict[-\s]?free"]),
        ("3TG minerals", [r"\b3tg\b", r"gold.*tantalum.*tungsten.*tin"]),
        ("100% RMAP compliant", [r"100\s*%.*rmap", r"rmap.*100\s*%"]),
    ],
    "NVID-F05": [
        ("privacy and data protection", [r"privacy", r"data\s+protection"]),
        ("safe/as intended", [r"safe", r"as\s+intended"]),
        ("transparent design/limitations", [r"transparen", r"limitations?"]),
        ("minimize bias", [r"bias", r"\bfair"]),
    ],
    "NVID-N01": [
        ("100%", [r"\b100\s*%"]),
        ("renewable electricity", [r"renewable\s+electricity"]),
    ],
    "NVID-N02": [
        ("12,952", [r"\b12[\s,]*952\b"]),
        ("Scope 1", [r"scope\s*1"]),
        ("MT CO2e", [r"mt\s+co2e", r"metric\s+tons?.*co2e"]),
    ],
    "NVID-N03": [
        ("6,912,577", [r"\b6[\s,]*912[\s,]*577\b"]),
        ("Scope 3", [r"scope\s*3"]),
        ("MT CO2e", [r"mt\s+co2e", r"metric\s+tons?.*co2e"]),
    ],
    "NVID-N04": [
        (
            "over 80%",
            [
                r"over\s+80\s*%",
                r"more\s+than\s+80\s*%",
                r">\s*80\s*%",
                r"80\s*%\+",
                r"80\+\s*%",
            ],
        ),
        ("supplier engagement", [r"supplier\s+engagement"]),
        ("Scope 3 category 1", [r"scope\s*3.*category\s*1", r"category\s*1"]),
    ],
    "NVID-N05": [
        ("84%", [r"\b84\s*%"]),
        ("landfill diversion", [r"landfill\s+diversion"]),
    ],
}


@dataclass(frozen=True)
class QueryResult:
    answer: str
    raw_stdout: str
    raw_stderr: str
    returncode: int
    elapsed_seconds: float
    command: list[str]
    error: str | None = None


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("co₂e", "co2e"))


def _token_f1(prediction: str, gold: str) -> dict[str, float]:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = sum(
        min(pred_counts.get(token, 0), gold_counts.get(token, 0))
        for token in gold_counts
    )
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _normalized_contains(prediction: str, gold: str) -> bool:
    pred = " ".join(_tokens(prediction))
    target = " ".join(_tokens(gold))
    return bool(target and target in pred)


def _rule_grade(item: dict[str, str], answer: str) -> dict[str, Any]:
    checks = RULE_CHECKS.get(item["id"])
    if not answer.strip():
        return {
            "label": "no_answer",
            "score": 0.0,
            "matched": [],
            "missing": [name for name, _patterns in checks] if checks else [],
            "rationale": "The query returned no answer.",
        }

    normalized_answer = answer.lower().replace("co₂e", "co2e")
    if NO_ANSWER_RE.search(normalized_answer):
        return {
            "label": "no_answer",
            "score": 0.0,
            "matched": [],
            "missing": [name for name, _patterns in checks] if checks else [],
            "rationale": "The answer says the requested fact was not found/reported.",
        }

    if not checks:
        token_f1 = _token_f1(answer, item["gold_answer"])["f1"]
        substring_match = _normalized_contains(answer, item["gold_answer"])
        if substring_match or token_f1 >= 0.8:
            label = "correct"
            score = 1.0
        elif token_f1 >= 0.35:
            label = "partial"
            score = token_f1
        else:
            label = "incorrect"
            score = 0.0

        return {
            "label": label,
            "score": score,
            "matched": [],
            "missing": [],
            "rationale": (
                "No deterministic rule exists for this question ID; "
                f"used token F1={token_f1}."
            ),
        }

    matched: list[str] = []
    missing: list[str] = []

    for name, patterns in checks:
        matched_check = any(
            re.search(pattern, normalized_answer, flags=re.IGNORECASE)
            for pattern in patterns
        )
        if matched_check:
            matched.append(name)
        else:
            missing.append(name)

    if not missing:
        label = "correct"
        score = 1.0
    elif matched:
        label = "partial"
        score = round(len(matched) / len(checks), 4)
    else:
        label = "incorrect"
        score = 0.0

    if missing:
        rationale = (
            f"Matched {len(matched)}/{len(checks)} required facts; "
            f"missing: {', '.join(missing)}."
        )
    else:
        rationale = "Matched all required facts."

    return {
        "label": label,
        "score": score,
        "matched": matched,
        "missing": missing,
        "rationale": rationale,
    }


def _build_query_command(args: argparse.Namespace, question: str) -> list[str]:
    root = Path(args.root)

    if args.query_runner == "bin" or args.query_runner == "auto":
        graphrag_bin = root / ".venv" / "bin" / "graphrag"
        if graphrag_bin.exists() and os.access(graphrag_bin, os.X_OK):
            return [
                str(graphrag_bin),
                "query",
                "--root",
                str(root),
                "--method",
                args.method,
                "--response-type",
                args.response_type,
                question,
            ]
        if args.query_runner == "bin":
            raise FileNotFoundError(f"GraphRAG binary not found: {graphrag_bin}")

    return [
        args.uv_bin,
        "run",
        "--project",
        str(root),
        "graphrag",
        "query",
        "--root",
        str(root),
        "--method",
        args.method,
        "--response-type",
        args.response_type,
        question,
    ]


def _run_query(args: argparse.Namespace, question: str) -> QueryResult:
    command = _build_query_command(args, question)
    env = os.environ.copy()
    env.setdefault("GRAPHRAG_API_KEY", args.api_key)
    env.setdefault("NO_COLOR", "1")
    if command[:2] == [args.uv_bin, "run"]:
        env.pop("VIRTUAL_ENV", None)

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=args.cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.query_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        return QueryResult(
            answer="",
            raw_stdout=_strip_ansi(_ensure_text(exc.stdout)).strip(),
            raw_stderr=_strip_ansi(_ensure_text(exc.stderr)).strip(),
            returncode=124,
            elapsed_seconds=round(elapsed, 3),
            command=command,
            error=f"query timed out after {args.query_timeout} seconds",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return QueryResult(
            answer="",
            raw_stdout="",
            raw_stderr="",
            returncode=1,
            elapsed_seconds=round(elapsed, 3),
            command=command,
            error=str(exc),
        )

    stdout = _strip_ansi(completed.stdout).strip()
    stderr = _strip_ansi(completed.stderr).strip()
    elapsed = time.perf_counter() - started

    return QueryResult(
        answer=stdout if completed.returncode == 0 else "",
        raw_stdout=stdout,
        raw_stderr=stderr,
        returncode=completed.returncode,
        elapsed_seconds=round(elapsed, 3),
        command=command,
        error=None if completed.returncode == 0 else stderr or stdout,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()

    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"No JSON object found in judge response: {text[:500]}")


def _call_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 300,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))

    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"No choices returned by judge endpoint: {body}")

    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    text = first.get("text")
    if isinstance(text, str):
        return text.strip()

    raise ValueError(f"Could not read judge response content: {body}")


def _judge_answer(
    args: argparse.Namespace,
    item: dict[str, str],
    answer: str,
) -> dict[str, Any]:
    if not args.judge:
        return {
            "enabled": False,
            "label": "not_judged",
            "score": None,
            "hallucination": None,
            "rationale": "",
        }

    if not answer.strip():
        return {
            "enabled": True,
            "label": "no_answer",
            "score": 0.0,
            "hallucination": False,
            "rationale": "The query returned no answer.",
        }

    system = (
        "You are a strict ESG QA evaluator. Grade whether the candidate answer "
        "correctly answers the question according to the gold answer. Ignore "
        "citation formatting. Penalize missing required facts, wrong numbers, "
        "wrong units, and unsupported extra claims. Return only JSON."
    )
    user = (
        "Use this schema exactly:\n"
        "{"
        '"label":"correct|partial|incorrect|no_answer",'
        '"score":1.0,'
        '"hallucination":false,'
        '"rationale":"short reason"'
        "}\n\n"
        "Scoring rules: correct=1.0 when all essential facts match; "
        "partial=0.5 when some essential facts are present but important detail "
        "is missing; incorrect/no_answer=0.0.\n\n"
        f"Question ID: {item['id']}\n"
        f"Question type: {item['type']}\n"
        f"Question: {item['question']}\n"
        f"Gold answer: {item['gold_answer']}\n"
        f"Candidate answer: {answer}"
    )

    try:
        response_text = _call_chat_completion(
            api_base=args.judge_api_base,
            api_key=args.api_key,
            model=args.judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=args.judge_timeout,
        )
        parsed = _extract_json_object(response_text)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        return {
            "enabled": True,
            "label": "judge_error",
            "score": None,
            "hallucination": None,
            "rationale": str(exc),
        }

    label = str(parsed.get("label", "judge_error")).strip().lower()
    if label not in VALID_LABELS:
        label = "judge_error"

    try:
        score = float(parsed.get("score"))
    except (TypeError, ValueError):
        score = None

    if label == "correct":
        score = 1.0
    elif label == "partial":
        score = 0.5
    elif label in {"incorrect", "no_answer"}:
        score = 0.0

    return {
        "enabled": True,
        "label": label,
        "score": score,
        "hallucination": bool(parsed.get("hallucination", False)),
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def _select_grade(rule: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    judge_label = str(judge.get("label", ""))
    if judge.get("enabled") and judge_label in VALID_LABELS:
        return {
            "source": "judge",
            "label": judge_label,
            "score": judge.get("score"),
            "rationale": judge.get("rationale", ""),
        }

    return {
        "source": "rules",
        "label": rule["label"],
        "score": rule["score"],
        "rationale": rule["rationale"],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successful_queries = sum(1 for row in rows if row["query"]["returncode"] == 0)
    graded_rows = [
        row for row in rows if isinstance(row["grade"].get("score"), int | float)
    ]
    label_counts: dict[str, int] = {}
    type_counts: dict[str, dict[str, Any]] = {}

    for row in rows:
        label = str(row["grade"].get("label", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1

        q_type = row["type"]
        bucket = type_counts.setdefault(
            q_type,
            {
                "total": 0,
                "correct": 0,
                "partial": 0,
                "incorrect": 0,
                "no_answer": 0,
                "scores": [],
            },
        )
        bucket["total"] += 1
        if label in VALID_LABELS:
            bucket[label] += 1
        score = row["grade"].get("score")
        if isinstance(score, int | float):
            bucket["scores"].append(float(score))

    score_values = [float(row["grade"]["score"]) for row in graded_rows]
    hallucination_values = [
        bool(row["judge"]["hallucination"])
        for row in rows
        if isinstance(row["judge"].get("hallucination"), bool)
    ]

    by_type: dict[str, dict[str, Any]] = {}
    for q_type, bucket in type_counts.items():
        scores = bucket.pop("scores")
        by_type[q_type] = {
            **bucket,
            "average_score": round(sum(scores) / len(scores), 4) if scores else None,
        }

    return {
        "total_questions": total,
        "successful_queries": successful_queries,
        "query_failures": total - successful_queries,
        "label_counts": label_counts,
        "average_score": (
            round(sum(score_values) / len(score_values), 4) if score_values else None
        ),
        "correct_rate": (
            round(label_counts.get("correct", 0) / total, 4) if total else None
        ),
        "partial_or_better_rate": (
            round(
                (label_counts.get("correct", 0) + label_counts.get("partial", 0))
                / total,
                4,
            )
            if total
            else None
        ),
        "hallucination_rate": (
            round(sum(hallucination_values) / len(hallucination_values), 4)
            if hallucination_values
            else None
        ),
        "by_type": by_type,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "type",
        "question",
        "gold_answer",
        "answer",
        "returncode",
        "elapsed_seconds",
        "grade_source",
        "grade_label",
        "grade_score",
        "grade_rationale",
        "rule_label",
        "rule_score",
        "rule_missing",
        "judge_label",
        "judge_score",
        "judge_hallucination",
        "judge_rationale",
        "token_f1",
        "gold_substring_match",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "answer": row["answer"],
                    "returncode": row["query"]["returncode"],
                    "elapsed_seconds": row["query"]["elapsed_seconds"],
                    "grade_source": row["grade"].get("source"),
                    "grade_label": row["grade"].get("label"),
                    "grade_score": row["grade"].get("score"),
                    "grade_rationale": row["grade"].get("rationale"),
                    "rule_label": row["rule_grade"].get("label"),
                    "rule_score": row["rule_grade"].get("score"),
                    "rule_missing": "; ".join(row["rule_grade"].get("missing", [])),
                    "judge_label": row["judge"].get("label"),
                    "judge_score": row["judge"].get("score"),
                    "judge_hallucination": row["judge"].get("hallucination"),
                    "judge_rationale": row["judge"].get("rationale"),
                    "token_f1": row["metrics"]["token_f1"]["f1"],
                    "gold_substring_match": row["metrics"]["gold_substring_match"],
                }
            )


def _normalize_question_row(
    row: Any,
    *,
    source: Path,
    row_number: int,
) -> dict[str, str]:
    if not isinstance(row, dict):
        raise ValueError(f"{source}:{row_number} is not a JSON object.")

    normalized: dict[str, str] = {}
    missing: list[str] = []
    for field in REQUIRED_QUESTION_FIELDS:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            missing.append(field)
        else:
            normalized[field] = str(value).strip()

    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"{source}:{row_number} missing required field(s): {fields}")

    return normalized


def _load_questions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    questions: list[dict[str, str]] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array of question objects.")
        for index, row in enumerate(payload, start=1):
            questions.append(
                _normalize_question_row(row, source=path, row_number=index)
            )
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                questions.append(
                    _normalize_question_row(
                        row,
                        source=path,
                        row_number=line_number,
                    )
                )

    if not questions:
        raise ValueError(f"Question file is empty: {path}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for item in questions:
        question_id = item["id"].upper()
        if question_id in seen:
            duplicates.append(item["id"])
        seen.add(question_id)
    if duplicates:
        raise ValueError(f"Duplicate question IDs in {path}: {', '.join(duplicates)}")

    return questions


def _select_questions(args: argparse.Namespace) -> list[dict[str, str]]:
    selected = _load_questions(Path(args.questions_file))
    if args.ids:
        wanted = {item.strip().upper() for item in args.ids.split(",") if item.strip()}
        selected = [item for item in selected if item["id"].upper() in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No questions selected.")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the NVIDIA gold QA set against the local Microsoft GraphRAG "
            "index and grade answers with deterministic checks. Optionally, "
            "also use the local OpenAI-compatible model as a judge."
        )
    )
    parser.add_argument(
        "--questions-file",
        default=str(DEFAULT_QUESTIONS_FILE),
        help=(
            "JSONL or JSON file containing id, type, question, and gold_answer "
            "fields."
        ),
    )
    parser.add_argument("--root", default="ms_graph", help="GraphRAG project root.")
    parser.add_argument(
        "--cwd",
        default=".",
        help="Working directory for GraphRAG subprocesses.",
    )
    parser.add_argument(
        "--method",
        choices=["local", "global", "drift", "basic"],
        default="local",
        help="GraphRAG query method.",
    )
    parser.add_argument(
        "--response-type",
        default="Single Sentence",
        help="GraphRAG response type hint.",
    )
    parser.add_argument(
        "--query-runner",
        choices=["auto", "bin", "uv"],
        default="uv",
        help=(
            "How to launch GraphRAG. The default matches the project's "
            "documented uv workflow."
        ),
    )
    parser.add_argument("--uv-bin", default="uv", help="uv executable name/path.")
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=600,
        help="Seconds to wait for each GraphRAG query.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval/results",
        help="Directory where the timestamped evaluation run is written.",
    )
    parser.add_argument(
        "--ids",
        default="",
        help="Optional comma-separated question IDs to run, e.g. NVID-F01,NVID-N01.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of questions to run from the selected set.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GRAPHRAG_API_KEY", "local"),
        help="API key value for local OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also grade answers with the local model endpoint. The default uses "
            "fast deterministic checks tailored to this NVIDIA gold set."
        ),
    )
    parser.add_argument(
        "--judge-api-base",
        default=os.getenv("JUDGE_API_BASE", "http://127.0.0.1:8080/v1"),
        help="OpenAI-compatible chat completions base URL for judging.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "bonsai-local"),
        help="Model name sent to the judge endpoint.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=180,
        help="Seconds to wait for each judge call.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected_questions = _select_questions(args)

    run_id = datetime.now().strftime("nvidia_graphrag_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    total = len(selected_questions)

    for index, item in enumerate(selected_questions, start=1):
        print(f"[{index}/{total}] {item['id']} {item['question']}", flush=True)
        query = _run_query(args, item["question"])
        metrics = {
            "token_f1": _token_f1(query.answer, item["gold_answer"]),
            "gold_substring_match": _normalized_contains(
                query.answer,
                item["gold_answer"],
            ),
        }
        rule = _rule_grade(item, query.answer)
        judge = _judge_answer(args, item, query.answer)
        grade = _select_grade(rule, judge)

        row = {
            **item,
            "answer": query.answer,
            "query": asdict(query),
            "metrics": metrics,
            "rule_grade": rule,
            "judge": judge,
            "grade": grade,
        }
        rows.append(row)

        label = grade.get("label", "unknown")
        score = grade.get("score")
        score_text = "n/a" if score is None else str(score)
        print(
            f"  -> {label} score={score_text} source={grade.get('source')} "
            f"elapsed={query.elapsed_seconds:.1f}s",
            flush=True,
        )

        _write_jsonl(run_dir / "results.jsonl", rows)
        _write_csv(run_dir / "results.csv", rows)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "questions_file": args.questions_file,
            "root": args.root,
            "method": args.method,
            "response_type": args.response_type,
            "query_runner": args.query_runner,
            "judge": args.judge,
            "judge_api_base": args.judge_api_base,
            "judge_model": args.judge_model,
        },
        "summary": _summarize(rows),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(run_dir / "results.jsonl", rows)
    _write_csv(run_dir / "results.csv", rows)

    print(json.dumps(summary["summary"], indent=2), flush=True)
    print(f"Wrote evaluation run to {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
