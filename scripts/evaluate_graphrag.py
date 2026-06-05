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
DEFAULT_COMPANY_ROOTS_FILE = Path("data/eval/company_roots.json")
REQUIRED_QUESTION_FIELDS = ("id", "type", "question", "gold_answer")
VALID_LABELS = {"correct", "partial", "incorrect", "no_answer"}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TOKEN_RE = re.compile(r"[a-z0-9]+")
LABEL_RE = re.compile(r"\b(correct|partial|incorrect|no[_ -]?answer)\b")


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return round(sorted_values[midpoint], 4)
    return round((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2, 4)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)
    index = round((len(sorted_values) - 1) * percentile)
    return round(sorted_values[index], 4)


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


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

    raise ValueError(f"No JSON object found in model response: {text[:500]}")


def _coerce_judge_label(raw_label: Any, score: float | None, rationale: str) -> str:
    label = str(raw_label or "").strip().lower()
    label = label.strip("\"'`.,:; ")
    label = re.sub(r"[\s-]+", "_", label)

    if label in VALID_LABELS:
        return label

    matches = {
        match.group(1).replace("-", "_").replace(" ", "_")
        for match in LABEL_RE.finditer(label)
    }
    if len(matches) == 1:
        return matches.pop()

    if score is None:
        return "judge_error"

    if score >= 0.75:
        return "correct"
    if score >= 0.25:
        return "partial"

    normalized_rationale = rationale.lower().replace("-", "_").replace(" ", "_")
    if "no_answer" in normalized_rationale or "does_not_provide" in normalized_rationale:
        return "no_answer"
    return "incorrect"


def _call_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    max_tokens: int,
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
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
        raise ValueError(f"No choices returned by model endpoint: {body}")

    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    text = first.get("text")
    if isinstance(text, str):
        return text.strip()

    raise ValueError(f"Could not read model response content: {body}")


def _load_company_roots(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Company roots file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    roots: dict[str, str] = {}
    for company, root in payload.items():
        company_name = str(company).strip().lower()
        root_path = str(root).strip()
        if not company_name or not root_path:
            raise ValueError(f"Invalid company root mapping in {path}: {company!r}")
        roots[company_name] = root_path

    return roots


def _normalize_question_row(
    row: Any,
    *,
    source: Path,
    row_number: int,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{source}:{row_number} is not a JSON object.")

    normalized: dict[str, Any] = dict(row)
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

    if "company" in normalized and normalized["company"] is not None:
        normalized["company"] = str(normalized["company"]).strip().lower()

    if "companies" in normalized and normalized["companies"] is not None:
        companies = normalized["companies"]
        if not isinstance(companies, list):
            raise ValueError(f"{source}:{row_number} field 'companies' must be a list.")
        normalized["companies"] = [
            str(company).strip().lower() for company in companies if str(company).strip()
        ]
        if not normalized["companies"]:
            raise ValueError(f"{source}:{row_number} field 'companies' is empty.")

    return normalized


def _load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    questions: list[dict[str, Any]] = []
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


def _select_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = _load_questions(Path(args.questions_file))
    if args.ids:
        wanted = {item.strip().upper() for item in args.ids.split(",") if item.strip()}
        selected = [item for item in selected if item["id"].upper() in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No questions selected.")
    return selected


def _get_companies(item: dict[str, Any]) -> list[str]:
    if item.get("companies"):
        return list(dict.fromkeys(item["companies"]))
    company = str(item.get("company", "")).strip().lower()
    return [company] if company else []


def _is_comparison_item(item: dict[str, Any]) -> bool:
    companies = _get_companies(item)
    return len(companies) > 1 or str(item.get("type", "")).lower() == "comparison"


def _build_query_command(
    *,
    root: str,
    question: str,
    method: str,
    response_type: str,
    query_runner: str,
    uv_bin: str,
) -> list[str]:
    root_path = Path(root)

    if query_runner in {"bin", "auto"}:
        graphrag_bin = root_path / ".venv" / "bin" / "graphrag"
        if graphrag_bin.exists() and os.access(graphrag_bin, os.X_OK):
            return [
                str(graphrag_bin),
                "query",
                "--root",
                str(root_path),
                "--method",
                method,
                "--response-type",
                response_type,
                question,
            ]
        if query_runner == "bin":
            raise FileNotFoundError(f"GraphRAG binary not found: {graphrag_bin}")

    return [
        uv_bin,
        "run",
        "--project",
        str(root_path),
        "graphrag",
        "query",
        "--root",
        str(root_path),
        "--method",
        method,
        "--response-type",
        response_type,
        question,
    ]


def _run_query(
    args: argparse.Namespace,
    *,
    root: str,
    question: str,
    response_type: str | None = None,
) -> QueryResult:
    command = _build_query_command(
        root=root,
        question=question,
        method=args.method,
        response_type=response_type or args.response_type,
        query_runner=args.query_runner,
        uv_bin=args.uv_bin,
    )
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


def _judge_answer(
    args: argparse.Namespace,
    item: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    if args.grading == "none":
        return {
            "enabled": False,
            "label": "not_judged",
            "score": None,
            "hallucination": None,
            "rationale": "",
        }

    if args.grading == "token":
        token_f1 = _token_f1(answer, item["gold_answer"])["f1"]
        substring_match = _normalized_contains(answer, item["gold_answer"])
        if substring_match or token_f1 >= 0.8:
            label = "correct"
            score = 1.0
        elif token_f1 >= 0.35:
            label = "partial"
            score = 0.5
        else:
            label = "incorrect"
            score = 0.0
        return {
            "enabled": True,
            "label": label,
            "score": score,
            "hallucination": None,
            "rationale": f"Token baseline grade with F1={token_f1}.",
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
        '"label":"correct",'
        '"score":1.0,'
        '"hallucination":false,'
        '"rationale":"short reason"'
        "}\n\n"
        "The label value must be exactly one of: correct, partial, "
        "incorrect, no_answer. Do not return the list of label choices as "
        "the label.\n\n"
        "Scoring rules: correct=1.0 when all essential facts match; "
        "partial=0.5 when some essential facts are present but important detail "
        "is missing; incorrect/no_answer=0.0. For comparison questions, all "
        "company-specific facts in the gold answer must be handled correctly.\n\n"
        f"Question ID: {item['id']}\n"
        f"Question type: {item['type']}\n"
        f"Companies: {', '.join(_get_companies(item)) or item.get('company', '')}\n"
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
            max_tokens=args.judge_max_tokens,
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

    try:
        score = float(parsed.get("score"))
    except (TypeError, ValueError):
        score = None

    raw_label = parsed.get("label", "judge_error")
    rationale = str(parsed.get("rationale", "")).strip()
    label = _coerce_judge_label(raw_label, score, rationale)

    if label == "correct":
        score = 1.0
    elif label == "partial":
        score = 0.5
    elif label in {"incorrect", "no_answer"}:
        score = 0.0

    return {
        "enabled": True,
        "label": label,
        "raw_label": raw_label,
        "score": score,
        "hallucination": bool(parsed.get("hallucination", False)),
        "rationale": rationale,
    }


def _comparison_subquestion(item: dict[str, Any], company: str) -> str:
    subquestions = item.get("subquestions")
    if isinstance(subquestions, dict):
        value = subquestions.get(company)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return (
        f"For {company}, answer only the parts of this cross-company ESG "
        f"comparison question that are supported by the {company} report: "
        f"{item['question']}"
    )


def _synthesize_comparison_answer(
    args: argparse.Namespace,
    item: dict[str, Any],
    company_results: dict[str, QueryResult],
) -> QueryResult:
    context_blocks: list[str] = []
    for company, result in company_results.items():
        answer = result.answer.strip() or "No answer returned."
        context_blocks.append(f"[{company}]\n{answer}")

    system = (
        "You synthesize cross-company ESG comparison answers. Use only the "
        "provided per-company GraphRAG answers. If a company answer lacks a "
        "requested fact, say that it is not available from that company answer. "
        "Keep the answer concise and preserve company names and numeric units."
    )
    user = (
        f"Comparison question:\n{item['question']}\n\n"
        "Per-company GraphRAG answers:\n"
        f"{chr(10).join(context_blocks)}\n\n"
        "Final comparison answer:"
    )

    started = time.perf_counter()
    try:
        answer = _call_chat_completion(
            api_base=args.judge_api_base,
            api_key=args.api_key,
            model=args.judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=args.synthesis_timeout,
            max_tokens=args.synthesis_max_tokens,
        )
        elapsed = time.perf_counter() - started
        failed_inputs = [
            company
            for company, result in company_results.items()
            if result.returncode != 0
        ]
        return QueryResult(
            answer=answer,
            raw_stdout=answer,
            raw_stderr="",
            returncode=1 if failed_inputs else 0,
            elapsed_seconds=round(elapsed, 3),
            command=[],
            error=(
                f"Some company queries failed: {', '.join(failed_inputs)}"
                if failed_inputs
                else None
            ),
        )
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        elapsed = time.perf_counter() - started
        return QueryResult(
            answer="",
            raw_stdout="",
            raw_stderr="",
            returncode=1,
            elapsed_seconds=round(elapsed, 3),
            command=[],
            error=str(exc),
        )


def _run_single_company_item(
    args: argparse.Namespace,
    item: dict[str, Any],
    company_roots: dict[str, str],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    company = str(item.get("company", "")).strip().lower()
    if not company:
        raise ValueError(f"{item['id']} missing company field.")
    root = company_roots.get(company)
    if not root:
        raise ValueError(f"{item['id']} uses unknown company root: {company}")

    result = _run_query(args, root=root, question=item["question"])
    query_payload = asdict(result)
    query_payload["company"] = company
    query_payload["root"] = root
    return result.answer, query_payload, {}


def _run_comparison_item(
    args: argparse.Namespace,
    item: dict[str, Any],
    company_roots: dict[str, str],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    companies = _get_companies(item)
    if len(companies) < 2:
        raise ValueError(f"{item['id']} comparison questions require companies.")

    company_results: dict[str, QueryResult] = {}
    for company in companies:
        root = company_roots.get(company)
        if not root:
            raise ValueError(f"{item['id']} uses unknown company root: {company}")
        subquestion = _comparison_subquestion(item, company)
        company_results[company] = _run_query(
            args,
            root=root,
            question=subquestion,
            response_type=args.comparison_response_type,
        )

    synthesized = _synthesize_comparison_answer(args, item, company_results)
    query_payload = asdict(synthesized)
    query_payload["elapsed_seconds"] = round(
        synthesized.elapsed_seconds
        + sum(result.elapsed_seconds for result in company_results.values()),
        3,
    )
    query_payload["company_queries"] = {
        company: {
            **asdict(result),
            "root": company_roots[company],
            "subquestion": _comparison_subquestion(item, company),
        }
        for company, result in company_results.items()
    }
    query_payload["companies"] = companies
    return synthesized.answer, query_payload, query_payload["company_queries"]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successful_queries = sum(1 for row in rows if row["query"]["returncode"] == 0)
    graded_rows = [
        row for row in rows if isinstance(row["judge"].get("score"), int | float)
    ]
    label_counts: dict[str, int] = {}
    type_counts: dict[str, dict[str, Any]] = {}
    company_counts: dict[str, dict[str, Any]] = {}
    elapsed_values: list[float] = []
    token_precision_values: list[float] = []
    token_recall_values: list[float] = []
    token_f1_values: list[float] = []
    substring_match_values: list[bool] = []
    hallucination_values: list[bool] = []

    for row in rows:
        label = str(row["judge"].get("label", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1
        elapsed = row.get("query", {}).get("elapsed_seconds")
        token_scores = row.get("metrics", {}).get("token_f1", {})
        token_precision = token_scores.get("precision")
        token_recall = token_scores.get("recall")
        token_f1 = token_scores.get("f1")
        substring_match = row.get("metrics", {}).get("gold_substring_match")
        hallucination = row.get("judge", {}).get("hallucination")

        if isinstance(elapsed, int | float):
            elapsed_values.append(float(elapsed))
        if isinstance(token_precision, int | float):
            token_precision_values.append(float(token_precision))
        if isinstance(token_recall, int | float):
            token_recall_values.append(float(token_recall))
        if isinstance(token_f1, int | float):
            token_f1_values.append(float(token_f1))
        if isinstance(substring_match, bool):
            substring_match_values.append(substring_match)
        if isinstance(hallucination, bool):
            hallucination_values.append(hallucination)

        for bucket_key, counts in (
            (row["type"], type_counts),
            (row.get("company") or ",".join(row.get("companies", [])), company_counts),
        ):
            if not bucket_key:
                continue
            bucket = counts.setdefault(
                bucket_key,
                {
                    "total": 0,
                    "correct": 0,
                    "partial": 0,
                    "incorrect": 0,
                    "no_answer": 0,
                    "scores": [],
                    "elapsed_seconds": [],
                    "token_precision": [],
                    "token_recall": [],
                    "token_f1": [],
                    "gold_substring_match": [],
                    "hallucination": [],
                },
            )
            bucket["total"] += 1
            if label in VALID_LABELS:
                bucket[label] += 1
            score = row["judge"].get("score")
            if isinstance(score, int | float):
                bucket["scores"].append(float(score))
            if isinstance(elapsed, int | float):
                bucket["elapsed_seconds"].append(float(elapsed))
            if isinstance(token_precision, int | float):
                bucket["token_precision"].append(float(token_precision))
            if isinstance(token_recall, int | float):
                bucket["token_recall"].append(float(token_recall))
            if isinstance(token_f1, int | float):
                bucket["token_f1"].append(float(token_f1))
            if isinstance(substring_match, bool):
                bucket["gold_substring_match"].append(substring_match)
            if isinstance(hallucination, bool):
                bucket["hallucination"].append(hallucination)

    def finalize_counts(counts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for key, bucket in counts.items():
            scores = bucket.pop("scores")
            elapsed = bucket.pop("elapsed_seconds")
            token_precision = bucket.pop("token_precision")
            token_recall = bucket.pop("token_recall")
            token_f1 = bucket.pop("token_f1")
            substring_matches = bucket.pop("gold_substring_match")
            hallucinations = bucket.pop("hallucination")
            output[key] = {
                **bucket,
                "average_score": _mean(scores),
                "correct_rate": (
                    round(bucket["correct"] / bucket["total"], 4)
                    if bucket["total"]
                    else None
                ),
                "partial_or_better_rate": (
                    round((bucket["correct"] + bucket["partial"]) / bucket["total"], 4)
                    if bucket["total"]
                    else None
                ),
                "average_elapsed_seconds": _mean(elapsed),
                "median_elapsed_seconds": _median(elapsed),
                "p95_elapsed_seconds": _percentile(elapsed, 0.95),
                "average_token_precision": _mean(token_precision),
                "average_token_recall": _mean(token_recall),
                "average_token_f1": _mean(token_f1),
                "gold_substring_match_rate": _rate(substring_matches),
                "hallucination_rate": _rate(hallucinations),
            }
        return output

    score_values = [float(row["judge"]["score"]) for row in graded_rows]

    return {
        "total_questions": total,
        "successful_queries": successful_queries,
        "query_failures": total - successful_queries,
        "label_counts": label_counts,
        "average_score": _mean(score_values),
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
        "average_elapsed_seconds": _mean(elapsed_values),
        "median_elapsed_seconds": _median(elapsed_values),
        "p95_elapsed_seconds": _percentile(elapsed_values, 0.95),
        "average_token_precision": _mean(token_precision_values),
        "average_token_recall": _mean(token_recall_values),
        "average_token_f1": _mean(token_f1_values),
        "gold_substring_match_rate": _rate(substring_match_values),
        "hallucination_rate": _rate(hallucination_values),
        "by_type": finalize_counts(type_counts),
        "by_company": finalize_counts(company_counts),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "type",
        "method",
        "company",
        "companies",
        "question",
        "gold_answer",
        "answer",
        "returncode",
        "elapsed_seconds",
        "judge_label",
        "judge_score",
        "judge_hallucination",
        "judge_rationale",
        "token_precision",
        "token_recall",
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
                    "method": row.get("method", ""),
                    "company": row.get("company", ""),
                    "companies": ",".join(row.get("companies", [])),
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "answer": row["answer"],
                    "returncode": row["query"]["returncode"],
                    "elapsed_seconds": row["query"]["elapsed_seconds"],
                    "judge_label": row["judge"].get("label"),
                    "judge_score": row["judge"].get("score"),
                    "judge_hallucination": row["judge"].get("hallucination"),
                    "judge_rationale": row["judge"].get("rationale"),
                    "token_precision": row["metrics"]["token_f1"]["precision"],
                    "token_recall": row["metrics"]["token_f1"]["recall"],
                    "token_f1": row["metrics"]["token_f1"]["f1"],
                    "gold_substring_match": row["metrics"]["gold_substring_match"],
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Microsoft GraphRAG answers against company-specific, "
            "multi-hop, or cross-company ESG gold question sets."
        )
    )
    parser.add_argument(
        "--questions-file",
        default=str(DEFAULT_QUESTIONS_FILE),
        help="JSONL or JSON file containing gold question rows.",
    )
    parser.add_argument(
        "--company-roots-file",
        default=str(DEFAULT_COMPANY_ROOTS_FILE),
        help="JSON file mapping company names to GraphRAG roots.",
    )
    parser.add_argument(
        "--cwd",
        default=".",
        help="Working directory for GraphRAG subprocesses.",
    )
    parser.add_argument(
        "--method",
        choices=["local", "global", "drift", "basic"],
        default="basic",
        help="GraphRAG query method.",
    )
    parser.add_argument(
        "--response-type",
        default="Single Sentence",
        help="GraphRAG response type hint for normal questions.",
    )
    parser.add_argument(
        "--comparison-response-type",
        default="Single Sentence",
        help="GraphRAG response type hint for per-company comparison subquestions.",
    )
    parser.add_argument(
        "--query-runner",
        choices=["auto", "bin", "uv"],
        default="uv",
        help="How to launch GraphRAG.",
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
        help="Optional comma-separated question IDs to run.",
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
        "--grading",
        choices=["judge", "token", "none"],
        default="judge",
        help="Answer grading strategy. Default is local model judge.",
    )
    parser.add_argument(
        "--judge-api-base",
        default=os.getenv("JUDGE_API_BASE", "http://127.0.0.1:8080/v1"),
        help="OpenAI-compatible chat completions base URL for judging/synthesis.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "bonsai-local"),
        help="Model name sent to the judge/synthesis endpoint.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=180,
        help="Seconds to wait for each judge call.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=300,
        help="Maximum judge completion tokens.",
    )
    parser.add_argument(
        "--synthesis-timeout",
        type=int,
        default=180,
        help="Seconds to wait for each comparison synthesis call.",
    )
    parser.add_argument(
        "--synthesis-max-tokens",
        type=int,
        default=700,
        help="Maximum comparison synthesis completion tokens.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected_questions = _select_questions(args)
    company_roots = _load_company_roots(Path(args.company_roots_file))

    stem = Path(args.questions_file).stem.replace("_gold_questions", "")
    run_id = (
        f"graphrag_eval_{stem}_{args.method}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    total = len(selected_questions)

    for index, item in enumerate(selected_questions, start=1):
        print(f"[{index}/{total}] {item['id']} {item['question']}", flush=True)
        try:
            if _is_comparison_item(item):
                answer, query_payload, company_queries = _run_comparison_item(
                    args,
                    item,
                    company_roots,
                )
            else:
                answer, query_payload, company_queries = _run_single_company_item(
                    args,
                    item,
                    company_roots,
                )
        except Exception as exc:
            answer = ""
            query_payload = {
                "answer": "",
                "raw_stdout": "",
                "raw_stderr": "",
                "returncode": 1,
                "elapsed_seconds": 0.0,
                "command": [],
                "error": str(exc),
            }
            company_queries = {}

        metrics = {
            "token_f1": _token_f1(answer, item["gold_answer"]),
            "gold_substring_match": _normalized_contains(answer, item["gold_answer"]),
        }
        judge = _judge_answer(args, item, answer)

        row = {
            **item,
            "method": args.method,
            "answer": answer,
            "query": query_payload,
            "company_queries": company_queries,
            "metrics": metrics,
            "judge": judge,
        }
        rows.append(row)

        label = judge.get("label", "not_judged")
        score = judge.get("score")
        score_text = "n/a" if score is None else str(score)
        print(
            f"  -> {label} score={score_text} "
            f"elapsed={query_payload['elapsed_seconds']:.1f}s",
            flush=True,
        )

        _write_jsonl(run_dir / "results.jsonl", rows)
        _write_csv(run_dir / "results.csv", rows)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "questions_file": args.questions_file,
            "company_roots_file": args.company_roots_file,
            "method": args.method,
            "response_type": args.response_type,
            "comparison_response_type": args.comparison_response_type,
            "query_runner": args.query_runner,
            "grading": args.grading,
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
