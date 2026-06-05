from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_FILES = (
    Path("data/eval/nvidia_gold_questions.jsonl"),
    Path("data/eval/meta_gold_questions.jsonl"),
    Path("data/eval/google_gold_questions.jsonl"),
    Path("data/eval/comparison_gold_questions.jsonl"),
)
DEFAULT_QUESTIONS_FILE = Path("data/eval/all_gold_questions.jsonl")
DEFAULT_COMPANY_ROOTS_FILE = Path("data/eval/company_roots.json")
DEFAULT_METHODS = ("basic", "local", "global")
VALID_METHODS = {"basic", "local", "global", "drift"}
RUN_DIR_RE = re.compile(r"Wrote evaluation run to (.+)")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Question source file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSONL.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object.")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_combined_questions(
    *,
    source_files: list[Path],
    questions_file: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for source_file in source_files:
        for row in _read_jsonl(source_file):
            question_id = str(row.get("id", "")).strip().upper()
            if not question_id:
                raise ValueError(f"{source_file} contains a row without an id.")
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question id across sources: {question_id}")
            seen_ids.add(question_id)
            rows.append(row)

    _write_jsonl(questions_file, rows)
    return rows


def _question_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(row.get("type", "unknown")) for row in rows)
    by_company = Counter(
        str(row.get("company") or ",".join(row.get("companies", [])) or "unknown")
        for row in rows
    )
    return {
        "total_questions": len(rows),
        "by_type": dict(sorted(by_type.items())),
        "by_company": dict(sorted(by_company.items())),
    }


def _stream_command(command: list[str], *, env: dict[str, str]) -> str:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    returncode = process.wait()
    output = "".join(output_lines)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=output)
    return output


def _run_method(args: argparse.Namespace, method: str) -> Path:
    evaluator = Path(__file__).with_name("evaluate_graphrag.py")
    command = [
        sys.executable,
        str(evaluator),
        "--questions-file",
        str(args.questions_file),
        "--company-roots-file",
        str(args.company_roots_file),
        "--method",
        method,
        "--output-dir",
        str(args.output_dir),
        "--cwd",
        args.cwd,
        "--query-runner",
        args.query_runner,
        "--uv-bin",
        args.uv_bin,
        "--response-type",
        args.response_type,
        "--comparison-response-type",
        args.comparison_response_type,
        "--query-timeout",
        str(args.query_timeout),
        "--grading",
        args.grading,
        "--judge-api-base",
        args.judge_api_base,
        "--judge-model",
        args.judge_model,
        "--judge-timeout",
        str(args.judge_timeout),
        "--judge-max-tokens",
        str(args.judge_max_tokens),
        "--synthesis-timeout",
        str(args.synthesis_timeout),
        "--synthesis-max-tokens",
        str(args.synthesis_max_tokens),
    ]

    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.ids:
        command.extend(["--ids", args.ids])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    print(f"\n=== Running {method} evaluation ===", flush=True)
    output = _stream_command(command, env=os.environ.copy())
    match = RUN_DIR_RE.search(output)
    if not match:
        raise RuntimeError(f"Could not determine output run directory for {method}.")

    run_dir = Path(match.group(1).strip())
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    return run_dir


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Evaluation summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_matrix_outputs(
    *,
    output_dir: Path,
    questions_file: Path,
    source_files: list[Path],
    run_dirs: dict[str, Path],
) -> tuple[Path, Path]:
    created_at = datetime.now().isoformat(timespec="seconds")
    matrix: dict[str, Any] = {
        "created_at": created_at,
        "questions_file": str(questions_file),
        "source_files": [str(path) for path in source_files],
        "runs": {},
    }

    for method, run_dir in run_dirs.items():
        summary = _load_summary(run_dir)
        matrix["runs"][method] = {
            "run_dir": str(run_dir),
            "summary": summary["summary"],
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"evaluation_matrix_{timestamp}.json"
    csv_path = output_dir / f"evaluation_matrix_{timestamp}.csv"
    json_path.write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "method",
        "run_dir",
        "total_questions",
        "successful_queries",
        "query_failures",
        "average_score",
        "correct_rate",
        "partial_or_better_rate",
        "hallucination_rate",
        "average_elapsed_seconds",
        "median_elapsed_seconds",
        "p95_elapsed_seconds",
        "average_token_precision",
        "average_token_recall",
        "average_token_f1",
        "gold_substring_match_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, payload in matrix["runs"].items():
            summary = payload["summary"]
            writer.writerow(
                {
                    "method": method,
                    "run_dir": payload["run_dir"],
                    **{field: summary.get(field) for field in fieldnames[2:]},
                }
            )

    return json_path, csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 120-question ESG benchmark and run the GraphRAG "
            "evaluation across basic, local, and global retrieval."
        )
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
        help="Combined JSONL benchmark path to write/read.",
    )
    parser.add_argument(
        "--source-files",
        default=",".join(str(path) for path in DEFAULT_SOURCE_FILES),
        help=(
            "Comma-separated source JSONL files. Defaults to NVIDIA, Meta, "
            "Google, and cross-company files; multihop_gold_questions.jsonl "
            "is intentionally excluded to avoid duplicate multi-hop rows."
        ),
    )
    parser.add_argument(
        "--company-roots-file",
        type=Path,
        default=DEFAULT_COMPANY_ROOTS_FILE,
        help="JSON mapping from company name to GraphRAG root.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated retrieval methods to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval/results"),
        help="Directory for evaluation run outputs.",
    )
    parser.add_argument("--cwd", default=".", help="Evaluator subprocess cwd.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use the existing combined questions file without rebuilding it.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build and validate the combined questions file.",
    )
    parser.add_argument(
        "--query-runner",
        choices=["auto", "bin", "uv"],
        default="uv",
        help="How evaluate_graphrag.py should launch GraphRAG.",
    )
    parser.add_argument("--uv-bin", default="uv", help="uv executable name/path.")
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=600,
        help="Seconds to wait for each GraphRAG query.",
    )
    parser.add_argument(
        "--response-type",
        default="Single Sentence",
        help="GraphRAG response type for normal questions.",
    )
    parser.add_argument(
        "--comparison-response-type",
        default="Single Sentence",
        help="GraphRAG response type for per-company comparison subquestions.",
    )
    parser.add_argument(
        "--ids",
        default="",
        help="Optional comma-separated question IDs passed to each method run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of questions passed to each method run.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key passed through to evaluate_graphrag.py.",
    )
    parser.add_argument(
        "--grading",
        choices=["judge", "token", "none"],
        default="judge",
        help="Answer grading strategy.",
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
    source_files = [Path(path) for path in _parse_csv_list(args.source_files)]
    methods = _parse_csv_list(args.methods)
    if not methods:
        raise ValueError("At least one retrieval method is required.")
    invalid_methods = sorted(set(methods) - VALID_METHODS)
    if invalid_methods:
        raise ValueError(
            f"Unsupported retrieval method(s): {', '.join(invalid_methods)}"
        )

    if args.skip_build:
        rows = _read_jsonl(args.questions_file)
    else:
        rows = _build_combined_questions(
            source_files=source_files,
            questions_file=args.questions_file,
        )

    counts = _question_counts(rows)
    print(json.dumps(counts, indent=2), flush=True)

    if args.build_only:
        print(f"Wrote combined benchmark to {args.questions_file}", flush=True)
        return 0

    run_dirs: dict[str, Path] = {}
    for method in methods:
        run_dirs[method] = _run_method(args, method)

    json_path, csv_path = _write_matrix_outputs(
        output_dir=args.output_dir,
        questions_file=args.questions_file,
        source_files=source_files,
        run_dirs=run_dirs,
    )
    print(f"\nWrote matrix summary to {json_path}", flush=True)
    print(f"Wrote matrix CSV to {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
