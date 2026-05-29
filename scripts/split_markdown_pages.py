#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


START_RE = re.compile(r"^--- PAGE\s+(\d+)\s+START ---\s*$")
END_RE = re.compile(r"^--- PAGE\s+(\d+)\s+END ---\s*$")


def split_markdown_by_page(
    input_file: Path,
    output_dir: Path,
    zero_pad: int = 3,
    keep_markers: bool = False,
) -> int:
    lines = input_file.read_text(encoding="utf-8").splitlines(keepends=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    current_page: int | None = None
    buffer: list[str] = []
    pages_written = 0

    for line_no, line in enumerate(lines, start=1):
        start_match = START_RE.match(line)
        if start_match:
            if current_page is not None:
                raise ValueError(
                    f"Nested PAGE START at line {line_no} while page {current_page} is open."
                )

            current_page = int(start_match.group(1))
            buffer = [line] if keep_markers else []
            continue

        end_match = END_RE.match(line)
        if end_match:
            if current_page is None:
                raise ValueError(f"PAGE END without PAGE START at line {line_no}.")

            end_page = int(end_match.group(1))
            if end_page != current_page:
                raise ValueError(
                    f"Mismatched PAGE END at line {line_no}: expected {current_page}, got {end_page}."
                )

            if keep_markers:
                buffer.append(line)

            page_text = "".join(buffer).strip("\n")
            if page_text:
                page_text += "\n"

            output_name = f"{input_file.stem}_page_{current_page:0{zero_pad}d}.md"
            output_path = output_dir / output_name
            output_path.write_text(page_text, encoding="utf-8")

            pages_written += 1
            current_page = None
            buffer = []
            continue

        if current_page is not None:
            buffer.append(line)

    if current_page is not None:
        raise ValueError(f"Unclosed page block for page {current_page}.")

    return pages_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split markdown into page files using PAGE START/END markers."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to source markdown file (for example: ms_graph/input/nvidia.md)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_stem>_pages next to the input file).",
    )
    parser.add_argument(
        "--zero-pad",
        type=int,
        default=3,
        help="Zero-padding width for page number in output filenames (default: 3).",
    )
    parser.add_argument(
        "--keep-markers",
        action="store_true",
        help="Keep PAGE START/END marker lines in each output file.",
    )
    args = parser.parse_args()

    input_file = args.input.resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_file.parent / f"{input_file.stem}_pages"
    )

    pages_written = split_markdown_by_page(
        input_file=input_file,
        output_dir=output_dir,
        zero_pad=args.zero_pad,
        keep_markers=args.keep_markers,
    )
    print(f"Wrote {pages_written} page files to: {output_dir}")


if __name__ == "__main__":
    main()
