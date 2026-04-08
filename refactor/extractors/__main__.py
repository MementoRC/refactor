"""CLI entry point for the documentation extraction engine.

Usage::

    python -m refactor.extractors --module path/to/file.py
    python -m refactor.extractors --package path/to/package/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from refactor.extractors.doc_gen import extract_module, extract_package, to_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m refactor.extractors",
        description="Extract structured documentation IR from Python source files.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--module",
        metavar="FILE",
        help="Single .py file to extract",
    )
    group.add_argument(
        "--package",
        metavar="DIR",
        help="Directory to extract recursively",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        default=False,
        help="Include _private methods and attributes",
    )
    parser.add_argument(
        "--include-dunder",
        action="store_true",
        default=False,
        help="Include __dunder__ methods",
    )
    parser.add_argument(
        "--include-inherited",
        action="store_true",
        default=False,
        help="Include methods from parent classes",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Log extraction progress to stderr",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the extraction CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    options = {
        "include_private": args.include_private,
        "include_dunder": args.include_dunder,
        "include_inherited": args.include_inherited,
    }

    if args.module:
        path = Path(args.module)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1
        source = path.read_text(encoding="utf-8")
        if args.verbose:
            print(f"Extracting module: {path}", file=sys.stderr)
        result = extract_module(source, str(path), options)
    else:
        pkg_path = Path(args.package)
        if not pkg_path.is_dir():
            print(f"Error: directory not found: {pkg_path}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"Extracting package: {pkg_path}", file=sys.stderr)
        result = extract_package(str(pkg_path), options)

    output = to_json(result, indent=args.indent)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        if args.verbose:
            print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
