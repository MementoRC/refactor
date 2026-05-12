from __future__ import annotations

from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

from refactor.core import Session
from refactor.rules.py312_migration import (
    CORE_RULES,
    IDIOMATIC_RULES,
    OPTIN_RULE_GROUPS,
)
from refactor.runner import expand_paths, run_files


def main() -> int:
    parser = ArgumentParser(
        prog="refactor-py312",
        description=(
            "AST-based Python 3.12 modernization rules. "
            "Default invocation runs CORE_RULES (breaking-change fixes) plus IDIOMATIC_RULES "
            "(non-breaking modernization). Higher-risk semantic transforms must be opted into "
            "with --enable=<group>."
        ),
    )
    parser.add_argument(
        "src", nargs="+", type=Path, help="Source paths (files or directories) to transform."
    )
    parser.add_argument(
        "-n",
        "--dont-apply",
        action="store_false",
        default=True,
        help="Show proposed changes without writing them.",
    )
    parser.add_argument(
        "--enable",
        action="append",
        default=[],
        metavar="GROUP",
        help=(
            "Enable an opt-in rule group. May be passed multiple times. "
            "Available groups: " + ", ".join(sorted(OPTIN_RULE_GROUPS) | {"all"})
            if OPTIN_RULE_GROUPS
            else "Enable an opt-in rule group (no groups available in this build)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1).",
    )

    options = parser.parse_args()

    rules = list(CORE_RULES) + list(IDIOMATIC_RULES)
    for group in options.enable:
        if group == "all":
            for group_rules in OPTIN_RULE_GROUPS.values():
                rules.extend(group_rules)
        elif group in OPTIN_RULE_GROUPS:
            rules.extend(OPTIN_RULE_GROUPS[group])
        else:
            parser.error(
                f"unknown --enable group: {group!r}. "
                f"Available: {sorted(OPTIN_RULE_GROUPS) + ['all']}"
            )

    session = Session(rules)
    files = chain.from_iterable(expand_paths(source_dest) for source_dest in options.src)
    return run_files(session, files, apply=options.dont_apply, workers=options.workers)


if __name__ == "__main__":
    raise SystemExit(main())
