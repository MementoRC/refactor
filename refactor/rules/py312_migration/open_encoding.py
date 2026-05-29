from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule


def _is_binary_mode(mode_node: ast.expr) -> bool:
    """Return True if the mode expression is a string literal containing 'b'."""
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return "b" in mode_node.value
    # Non-literal expression — be conservative and treat as unknown (not binary)
    return False


def _mode_is_unknown_non_literal(mode_node: ast.expr) -> bool:
    """Return True when the mode is a non-literal expression (variable, call, etc.)."""
    return not isinstance(mode_node, ast.Constant)


class OpenEncodingRule(Rule):
    """Add ``encoding='utf-8'`` to bare ``open()`` calls in text mode.

    py3.12 emits an EncodingWarning for ``open()`` without an explicit
    ``encoding`` argument when the locale encoding differs from UTF-8; py3.15
    will likely turn this into an error.

    Conservative match:
    - Only bare ``open(...)`` by name (not ``builtins.open``).
    - Skips calls that already have an ``encoding=`` keyword.
    - Skips calls where mode contains ``'b'`` (binary).
    - Skips calls where mode is a non-literal expression (unknown — be safe).

    Enable via ``--enable=encoding-warning``.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Call)

        # Only bare name open(...)
        assert isinstance(node.func, ast.Name)
        assert node.func.id == "open"

        # Skip if encoding= is already present
        existing_kw_names = {kw.arg for kw in node.keywords}
        assert "encoding" not in existing_kw_names

        # Determine mode from 2nd positional arg or mode= keyword
        mode_node: ast.expr | None = None
        if len(node.args) >= 2:
            mode_node = node.args[1]
        else:
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode_node = kw.value
                    break

        if mode_node is not None:
            # Skip binary mode
            assert not _is_binary_mode(mode_node)
            # Skip non-literal mode (conservative — don't guess)
            assert not _mode_is_unknown_non_literal(mode_node)

        # Build the new keyword: encoding="utf-8"
        encoding_kw = ast.keyword(arg="encoding", value=ast.Constant(value="utf-8"))

        new_call = clone(node)
        new_call.keywords = list(node.keywords) + [encoding_kw]
        return Replace(node, new_call)
