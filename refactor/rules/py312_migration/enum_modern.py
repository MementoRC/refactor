from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule


def _is_name(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_attr(node: ast.expr, value_name: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == value_name
        and node.attr == attr
    )


class StrEnumRule(Rule):
    """Replace ``class Foo(str, Enum):`` with ``class Foo(StrEnum):``.

    StrEnum was added in Python 3.11 and is the idiomatic replacement for the
    ``(str, Enum)`` mixin pattern.

    This rule changes the class bases only; the developer must update the
    ``from enum import Enum`` import to ``from enum import StrEnum`` (or add
    it) manually.  A follow-up rule can automate this.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ClassDef)
        assert len(node.bases) == 2
        first, second = node.bases
        assert _is_name(first, "str")
        if _is_name(second, "Enum"):
            new_base: ast.expr = ast.Name(id="StrEnum", ctx=ast.Load())
        elif _is_attr(second, "enum", "Enum"):
            new_base = ast.Attribute(
                value=ast.Name(id="enum", ctx=ast.Load()),
                attr="StrEnum",
                ctx=ast.Load(),
            )
        else:
            return None
        new_node = clone(node)
        new_node.bases = [new_base]
        return Replace(node, new_node)


class IntEnumRule(Rule):
    """Replace ``class Foo(int, Enum):`` with ``class Foo(IntEnum):``.

    IntEnum has existed since Python 3.4, but the ``(int, Enum)`` mixin form
    is non-idiomatic.

    This rule changes the class bases only; the developer must ensure that
    ``IntEnum`` is importable (``from enum import IntEnum``) at the call site.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ClassDef)
        assert len(node.bases) == 2
        first, second = node.bases
        assert _is_name(first, "int")
        if _is_name(second, "Enum"):
            new_base: ast.expr = ast.Name(id="IntEnum", ctx=ast.Load())
        elif _is_attr(second, "enum", "Enum"):
            new_base = ast.Attribute(
                value=ast.Name(id="enum", ctx=ast.Load()),
                attr="IntEnum",
                ctx=ast.Load(),
            )
        else:
            return None
        new_node = clone(node)
        new_node.bases = [new_base]
        return Replace(node, new_node)
