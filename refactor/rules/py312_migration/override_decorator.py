from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule


def _top_level_class_methods(tree: ast.AST, class_name: str) -> frozenset[str]:
    """Return the set of method names defined in the top-level ClassDef named *class_name*."""
    if not isinstance(tree, ast.Module):
        return frozenset()
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == class_name:
            return frozenset(
                node.name
                for node in stmt.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return frozenset()


def _has_override_decorator(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function already has @override or @typing.override."""
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "override":
            return True
        if (
            isinstance(dec, ast.Attribute)
            and dec.attr == "override"
            and isinstance(dec.value, ast.Name)
            and dec.value.id == "typing"
        ):
            return True
    return False


class OverrideDecoratorRule(Rule):
    """Add @typing.override to methods that override a same-module base class method.

    Conservative: only transforms when the base class is defined as a top-level
    ClassDef in the same module and has a method with the identical name.
    Does NOT add an import for typing — the developer must ensure ``import typing``
    (or ``from typing import override``) is already present.

    Enable via ``--enable=override-decorator``.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

        # Must not already be decorated with @override / @typing.override
        assert not _has_override_decorator(node)

        # Find the enclosing ClassDef via ancestry
        enclosing_class: ast.ClassDef | None = None
        for parent in self.context.ancestry.get_parents(node):
            if isinstance(parent, ast.ClassDef):
                enclosing_class = parent
                break

        assert enclosing_class is not None

        # The enclosing class must have at least one Name base
        base_names = [base.id for base in enclosing_class.bases if isinstance(base, ast.Name)]
        assert len(base_names) > 0

        # At least one Name base must be a top-level ClassDef in this module with
        # a method of the same name
        method_name = node.name
        found_in_base = any(
            method_name in _top_level_class_methods(self.context.tree, base_name)
            for base_name in base_names
        )
        assert found_in_base

        # Build new decorator: typing.override
        new_dec = ast.Attribute(
            value=ast.Name(id="typing", ctx=ast.Load()),
            attr="override",
            ctx=ast.Load(),
        )

        new_func = clone(node)
        new_func.decorator_list = [new_dec] + list(node.decorator_list)
        return Replace(node, new_func)
