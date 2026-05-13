from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule

# Mapping from typing alias attr name → builtin name (excluding Type, handled separately)
_DEPRECATED_ALIASES: dict[str, str] = {
    "Dict": "dict",
    "List": "list",
    "Set": "set",
    "FrozenSet": "frozenset",
    "Tuple": "tuple",
}


def _typing_imports(tree: ast.AST) -> frozenset[str]:
    """Return the set of names imported from `typing` at module level."""
    imported: set[str] = set()
    if not isinstance(tree, ast.Module):
        return frozenset()
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "typing":
            for alias in stmt.names:
                imported.add(alias.asname if alias.asname else alias.name)
    return frozenset(imported)


class TypingDeprecatedAliasRule(Rule):
    """Replace deprecated typing aliases with builtin generics in subscripted contexts.

    Handles both qualified form (typing.Dict[X]) and bare form (Dict[X] when
    ``from typing import Dict`` is present).  The slice contents are preserved
    unchanged — only the subscript value (the alias) is replaced with the
    builtin name.

    Targets: Dict→dict, List→list, Set→set, FrozenSet→frozenset, Tuple→tuple.
    typing.Type is handled by TypingTypeRule.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Subscript)

        value = node.value

        if isinstance(value, ast.Attribute):
            # Qualified form: typing.Dict[...]
            assert isinstance(value.value, ast.Name)
            assert value.value.id == "typing"
            assert value.attr in _DEPRECATED_ALIASES
            builtin_name = _DEPRECATED_ALIASES[value.attr]
        elif isinstance(value, ast.Name):
            # Bare form: Dict[...] — only when imported from typing
            assert value.id in _DEPRECATED_ALIASES
            imported = _typing_imports(self.context.tree)
            assert value.id in imported
            builtin_name = _DEPRECATED_ALIASES[value.id]
        else:
            return None

        new_node = clone(node)
        new_node.value = ast.Name(id=builtin_name, ctx=ast.Load())
        return Replace(node, new_node)


class TypingTypeRule(Rule):
    """Replace typing.Type[X] and bare Type[X] (when imported) with type[X].

    Kept as a separate rule from TypingDeprecatedAliasRule because the
    transformation shadows the ``type`` builtin keyword (used in ``type(obj)``
    calls), making it mildly riskier than the other alias replacements.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Subscript)

        value = node.value

        if isinstance(value, ast.Attribute):
            # Qualified form: typing.Type[...]
            assert isinstance(value.value, ast.Name)
            assert value.value.id == "typing"
            assert value.attr == "Type"
        elif isinstance(value, ast.Name):
            # Bare form: Type[...] — only when imported from typing
            assert value.id == "Type"
            imported = _typing_imports(self.context.tree)
            assert "Type" in imported
        else:
            return None

        new_node = clone(node)
        new_node.value = ast.Name(id="type", ctx=ast.Load())
        return Replace(node, new_node)


class TypingOptionalRule(Rule):
    """Replace Optional[X] with X | None.

    Handles both qualified (typing.Optional[X]) and bare (Optional[X] when
    imported from typing) forms.

    Edge case: Optional[X | Y] → X | Y | None.  The existing BitOr chain is
    preserved as the left operand; None is chained on the right without extra
    parentheses.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Subscript)

        value = node.value

        if isinstance(value, ast.Attribute):
            # Qualified form: typing.Optional[...]
            assert isinstance(value.value, ast.Name)
            assert value.value.id == "typing"
            assert value.attr == "Optional"
        elif isinstance(value, ast.Name):
            # Bare form: Optional[...] — only when imported from typing
            assert value.id == "Optional"
            imported = _typing_imports(self.context.tree)
            assert "Optional" in imported
        else:
            return None

        # Extract the single slice argument (the inner type)
        inner = node.slice

        # Build X | None (chains correctly even when inner is already a BitOr)
        new_node = ast.BinOp(
            left=inner,
            op=ast.BitOr(),
            right=ast.Constant(value=None),
        )
        return Replace(node, new_node)
