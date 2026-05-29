from __future__ import annotations

import ast
import sys

from refactor import Replace
from refactor.actions import InsertBefore
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


# ---------------------------------------------------------------------------
# PEP 695 opt-in rules (--enable=pep695-types / --enable=pep695-generics)
# These produce syntax that REQUIRES Python 3.12+.
# ---------------------------------------------------------------------------


class PEP695TypeAliasRule(Rule):
    """Replace ``X: TypeAlias = Y`` (and ``X: typing.TypeAlias = Y``) with
    the PEP 695 ``type X = Y`` statement.

    Opt-in only: ``--enable=pep695-types``.
    Requires Python 3.12+ at runtime (``ast.TypeAlias`` does not exist earlier).
    """

    def match(self, node: ast.AST) -> Replace | None:
        # Guard: ast.TypeAlias doesn't exist on Python <3.12 — bail out.
        if sys.version_info < (3, 12):
            return None

        assert isinstance(node, ast.AnnAssign)
        assert node.value is not None
        assert node.simple == 1

        annotation = node.annotation
        if isinstance(annotation, ast.Attribute):
            # typing.TypeAlias
            assert isinstance(annotation.value, ast.Name)
            assert annotation.value.id == "typing"
            assert annotation.attr == "TypeAlias"
        elif isinstance(annotation, ast.Name):
            assert annotation.id == "TypeAlias"
        else:
            return None

        assert isinstance(node.target, ast.Name)
        alias_name = node.target.id

        new_node = ast.TypeAlias(  # type: ignore[attr-defined]  # 3.12+
            name=ast.Name(id=alias_name, ctx=ast.Store()),
            type_params=[],
            value=node.value,
        )
        ast.fix_missing_locations(new_node)
        return Replace(node, new_node)


def _module_typevar_assignments(tree: ast.AST) -> dict[str, ast.Assign]:
    """Return a mapping of name → Assign for simple ``T = TypeVar('T')``
    declarations at module top level (no kwargs, no bounds)."""
    result: dict[str, ast.Assign] = {}
    if not isinstance(tree, ast.Module):
        return result
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        call = stmt.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        is_typevar = (isinstance(func, ast.Name) and func.id == "TypeVar") or (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "typing"
            and func.attr == "TypeVar"
        )
        if not is_typevar:
            continue
        # Must have exactly one positional arg (the name string), no kwargs.
        if len(call.args) != 1 or call.keywords:
            continue
        result[target.id] = stmt
    return result


def _classes_using_typevar(tree: ast.AST, typevar_name: str) -> list[ast.ClassDef]:
    """Return all ClassDef nodes at module level whose bases include
    ``Generic[T]`` (bare or ``typing.Generic[T]``) for the given typevar."""
    classes: list[ast.ClassDef] = []
    if not isinstance(tree, ast.Module):
        return classes
    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef):
            continue
        for base in stmt.bases:
            if _is_generic_subscript(base, typevar_name):
                classes.append(stmt)
                break
    return classes


def _is_generic_subscript(node: ast.expr, typevar_name: str) -> bool:
    """Return True if *node* is ``Generic[T]`` or ``typing.Generic[T]``."""
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    is_generic = (isinstance(value, ast.Name) and value.id == "Generic") or (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "typing"
        and value.attr == "Generic"
    )
    if not is_generic:
        return False
    slc = node.slice
    return isinstance(slc, ast.Name) and slc.id == typevar_name


class PEP695GenericClassRule(Rule):
    """Transform ``class Foo(Generic[T]):`` to ``class Foo[T]:`` using PEP 695
    type-parameter syntax.

    Conservative scope — only fires when ALL of:

    * ``T = TypeVar('T')`` exists at module top-level with no kwargs.
    * ``T`` appears in Generic bases of EXACTLY one class.
    * The class has ``Generic[T]`` as a base (bare or ``typing.Generic[T]``).

    Opt-in only: ``--enable=pep695-generics``.
    Requires Python 3.12+ at runtime (``ast.TypeVar`` node doesn't exist earlier).

    NOTE: The original ``T = TypeVar('T')`` assignment is left in place (orphaned).
    Removing it is deferred to a Phase 4 cleanup pass.
    """

    def match(self, node: ast.AST) -> Replace | None:
        # Guard: ast.TypeVar (AST node) doesn't exist on Python <3.12.
        if sys.version_info < (3, 12):
            return None

        assert isinstance(node, ast.ClassDef)

        # Find which typevar name is used in Generic[T] bases.
        typevar_name: str | None = None
        for base in node.bases:
            if not isinstance(base, ast.Subscript):
                continue
            slc = base.slice
            if not isinstance(slc, ast.Name):
                continue
            value = base.value
            is_generic = (isinstance(value, ast.Name) and value.id == "Generic") or (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "typing"
                and value.attr == "Generic"
            )
            if is_generic:
                typevar_name = slc.id
                break

        assert typevar_name is not None

        tree = self.context.tree

        # Verify T = TypeVar('T') exists at module level with no kwargs.
        module_tvars = _module_typevar_assignments(tree)
        assert typevar_name in module_tvars

        # Verify T is used in EXACTLY one class at module level.
        classes_using = _classes_using_typevar(tree, typevar_name)
        assert len(classes_using) == 1

        # Build new bases without Generic[T].
        new_bases = [b for b in node.bases if not _is_generic_subscript(b, typevar_name)]

        # Build PEP 695 TypeVar node.
        tp_node = ast.TypeVar(name=typevar_name, bound=None)  # type: ignore[attr-defined]  # 3.12+

        new_class = clone(node)
        new_class.type_params = [tp_node]  # type: ignore[attr-defined]
        new_class.bases = new_bases
        ast.fix_missing_locations(new_class)
        return Replace(node, new_class)


def _has_future_annotations(tree: ast.AST) -> bool:
    """Return True if the module already imports `annotations` from `__future__`."""
    if not isinstance(tree, ast.Module):
        return False
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            if any(alias.name == "annotations" for alias in stmt.names):
                return True
    return False


def _iter_annotations(tree: ast.Module):
    """Yield every annotation expression node in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            yield node.annotation
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.returns is not None
        ):
            yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _contains_pep604_union(annotation: ast.AST) -> bool:
    """Return True if an annotation subtree contains a PEP 604 (X | Y) union."""
    return any(
        isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr)
        for sub in ast.walk(annotation)
    )


def _module_uses_pep604_union(tree: ast.Module) -> bool:
    return any(_contains_pep604_union(ann) for ann in _iter_annotations(tree))


def _future_anchor_index(body: list[ast.stmt]) -> int:
    """Index of the statement the future import must precede (after a docstring)."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return 1
    return 0


class EnsureFutureAnnotationsImportRule(Rule):
    """Insert ``from __future__ import annotations`` when modernized annotations
    use PEP 604 unions (``X | None``) but the module lacks the future import.

    TypingOptionalRule rewrites ``Optional[X]`` to ``X | None``. Without the
    future import, that annotation is evaluated eagerly at definition time, which
    raises ``TypeError`` for forward references (``Optional["Foo"]`` becomes
    ``"Foo" | None``). Importing ``annotations`` makes annotations lazy strings,
    fixing the whole class of runtime breakage from emitted unions.
    """

    def match(self, node: ast.AST) -> InsertBefore | None:
        assert isinstance(node, ast.stmt)

        tree = self.context.tree
        assert isinstance(tree, ast.Module)

        body = tree.body
        anchor_index = _future_anchor_index(body)
        assert anchor_index < len(body)
        # Fire once, anchored on the statement that should follow the import.
        assert node is body[anchor_index]

        assert not _has_future_annotations(tree)
        assert _module_uses_pep604_union(tree)

        future_import = ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
        return InsertBefore(node, target=future_import)
