from __future__ import annotations

import ast
from copy import deepcopy
from typing import Iterator

from refactor import Rule
from refactor.actions import InsertAfter, Replace

__all__ = [
    "ConvertTypeToCython",
    "ConvertFunctionAnnotations",
    "AddCythonMarkerDecorator",
    "DeclareClassAttributes",
    "AddCythonImport",
    "ALL_RULES",
]

# Mapping from Python built-in type names to cython attribute names
_TYPE_MAP: dict[str, str] = {
    "int": "int",
    "float": "double",
    "complex": "doublecomplex",
    "bool": "bint",
}

_SUPPORTED_MARKERS = frozenset({"cfunc", "ccall", "cclass", "nogil", "inline"})


def _make_cython_attr(attr: str) -> ast.Attribute:
    """Return an ast.Attribute node for ``cython.<attr>``."""
    return ast.Attribute(
        value=ast.Name(id="cython", ctx=ast.Load()),
        attr=attr,
        ctx=ast.Load(),
    )


def _is_bare_type_name(node: ast.expr) -> bool:
    """Return True if *node* is a simple Name whose id is in _TYPE_MAP."""
    return isinstance(node, ast.Name) and node.id in _TYPE_MAP


def _is_already_cython(node: ast.expr) -> bool:
    """Return True if *node* is already a ``cython.<something>`` attribute."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "cython"
    )


def _has_cython_decorator(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function already has any cython.* decorator."""
    for dec in func_node.decorator_list:
        if _is_already_cython(dec):
            return True
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and isinstance(dec.func.value, ast.Name)
            and dec.func.value.id == "cython"
        ):
            return True
    return False


def _inside_cython_compiled_guard(node: ast.AST, source: str) -> bool:
    """Heuristic: check if node is inside an ``if not cython.compiled:`` block.

    We do this by checking the source text of the lines around the node.
    A full ancestry walk would be more robust but is heavier; this simple
    heuristic covers the common case.
    """
    # We can't easily check ancestry here without the context providers,
    # so we rely on the source lines instead.
    lines = source.splitlines()
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return False
    # Walk backwards from the node's line looking for the guard
    for i in range(lineno - 2, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("if not cython.compiled") or stripped.startswith(
            "if not cython.compiled:"
        ):
            return True
        # Stop at blank lines or top-level defs/classes to avoid false positives
        if stripped == "" and i < lineno - 10:
            break
    return False


# ---------------------------------------------------------------------------
# Rule 1: ConvertTypeToCython
# ---------------------------------------------------------------------------


class ConvertTypeToCython(Rule):
    """Convert bare Python type annotations in variable declarations to
    their cython equivalents (e.g. ``x: int = 0`` → ``x: cython.int = 0``).
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.AnnAssign)
        assert _is_bare_type_name(node.annotation)
        assert not _is_already_cython(node.annotation)
        assert not _inside_cython_compiled_guard(node, self.context.source)

        cython_attr = _TYPE_MAP[node.annotation.id]  # type: ignore[union-attr]
        new_node = deepcopy(node)
        new_node.annotation = _make_cython_attr(cython_attr)
        return Replace(node, new_node)


# ---------------------------------------------------------------------------
# Rule 2: ConvertFunctionAnnotations
# ---------------------------------------------------------------------------


class ConvertFunctionAnnotations(Rule):
    """Convert function parameter and return type annotations that use plain
    Python types to cython equivalents, and add ``@cython.ccall`` /
    ``@cython.returns(...)`` decorators as appropriate.
    """

    def match(
        self, node: ast.AST
    ) -> Replace | None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        assert not _inside_cython_compiled_guard(node, self.context.source)

        args = node.args

        # Collect parameter annotations that need conversion (skip self/cls and */**args)
        params_to_convert: list[ast.arg] = []
        for arg in args.args + args.posonlyargs + args.kwonlyargs:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is not None and _is_bare_type_name(arg.annotation):
                params_to_convert.append(arg)

        # Check return annotation
        return_convertible = (
            node.returns is not None and _is_bare_type_name(node.returns)
        )

        # Only proceed if there is something to convert
        assert params_to_convert or return_convertible

        # Don't add @cython.ccall if function already has a cython decorator
        add_ccall = not _has_cython_decorator(node)

        new_node = deepcopy(node)

        # Replace parameter annotations
        def _patch_args(arg_list: list[ast.arg]) -> None:
            for arg in arg_list:
                if arg.arg in ("self", "cls"):
                    continue
                if arg.annotation is not None and _is_bare_type_name(arg.annotation):
                    arg.annotation = _make_cython_attr(_TYPE_MAP[arg.annotation.id])

        _patch_args(new_node.args.args)
        _patch_args(new_node.args.posonlyargs)
        _patch_args(new_node.args.kwonlyargs)

        # Replace return annotation
        if return_convertible:
            cython_return_attr = _TYPE_MAP[new_node.returns.id]  # type: ignore[union-attr]
            new_node.returns = _make_cython_attr(cython_return_attr)

        # Build new decorator list
        new_decorators: list[ast.expr] = []

        if add_ccall:
            new_decorators.append(_make_cython_attr("ccall"))

        if return_convertible:
            cython_return_attr = _TYPE_MAP[node.returns.id]  # type: ignore[union-attr]
            new_decorators.append(
                ast.Call(
                    func=_make_cython_attr("returns"),
                    args=[_make_cython_attr(cython_return_attr)],
                    keywords=[],
                )
            )

        new_node.decorator_list = new_decorators + list(new_node.decorator_list)

        return Replace(node, new_node)


# ---------------------------------------------------------------------------
# Rule 3: AddCythonMarkerDecorator
# ---------------------------------------------------------------------------


class AddCythonMarkerDecorator(Rule):
    """Convert ``# cython: <marker>`` comments immediately preceding a
    function, async function, or class definition into ``@cython.<marker>``
    decorators.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

        lines = self.context.source.splitlines()

        # The node's lineno is 1-based; account for decorator_list shifting
        # the actual def/class keyword line.
        if node.decorator_list:
            def_lineno = node.decorator_list[0].lineno
        else:
            def_lineno = node.lineno

        # The line immediately before the first decorator / def keyword
        preceding_lineno = def_lineno - 2  # 0-based index
        assert preceding_lineno >= 0

        preceding_line = lines[preceding_lineno].strip()
        assert preceding_line.startswith("# cython:")

        marker_value = preceding_line[len("# cython:"):].strip()
        assert marker_value in _SUPPORTED_MARKERS

        # Idempotency guard: if the first existing decorator is already
        # cython.<marker_value>, don't add it again.
        if node.decorator_list:
            first = node.decorator_list[0]
            if (
                isinstance(first, ast.Attribute)
                and isinstance(first.value, ast.Name)
                and first.value.id == "cython"
                and first.attr == marker_value
            ):
                return None

        new_node = deepcopy(node)
        new_node.decorator_list = [_make_cython_attr(marker_value)] + list(
            new_node.decorator_list
        )
        return Replace(node, new_node)


# ---------------------------------------------------------------------------
# Rule 4: AddCythonImport
# ---------------------------------------------------------------------------


class AddCythonImport(Rule):
    """If the module uses any ``cython.*`` references but doesn't yet
    ``import cython``, insert ``import cython`` after the last existing
    import statement.

    Strategy: match on each top-level import statement and fire only on the
    *last* one, so we have a node with source positions (ast.Module itself
    is skipped by the Session's fixed-point walker because it has no line
    number attributes).
    """

    def match(self, node: ast.AST) -> InsertAfter | None:
        assert isinstance(node, (ast.Import, ast.ImportFrom))

        tree = self.context.tree

        # Check whether ``import cython`` already exists — if so, nothing to do.
        for stmt in ast.walk(tree):
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name == "cython" and alias.asname is None:
                        return None
            if isinstance(stmt, ast.ImportFrom):
                if stmt.module == "cython":
                    return None

        # Check whether any cython.* attribute is used anywhere in the tree.
        has_cython_usage = any(
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "cython"
            for n in ast.walk(tree)
        )
        assert has_cython_usage

        # Only fire on the *last* top-level import statement so we insert
        # ``import cython`` exactly once, right after all existing imports.
        last_import: ast.stmt | None = None
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                last_import = stmt

        assert last_import is node, "Not the last import statement"

        import_node = ast.Import(names=[ast.alias(name="cython")])
        return InsertAfter(node, import_node)


# ---------------------------------------------------------------------------
# Rule 5: DeclareClassAttributes
# ---------------------------------------------------------------------------


def _has_cclass_decorator(class_node: ast.ClassDef) -> bool:
    """Return True if the class has a ``@cython.cclass`` decorator."""
    for dec in class_node.decorator_list:
        if (
            isinstance(dec, ast.Attribute)
            and isinstance(dec.value, ast.Name)
            and dec.value.id == "cython"
            and dec.attr == "cclass"
        ):
            return True
    return False


def _has_existing_declare(class_node: ast.ClassDef) -> bool:
    """Return True if any class-level statement is a ``cython.declare(...)`` call."""
    for stmt in class_node.body:
        # Look for: attr = cython.declare(...)
        if isinstance(stmt, ast.Assign):
            if (
                isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and isinstance(stmt.value.func.value, ast.Name)
                and stmt.value.func.value.id == "cython"
                and stmt.value.func.attr == "declare"
            ):
                return True
    return False


def _find_init_method(
    class_node: ast.ClassDef,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the ``__init__`` method of a class, or None."""
    for stmt in class_node.body:
        if (
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name == "__init__"
        ):
            return stmt
    return None


def _param_type_map(
    init_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    """Build a mapping from param name → cython type string for typed params."""
    result: dict[str, str] = {}
    for arg in init_node.args.args + init_node.args.posonlyargs + init_node.args.kwonlyargs:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is not None and _is_bare_type_name(arg.annotation):
            result[arg.arg] = _TYPE_MAP[arg.annotation.id]  # type: ignore[union-attr]
    return result


def _collect_self_assignments(
    init_node: ast.FunctionDef | ast.AsyncFunctionDef,
    param_types: dict[str, str],
) -> list[tuple[str, str]]:
    """Return list of (attr_name, cython_type) for ``self.x = param`` assignments
    where *param* has a known type from *param_types*.
    """
    seen: dict[str, str] = {}
    for stmt in ast.walk(init_node):
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        attr_name = target.attr
        # The RHS must be a simple Name that maps to a typed param
        if isinstance(stmt.value, ast.Name) and stmt.value.id in param_types:
            if attr_name not in seen:
                seen[attr_name] = param_types[stmt.value.id]
    return list(seen.items())


def _make_declare_assign(attr_name: str, cython_type: str) -> ast.Assign:
    """Return ``attr_name = cython.declare(cython.<cython_type>)``."""
    return ast.Assign(
        targets=[ast.Name(id=attr_name, ctx=ast.Store())],
        value=ast.Call(
            func=_make_cython_attr("declare"),
            args=[_make_cython_attr(cython_type)],
            keywords=[],
        ),
        lineno=0,
        col_offset=0,
    )


class DeclareClassAttributes(Rule):
    """For classes decorated with ``@cython.cclass``, scan ``__init__`` for
    ``self.attr = param`` assignments where *param* has a type annotation, and
    insert class-level ``attr = cython.declare(cython.type)`` declarations
    before ``__init__``.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ClassDef)
        assert _has_cclass_decorator(node)
        assert not _has_existing_declare(node)

        init_node = _find_init_method(node)
        assert init_node is not None

        param_types = _param_type_map(init_node)
        assert param_types  # must have at least one typed param

        assignments = _collect_self_assignments(init_node, param_types)
        assert assignments  # must have at least one self.attr = typed_param

        new_node = deepcopy(node)

        # Build declare statements to insert before __init__
        declare_stmts: list[ast.stmt] = [
            _make_declare_assign(attr_name, cython_type)
            for attr_name, cython_type in assignments
        ]

        # Find the index of __init__ in the new class body and insert before it
        new_body: list[ast.stmt] = []
        for stmt in new_node.body:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name == "__init__"
            ):
                new_body.extend(declare_stmts)
            new_body.append(stmt)

        new_node.body = new_body
        return Replace(node, new_node)


# ---------------------------------------------------------------------------
# All rules — ordered for correct fixed-point application
# ---------------------------------------------------------------------------

ALL_RULES = [
    ConvertTypeToCython,
    ConvertFunctionAnnotations,
    AddCythonMarkerDecorator,
    DeclareClassAttributes,
    AddCythonImport,
]

if __name__ == "__main__":
    import refactor

    refactor.run(ALL_RULES)
