from __future__ import annotations

import ast
import copy

from refactor.actions import Replace
from refactor.core import Rule


class FlagDynamicPatterns(Rule):
    """Flag functions containing Rust-incompatible dynamic patterns.

    Adds a ``@_needs_manual_rust_conversion("reason")`` decorator to any
    function definition whose body (or signature) contains patterns that have
    no direct Rust equivalent.
    """

    DYNAMIC_CALLS: dict[str, str] = {
        "getattr": "dynamic attribute access",
        "setattr": "dynamic attribute setting",
        "delattr": "dynamic attribute deletion",
        "exec": "dynamic code execution",
        "eval": "dynamic code evaluation",
        "globals": "runtime introspection",
        "locals": "runtime introspection",
        "__import__": "dynamic import",
    }

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

        # Don't re-flag already-flagged functions
        assert not any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "_needs_manual_rust_conversion"
            for d in node.decorator_list
        )

        reasons: set[str] = set()

        # Check signature-level patterns
        if node.args.kwarg:
            reasons.add("dynamic keyword arguments")
        if node.args.vararg:
            reasons.add("variadic arguments")

        # Walk entire function (including body) for dynamic calls
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in self.DYNAMIC_CALLS
            ):
                reasons.add(self.DYNAMIC_CALLS[child.func.id])

        assert reasons  # nothing flagged → no match

        new_node = copy.deepcopy(node)
        reason_str = ", ".join(sorted(reasons))
        decorator = ast.Call(
            func=ast.Name(id="_needs_manual_rust_conversion", ctx=ast.Load()),
            args=[ast.Constant(value=reason_str)],
            keywords=[],
        )
        new_node.decorator_list.insert(0, decorator)
        return Replace(node, new_node)


def _type_name_for_default(default: ast.expr) -> str | None:
    """Return the Python type name string for a simple literal default, or None."""
    if isinstance(default, ast.Constant):
        # bool MUST be checked before int (bool is a subclass of int)
        if isinstance(default.value, bool):
            return "bool"
        if isinstance(default.value, int):
            return "int"
        if isinstance(default.value, float):
            return "float"
        if isinstance(default.value, str):
            return "str"
        # None or other constants — skip
        return None
    if isinstance(default, ast.List):
        return "list"
    if isinstance(default, ast.Dict):
        return "dict"
    if isinstance(default, ast.Tuple):
        return "tuple"
    if isinstance(default, ast.Set):
        return "set"
    return None


class InferTypeAnnotations(Rule):
    """Add type annotations inferred from default values and add ``-> None``
    return annotations to functions that never return a value.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

        args = node.args
        # Build a mapping: arg index → inferred annotation (for args with defaults)
        # ast stores defaults right-aligned against the positional args list.
        # args.defaults aligns to the END of args.args.
        n_args = len(args.args)
        n_defaults = len(args.defaults)
        offset = n_args - n_defaults  # first arg index that has a default

        new_annotations: dict[int, str] = {}
        for i, default in enumerate(args.defaults):
            arg_idx = offset + i
            arg = args.args[arg_idx]
            # Skip self/cls and already-annotated params
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is not None:
                continue
            type_name = _type_name_for_default(default)
            if type_name is not None:
                new_annotations[arg_idx] = type_name

        # Determine whether to add -> None return annotation
        add_none_return = False
        if node.returns is None:
            has_value_return = any(
                isinstance(child, ast.Return) and child.value is not None
                for child in ast.walk(node)
            )
            if not has_value_return:
                add_none_return = True

        # Nothing to do → no match
        assert new_annotations or add_none_return

        new_node = copy.deepcopy(node)

        for arg_idx, type_name in new_annotations.items():
            new_node.args.args[arg_idx].annotation = ast.Name(id=type_name, ctx=ast.Load())

        if add_none_return:
            new_node.returns = ast.Constant(value=None)

        return Replace(node, new_node)


class EnsureAnnotationCompleteness(Rule):
    """Flag functions that have SOME annotations but not ALL.

    Incomplete typing is worse than none for Rust conversion.  Adds a
    ``@_needs_type_annotation('param1', 'param2')`` decorator listing every
    unannotated parameter.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

        # Don't re-flag already-flagged functions
        assert not any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "_needs_type_annotation"
            for d in node.decorator_list
        )

        args = node.args
        # Collect all params (excluding self/cls)
        params = [a for a in args.args if a.arg not in ("self", "cls")]
        # Also consider *args and **kwargs
        if args.vararg:
            params.append(args.vararg)
        if args.kwarg:
            params.append(args.kwarg)

        annotated = [a for a in params if a.annotation is not None]
        unannotated = [a for a in params if a.annotation is None]

        # A return annotation also counts as evidence of partial annotation intent
        has_return_annotation = node.returns is not None

        # Only flag when SOME annotation context is present but params are incomplete
        assert annotated or has_return_annotation  # at least something annotated
        assert unannotated  # at least one param missing annotation

        new_node = copy.deepcopy(node)
        decorator = ast.Call(
            func=ast.Name(id="_needs_type_annotation", ctx=ast.Load()),
            args=[ast.Constant(value=a.arg) for a in unannotated],
            keywords=[],
        )
        new_node.decorator_list.insert(0, decorator)
        return Replace(node, new_node)
