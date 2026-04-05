from __future__ import annotations

import ast
import textwrap

from refactor.actions import InsertAfter, Replace, _Rename
from refactor.context import Context
from refactor.internal.action_optimizer import optimize, rename_optimizer


def _make_context(source: str) -> Context:
    tree = ast.parse(source)
    return Context(source=source, tree=tree)


# ---------------------------------------------------------------------------
# rename_optimizer – should fire and return _Rename
# ---------------------------------------------------------------------------


def test_rename_optimizer_function():
    source = textwrap.dedent("""\
        def old_name():
            pass
    """)
    tree = ast.parse(source)
    original_node = tree.body[0]  # FunctionDef old_name

    new_source = textwrap.dedent("""\
        def new_name():
            pass
    """)
    new_tree = ast.parse(new_source)
    target_node = new_tree.body[0]  # FunctionDef new_name

    action = Replace(original_node, target_node)
    context = _make_context(source)

    result = rename_optimizer(action, context)

    assert isinstance(result, _Rename)
    assert result.node is original_node
    assert result.target is target_node


def test_rename_optimizer_class():
    source = textwrap.dedent("""\
        class OldClass:
            pass
    """)
    tree = ast.parse(source)
    original_node = tree.body[0]  # ClassDef OldClass

    new_source = textwrap.dedent("""\
        class NewClass:
            pass
    """)
    new_tree = ast.parse(new_source)
    target_node = new_tree.body[0]  # ClassDef NewClass

    action = Replace(original_node, target_node)
    context = _make_context(source)

    result = rename_optimizer(action, context)

    assert isinstance(result, _Rename)
    assert result.node is original_node
    assert result.target is target_node


def test_rename_optimizer_async_function():
    source = textwrap.dedent("""\
        async def old_handler():
            pass
    """)
    tree = ast.parse(source)
    original_node = tree.body[0]  # AsyncFunctionDef old_handler

    new_source = textwrap.dedent("""\
        async def new_handler():
            pass
    """)
    new_tree = ast.parse(new_source)
    target_node = new_tree.body[0]  # AsyncFunctionDef new_handler

    action = Replace(original_node, target_node)
    context = _make_context(source)

    result = rename_optimizer(action, context)

    assert isinstance(result, _Rename)
    assert result.node is original_node
    assert result.target is target_node


# ---------------------------------------------------------------------------
# optimize – non-Replace actions pass through unchanged
# ---------------------------------------------------------------------------


def test_optimize_passthrough_non_replace():
    source = textwrap.dedent("""\
        x = 1
        y = 2
    """)
    tree = ast.parse(source)
    node = tree.body[0]
    context = _make_context(source)

    action = InsertAfter(node, node)
    result = optimize(action, context)

    assert result is action


# ---------------------------------------------------------------------------
# optimize – Replace on non-named nodes passes through unchanged
# ---------------------------------------------------------------------------


def test_optimize_passthrough_non_named_node():
    source = "x = 1"
    tree = ast.parse(source)
    # ast.Constant is not a named node (not FunctionDef/AsyncFunctionDef/ClassDef)
    original_node = tree.body[0].value  # Constant(1)
    target_node = ast.Constant(2)

    action = Replace(original_node, target_node)
    context = _make_context(source)

    result = optimize(action, context)

    assert result is action


# ---------------------------------------------------------------------------
# optimize – Replace that changes body (not just name) passes through
# ---------------------------------------------------------------------------


def test_optimize_passthrough_body_change():
    source = textwrap.dedent("""\
        def old_name():
            pass
    """)
    tree = ast.parse(source)
    original_node = tree.body[0]  # FunctionDef old_name

    # Same name, different body – not a pure rename
    new_source = textwrap.dedent("""\
        def old_name():
            return 42
    """)
    new_tree = ast.parse(new_source)
    target_node = new_tree.body[0]

    action = Replace(original_node, target_node)
    context = _make_context(source)

    result = optimize(action, context)

    # Same name means rename_optimizer's assertion `node.name != target.name`
    # will fail, so it returns the original action unchanged.
    assert result is action


# ---------------------------------------------------------------------------
# optimize – Replace with multiple field changes passes through unchanged
# ---------------------------------------------------------------------------


def test_optimize_multiple_changes():
    source = textwrap.dedent("""\
        def old_name(x, y):
            return x + y
    """)
    tree = ast.parse(source)
    original_node = tree.body[0]  # FunctionDef old_name(x, y)

    # Different name AND different arguments – more than one field changed
    new_source = textwrap.dedent("""\
        def new_name(a, b, c):
            return a + b + c
    """)
    new_tree = ast.parse(new_source)
    target_node = new_tree.body[0]

    action = Replace(original_node, target_node)
    context = _make_context(source)

    # rename_optimizer requires exactly 1 change (the name field).
    # Multiple field changes make expect_changes assert and bail out,
    # so optimize returns the original action.
    result = optimize(action, context)

    assert result is action
