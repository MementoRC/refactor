from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator
from pathlib import Path

import refactor
from refactor import common
from refactor.context import Context
from refactor.internal.position_provider import infer_identifier_position

SOURCE_DIR = Path(refactor.__file__).parent


def iter_contexts() -> Iterator[Context]:
    for file in SOURCE_DIR.rglob("*.py"):
        source = file.read_text()
        tree = ast.parse(source)
        yield Context(source, tree)


def test_position_provider_for_definitions():
    for context in iter_contexts():
        nodes = [
            node
            for node in ast.walk(context.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        for node in nodes:
            position = infer_identifier_position(node, node.name, context)
            assert position is not None
            known_location = common._get_known_location_from_source(context.source, position)
            assert known_location == node.name


def test_infer_position_simple_function():
    source = textwrap.dedent("""\
        def foo():
            pass
    """)
    tree = ast.parse(source)
    context = Context(source, tree)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    position = infer_identifier_position(node, "foo", context)
    assert position is not None
    assert common._get_known_location_from_source(source, position) == "foo"


def test_infer_position_async_function():
    source = textwrap.dedent("""\
        async def bar():
            pass
    """)
    tree = ast.parse(source)
    context = Context(source, tree)
    node = tree.body[0]
    assert isinstance(node, ast.AsyncFunctionDef)
    position = infer_identifier_position(node, "bar", context)
    assert position is not None
    assert common._get_known_location_from_source(source, position) == "bar"


def test_infer_position_class():
    source = textwrap.dedent("""\
        class MyClass:
            pass
    """)
    tree = ast.parse(source)
    context = Context(source, tree)
    node = tree.body[0]
    assert isinstance(node, ast.ClassDef)
    position = infer_identifier_position(node, "MyClass", context)
    assert position is not None
    assert common._get_known_location_from_source(source, position) == "MyClass"


def test_infer_position_decorated_function():
    source = textwrap.dedent("""\
        @decorator
        def decorated():
            pass
    """)
    tree = ast.parse(source)
    context = Context(source, tree)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    position = infer_identifier_position(node, "decorated", context)
    assert position is not None
    assert common._get_known_location_from_source(source, position) == "decorated"


def test_infer_position_returns_none_for_unsupported():
    source = textwrap.dedent("""\
        x = 1
    """)
    tree = ast.parse(source)
    context = Context(source, tree)
    # ast.Assign is not registered with infer_identifier_position;
    # the base singledispatch returns None (implicitly via ...)
    node = tree.body[0]
    assert isinstance(node, ast.Assign)
    result = infer_identifier_position(node, "x", context)
    assert result is None


def test_infer_position_multiline_decorators():
    source = textwrap.dedent("""\
        @decorator_one(
            arg=True,
        )
        @decorator_two
        def multi_decorated():
            pass
    """)
    tree = ast.parse(source)
    context = Context(source, tree)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    position = infer_identifier_position(node, "multi_decorated", context)
    assert position is not None
    assert common._get_known_location_from_source(source, position) == "multi_decorated"
