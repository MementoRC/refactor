from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator

import pytest

from refactor import Session, common
from refactor.actions import BaseAction, Erase, InsertAfter, InsertBefore, Replace
from refactor.core import MaybeOverlappingActions, Rule


def test_chained_replace_and_insert_after():
    """Rule returns an iterator of Replace + InsertAfter. Verify both are applied."""

    class RenameAndInsert(Rule):
        def match(self, node) -> Iterator[BaseAction]:
            assert isinstance(node, ast.FunctionDef)
            assert node.name == "target"

            new_node = common.clone(node)
            new_node.name = "renamed"
            yield Replace(node, new_node)

            new_stmt = ast.parse("x = 1").body[0]
            yield InsertAfter(node, new_stmt)

    source = textwrap.dedent("""\
        def target():
            pass
        """)
    expected = textwrap.dedent("""\
        def renamed():
            pass
        x = 1
        """)
    result = Session([RenameAndInsert]).run(source)
    assert result == expected


def test_chained_multiple_inserts():
    """Rule returns a rename + multiple InsertAfter actions. Verify all are applied."""

    class RenameAndInsertMultiple(Rule):
        def match(self, node) -> Iterator[BaseAction]:
            assert isinstance(node, ast.FunctionDef)
            assert node.name == "anchor"

            # Rename first so the rule doesn't re-match on the next pass.
            new_node = common.clone(node)
            new_node.name = "anchor_done"
            yield Replace(node, new_node)
            yield InsertAfter(node, ast.parse("a = 1").body[0])
            yield InsertAfter(node, ast.parse("b = 2").body[0])

    source = textwrap.dedent("""\
        def anchor():
            pass
        """)
    result = Session([RenameAndInsertMultiple]).run(source)
    assert "def anchor_done" in result
    assert "a = 1" in result
    assert "b = 2" in result
    anchor_pos = result.index("def anchor_done")
    assert result.index("a = 1") > anchor_pos
    assert result.index("b = 2") > anchor_pos


def test_chained_erase_and_replace_overlapping_raises():
    """Erase + Replace on siblings in the same body raises MaybeOverlappingActions."""

    class EraseFirstReplaceSecond(Rule):
        def match(self, node) -> Iterator[BaseAction]:
            assert isinstance(node, ast.FunctionDef)
            assert node.name == "container"

            body = node.body
            assert len(body) >= 2
            assert isinstance(body[0], ast.Pass)
            assert isinstance(body[1], ast.Assign)

            yield Erase(body[0])
            yield Replace(body[1], ast.parse("result = 42").body[0])

    source = textwrap.dedent("""\
        def container():
            pass
            x = 1
        """)
    with pytest.raises(MaybeOverlappingActions):
        Session([EraseFirstReplaceSecond]).run(source)


def test_chained_insert_before_and_after():
    """Rule returns a rename + InsertBefore + InsertAfter around the same node."""

    class RenameAndWrap(Rule):
        def match(self, node) -> Iterator[BaseAction]:
            assert isinstance(node, ast.FunctionDef)
            assert node.name == "middle"

            # Rename so the rule doesn't re-match on the next pass.
            new_node = common.clone(node)
            new_node.name = "middle_done"
            yield Replace(node, new_node)
            yield InsertBefore(node, ast.parse("before = True").body[0])
            yield InsertAfter(node, ast.parse("after = True").body[0])

    source = textwrap.dedent("""\
        def middle():
            pass
        """)
    result = Session([RenameAndWrap]).run(source)
    assert "before = True" in result
    assert "after = True" in result
    assert "def middle_done" in result
    middle_pos = result.index("def middle_done")
    assert result.index("before = True") < middle_pos
    assert result.index("after = True") > middle_pos


def test_single_action_not_chained():
    """Rule returning a single action (not iterator) works normally."""

    class SimpleRename(Rule):
        def match(self, node) -> Replace:
            assert isinstance(node, ast.FunctionDef)
            assert node.name == "old_name"

            new_node = common.clone(node)
            new_node.name = "new_name"
            return Replace(node, new_node)

    source = textwrap.dedent("""\
        def old_name():
            pass
        """)
    expected = textwrap.dedent("""\
        def new_name():
            pass
        """)
    result = Session([SimpleRename]).run(source)
    assert result == expected
