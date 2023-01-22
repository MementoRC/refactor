from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from typing import ClassVar

import pytest

from refactor import common
from refactor.actions import InsertAfter, LazyReplace, Replace
from refactor.change import Change
from refactor.context import Configuration, Context, Representative
from refactor.core import Rule, Session, RuleCollection

fake_ctx = Context(source="<test>", tree=ast.AST())
test_file = common._FileInfo()

__all__ = ["PlusToMinusRule"]

@pytest.mark.parametrize(
    "source, expected, target_func, replacement",
    [
        (
                "2 + 2 == 4",
                "2 + 2 == 5",
                lambda mod: mod.body[0].value.comparators[0],
                ast.Constant(5),
        ),
        (
                "2       + 2 == 4",
                "2       + 2 == 5",
                lambda mod: mod.body[0].value.comparators[0],
                ast.Constant(5),
        ),
        (
                "2 + 2 == 4 # :)",
                "2 + 2 == 5 # :)",
                lambda mod: mod.body[0].value.comparators[0],
                ast.Constant(5),
        ),
    ],
)
def test_apply_simple(source, expected, target_func, replacement):
    tree = ast.parse(source)
    action = Replace(target_func(tree), replacement)
    assert action.apply(fake_ctx, source) == expected


@pytest.mark.parametrize(
    "source, expected, target_func",
    [
        (
                """
                import x # comments
                print(x.y) # comments here
                def something(x, y):
                    return x + y # comments
            """,
                """
                import x # comments
                import x
                print(x.y) # comments here
                def something(x, y):
                    return x + y # comments
            """,
                lambda mod: mod.body[0],
        )
    ],
)
def test_apply_new_statement(source, expected, target_func):
    source = textwrap.dedent(source)
    expected = textwrap.dedent(expected)

    tree = ast.parse(source)
    original_node = target_func(tree)
    action = InsertAfter(original_node, original_node)
    assert action.apply(fake_ctx, source) == expected


class SimpleAction(LazyReplace):
    def build(self):
        node = self.branch()
        node.op = ast.Sub()
        return node


class PlusToMinusRule(Rule):
    def match(self, node):
        assert isinstance(node, ast.BinOp)
        assert isinstance(node.op, ast.Add)

        return SimpleAction(node)


class MultToMinusRule(Rule):
    def match(self, node):
        assert isinstance(node, ast.BinOp)
        assert isinstance(node.op, ast.Mult)

        return SimpleAction(node)


class DivToMinusRule(Rule):
    def match(self, node):
        assert isinstance(node, ast.BinOp)
        assert isinstance(node.op, ast.Div)

        return SimpleAction(node)


class ModuloToMinusRule(Rule):
    def match(self, node):
        assert isinstance(node, ast.BinOp)
        assert isinstance(node.op, ast.Mod)

        return SimpleAction(node)


class PowToMinusRule(Rule):
    def match(self, node):
        assert isinstance(node, ast.BinOp)
        assert isinstance(node.op, ast.Pow)

        return SimpleAction(node)


class SimpleRepresentative(Representative):
    name = "simple"

    def infer_value(self, node):
        return ast.Constant(42)


class PlaceholderReplacer(Rule):
    context_providers = (SimpleRepresentative,)

    def match(self, node):
        assert isinstance(node, ast.Name)
        assert node.id == "placeholder"

        return Replace(node, self.context["simple"].infer_value(node))


@pytest.mark.parametrize(
    "source, expected, rules",
    [
        ("1+1", "1 - 1", PlusToMinusRule),
        ("print(1 + 1)", "print(1 - 1)", PlusToMinusRule),
        (
                "print(1 + 1, some_other_stuff) and 2 + 2",
                "print(1 - 1, some_other_stuff) and 2 - 2",
                PlusToMinusRule,
        ),
        (
                """
                print(
                    1 +
                    2
                )
            """,
                """
                print(
                    1 - 2
                )
            """,
                PlusToMinusRule,
        ),
        (
                "print(x, y, placeholder, z)",
                "print(x, y, 42, z)",
                PlaceholderReplacer,
        ),
    ]
    + [
        ("1*1", "1*1", PlusToMinusRule),
        (
                "print(no,change,style)",
                "print(no,change,style)",
                PlusToMinusRule,
        ),
    ],
)
def test_session_simple(source, rules, expected):
    if isinstance(rules, type):
        rules = [rules]

    source = textwrap.dedent(source)
    expected = textwrap.dedent(expected)

    session = Session(rules)
    assert session.run(source) == expected


@pytest.mark.parametrize(
    "source, expected, rules",
    [
        ("1+1", "1 - 1", "PlusToMinusRule"),
        ("1+1", "1+1", "Plus_MinusRule"),
    ],
)
def test_session_simple_with_string(source, rules, expected):
    if isinstance(rules, str):
        rules = [rules]

    source = textwrap.dedent(source)
    expected = textwrap.dedent(expected)

    session = Session(rules)
    assert session.run(source) == expected


@dataclass
class CollectPlusToMinusMultToMinusRule(RuleCollection):
    rules = [PlusToMinusRule, MultToMinusRule, ]


class CollectMultToMinusPlusToMinusRule(RuleCollection):
    rules = [MultToMinusRule, PlusToMinusRule]


@pytest.mark.parametrize(
    "source, expected, rules",
    [
        ("1+1*2", "1 - 1 - 2", CollectPlusToMinusMultToMinusRule),
        ("1+1*2", "1 - 1 - 2", CollectMultToMinusPlusToMinusRule),
        ("1*1+2", "1 - 1 - 2", CollectPlusToMinusMultToMinusRule),
        ("1*1+2", "1 - 1 - 2", CollectMultToMinusPlusToMinusRule),
    ]
)
def test_session_collection(source, rules, expected):
    if isinstance(rules, type):
        rules = [rules]

    source = textwrap.dedent(source)
    expected = textwrap.dedent(expected)

    session = Session(rules)
    assert session.run(source) == expected


class CollectPlusToMinusMultToMinusRuleWithString(RuleCollection):
    rules: ClassVar = ["PlusToMinusRule", MultToMinusRule, ]


class CollectPlus_MinusMultToMinusRuleWithString(RuleCollection):
    rules: ClassVar = ["Plus_MinusRule", MultToMinusRule, ]


@pytest.mark.parametrize(
    "source, expected, rules",
    [
        ("1+1*2", "1 - 1 - 2", CollectPlusToMinusMultToMinusRuleWithString),
        ("1+1*2", "1+1 - 2", CollectPlus_MinusMultToMinusRuleWithString),
    ]
)
def test_session_collection_with_string(source, rules, expected):
    if isinstance(rules, type):
        rules = [rules]

    source = textwrap.dedent(source)
    expected = textwrap.dedent(expected)

    session = Session(rules)
    assert session.run(source) == expected


class CollectCollectRule(RuleCollection):
    rules = [DivToMinusRule, CollectPlusToMinusMultToMinusRule, ModuloToMinusRule]


@pytest.mark.parametrize(
    "source, expected, rules",
    [
        ("1+1*2/3%4", "1 - 1 - 2 - 3 - 4", CollectCollectRule),
    ]
)
def test_session_multicollection(source, rules, expected):
    if isinstance(rules, type):
        rules = [rules]

    source = textwrap.dedent(source)
    expected = textwrap.dedent(expected)

    session = Session(rules)
    assert session.run(source) == expected


def test_session_run_file(tmp_path):
    paths = set()

    class PathCollector(Rule):
        def match(self, node):
            if self.context.file is not None:
                paths.add(self.context.file)

    session = Session([PlaceholderReplacer, PathCollector])

    file_1 = tmp_path / "test.py"
    file_2 = tmp_path / "test2.py"
    file_3 = tmp_path / "test3.py"

    with open(file_1, "w") as handle:
        handle.write("something + something\n")

    assert session.run_file(file_1) is None
    assert paths == {file_1}

    with open(file_2, "w") as handle:
        handle.write("2 + placeholder + 3")

    change = session.run_file(file_2)
    assert change is not None
    assert isinstance(change, Change)
    assert change.original_source == "2 + placeholder + 3"
    assert change.refactored_source == "2 + 42 + 3"
    assert paths == {file_1, file_2}

    with open(file_3, "w") as handle:
        handle.write("syntax? error?")

    assert session.run_file(file_3) is None
    assert paths == {file_1, file_2}


def test_session_run_file_collection(tmp_path):
    paths = set()

    init = tmp_path / "__init__.py"
    with open(init, "w") as handle:
        handle.write("\n")

    file = tmp_path / "collect_string_collection.py"
    with open(file, "w") as handle:
        handle.write("from refactor.core import RuleCollection\n"
                     "class CollectStringCollection(RuleCollection):\n"
                     "    rules = [\"TestThis\", \"TestThat\"]\n")

    file = tmp_path / "test_this.py"
    with open(file, "w") as handle:
        handle.write("import ast\n"
                     "from refactor.core import Rule\n"
                     "from refactor.actions import LazyReplace\n"
                     "from .test_that import TestThat\n"
                     "class SimpleAction(LazyReplace):\n"
                     "    def build(self):\n"
                     "        node = self.branch()\n"
                     "        node.op = ast.Sub()\n"
                     "        return node\n"
                     "class TestThis(Rule):\n"
                     "    def match(self, node):\n"
                     "        assert isinstance(node, ast.BinOp)\n"
                     "        assert isinstance(node.op, ast.Add)\n"
                     "        return SimpleAction(node)\n")

    file = tmp_path / "test_that.py"
    with open(file, "w") as handle:
        handle.write("import ast\n"
                     "from refactor.core import Rule\n"
                     "from refactor.actions import LazyReplace\n"
                     "class SimpleAction(LazyReplace):\n"
                     "    def build(self):\n"
                     "        node = self.branch()\n"
                     "        node.op = ast.Sub()\n"
                     "        return node\n"
                     "class TestThat(Rule):\n"
                     "    def match(self, node):\n"
                     "        assert isinstance(node, ast.BinOp)\n"
                     "        assert isinstance(node.op, ast.Mult)\n"
                     "        return SimpleAction(node)\n")

    test_file = tmp_path / "test.py"
    with open(test_file, "w") as handle:
        handle.write("1+1*1")

    module = SourceFileLoader("collect_string_collection", str(tmp_path) + "/collect_string_collection.py").load_module()
    session = Session([module.CollectStringCollection])

    change = session.run_file(test_file)
    assert change is not None
    assert isinstance(change, Change)
    assert change.original_source == "1+1*1"
    assert change.refactored_source == "1 - 1 - 1"


class MirrorAction(LazyReplace):
    def build(self):
        return self.node


class RecursiveRule(Rule):
    def match(self, node):
        return MirrorAction(node)


def test_session_run_deterministic():
    session = Session([RecursiveRule])

    # Using _run here to see the 'changed' flag in action.
    refactored_source, changed = session._run("2 + 2 + 3 + 4", file_info=test_file)
    assert not changed
    assert refactored_source == "2 + 2 + 3 + 4"


class InvalidRule(Rule):
    def match(self, node):
        assert isinstance(node, ast.Name)
        return Replace(node, ast.Name("!id"))


def test_session_run_invalid_code():
    session = Session([InvalidRule])
    assert session.run("!id = 1") == "!id = 1"


def test_session_run_invalid_generated_code():
    session = Session([InvalidRule])
    with pytest.raises(ValueError, match=r"Generated source is unparsable\.$"):
        session.run("z = 1")


INVALID_GENERATED_SOURCE_CODE = re.compile(
    r"Generated source is unparsable\.\n"
    r"See (?P<file_name>.*) for the generated source."
)


def test_session_run_invalid_generated_code_debug_mode():
    session = Session([InvalidRule], config=Configuration(debug_mode=True))
    with pytest.raises(
            ValueError,
            match=INVALID_GENERATED_SOURCE_CODE,
    ) as exc_info:
        session.run("z = 1")

    match = INVALID_GENERATED_SOURCE_CODE.match(str(exc_info.value))
    assert match is not None
    with open(match["file_name"]) as handle:
        assert handle.read() == "!id = 1"


class ChangeSign(Rule):
    def match(self, node):
        assert isinstance(node, ast.BinOp)

        new_node = common.clone(node)
        if isinstance(node.op, ast.Add):
            new_node.op = ast.Sub()
        elif isinstance(node.op, ast.Sub):
            new_node.op = ast.Add()
        else:
            return None

        return Replace(node, new_node)


def test_session_run_deterministic_for_on_off_rules():
    session = Session([ChangeSign])
    assert session.run("2 + 2") == "2 - 2"
