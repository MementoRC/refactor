"""Tests for OverrideDecoratorRule."""

from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.override_decorator import OverrideDecoratorRule


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


def test_override_added_to_clearly_overriding_method():
    source = """\
        class Base:
            def foo(self):
                pass

        class Child(Base):
            def foo(self):
                pass
        """
    expected = """\
        class Base:
            def foo(self):
                pass

        class Child(Base):
            @typing.override
            def foo(self):
                pass
        """
    assert _run(OverrideDecoratorRule, source=source) == textwrap.dedent(expected)


def test_noop_new_method_not_in_parent():
    source = """\
        class Base:
            def foo(self):
                pass

        class Child(Base):
            def bar(self):
                pass
        """
    result = _run(OverrideDecoratorRule, source=source)
    assert result == textwrap.dedent(source)


def test_noop_parent_not_in_same_module():
    source = """\
        class Child(ExternalBase):
            def foo(self):
                pass
        """
    result = _run(OverrideDecoratorRule, source=source)
    assert result == textwrap.dedent(source)


def test_noop_already_has_typing_override():
    source = """\
        class Base:
            def foo(self):
                pass

        class Child(Base):
            @typing.override
            def foo(self):
                pass
        """
    result = _run(OverrideDecoratorRule, source=source)
    assert result == textwrap.dedent(source)


def test_noop_already_has_bare_override():
    source = """\
        class Base:
            def foo(self):
                pass

        class Child(Base):
            @override
            def foo(self):
                pass
        """
    result = _run(OverrideDecoratorRule, source=source)
    assert result == textwrap.dedent(source)


def test_async_method_override_added():
    source = """\
        class Base:
            async def fetch(self):
                pass

        class Child(Base):
            async def fetch(self):
                pass
        """
    expected = """\
        class Base:
            async def fetch(self):
                pass

        class Child(Base):
            @typing.override
            async def fetch(self):
                pass
        """
    assert _run(OverrideDecoratorRule, source=source) == textwrap.dedent(expected)
