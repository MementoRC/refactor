"""Tests for typing_modern rules."""

from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.typing_modern import (
    TypingDeprecatedAliasRule,
    TypingOptionalRule,
    TypingTypeRule,
)


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


# ---------------------------------------------------------------------------
# TypingDeprecatedAliasRule
# ---------------------------------------------------------------------------


def test_typing_dict_qualified_to_builtin():
    source = """\
        import typing

        def foo() -> typing.Dict[str, int]:
            return {}
        """
    expected = """\
        import typing

        def foo() -> dict[str, int]:
            return {}
        """
    assert _run(TypingDeprecatedAliasRule, source=source) == textwrap.dedent(expected)


def test_typing_dict_bare_to_builtin():
    source = """\
        from typing import Dict

        def foo() -> Dict[str, int]:
            return {}
        """
    expected = """\
        from typing import Dict

        def foo() -> dict[str, int]:
            return {}
        """
    assert _run(TypingDeprecatedAliasRule, source=source) == textwrap.dedent(expected)


def test_typing_list_qualified_to_builtin():
    source = """\
        import typing

        def foo() -> typing.List[int]:
            return []
        """
    expected = """\
        import typing

        def foo() -> list[int]:
            return []
        """
    assert _run(TypingDeprecatedAliasRule, source=source) == textwrap.dedent(expected)


def test_typing_tuple_qualified_to_builtin():
    source = """\
        import typing

        def foo() -> typing.Tuple[int, ...]:
            return (1,)
        """
    expected = """\
        import typing

        def foo() -> tuple[int, ...]:
            return (1,)
        """
    assert _run(TypingDeprecatedAliasRule, source=source) == textwrap.dedent(expected)


def test_typing_dict_nested_in_callable():
    source = """\
        import typing
        from typing import Callable

        def foo(cb: Callable[[typing.Dict[str, int]], None]) -> None:
            pass
        """
    expected = """\
        import typing
        from typing import Callable

        def foo(cb: Callable[[dict[str, int]], None]) -> None:
            pass
        """
    assert _run(TypingDeprecatedAliasRule, source=source) == textwrap.dedent(expected)


def test_builtin_dict_no_transform():
    """dict[str, int] is already the builtin form — rule must not fire."""
    source = """\
        def foo() -> dict[str, int]:
            return {}
        """
    assert _run(TypingDeprecatedAliasRule, source=source) == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# TypingTypeRule
# ---------------------------------------------------------------------------


def test_typing_type_qualified_to_builtin():
    source = """\
        import typing

        def foo(cls: typing.Type[Foo]) -> None:
            pass
        """
    expected = """\
        import typing

        def foo(cls: type[Foo]) -> None:
            pass
        """
    assert _run(TypingTypeRule, source=source) == textwrap.dedent(expected)


def test_typing_type_bare_to_builtin():
    source = """\
        from typing import Type

        def foo(cls: Type[Foo]) -> None:
            pass
        """
    expected = """\
        from typing import Type

        def foo(cls: type[Foo]) -> None:
            pass
        """
    assert _run(TypingTypeRule, source=source) == textwrap.dedent(expected)


def test_type_call_no_transform():
    """type(obj) is a builtin call, not a Type subscript — must not transform."""
    source = """\
        def foo(obj):
            return type(obj)
        """
    assert _run(TypingTypeRule, source=source) == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# TypingOptionalRule
# ---------------------------------------------------------------------------


def test_optional_bare_to_union_none():
    source = """\
        from typing import Optional

        def foo(x: Optional[int]) -> None:
            pass
        """
    expected = """\
        from typing import Optional

        def foo(x: int | None) -> None:
            pass
        """
    assert _run(TypingOptionalRule, source=source) == textwrap.dedent(expected)


def test_optional_qualified_to_union_none():
    source = """\
        import typing

        def foo(x: typing.Optional[str]) -> None:
            pass
        """
    expected = """\
        import typing

        def foo(x: str | None) -> None:
            pass
        """
    assert _run(TypingOptionalRule, source=source) == textwrap.dedent(expected)


def test_optional_existing_bitor_chains_none():
    """Optional[X | Y] should become X | Y | None, not (X | Y) | None."""
    source = """\
        from typing import Optional

        def foo(x: Optional[int | str]) -> None:
            pass
        """
    expected = """\
        from typing import Optional

        def foo(x: int | str | None) -> None:
            pass
        """
    assert _run(TypingOptionalRule, source=source) == textwrap.dedent(expected)
