"""Tests for enum_modern rules."""

from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.enum_modern import IntEnumRule, StrEnumRule


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


# ---------------------------------------------------------------------------
# StrEnumRule
# ---------------------------------------------------------------------------


def test_str_enum_mixin_replaced():
    source = """\
        from enum import Enum

        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"
        """
    expected = """\
        from enum import Enum

        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"
        """
    assert _run(StrEnumRule, source=source) == textwrap.dedent(expected)


def test_str_enum_qualified_style():
    source = """\
        import enum

        class Color(str, enum.Enum):
            RED = "red"
        """
    expected = """\
        import enum

        class Color(enum.StrEnum):
            RED = "red"
        """
    assert _run(StrEnumRule, source=source) == textwrap.dedent(expected)


def test_str_enum_noop_plain_enum():
    source = """\
        from enum import Enum

        class Color(Enum):
            RED = "red"
        """
    assert _run(StrEnumRule, source=source) == textwrap.dedent(source)


def test_str_enum_noop_plain_str():
    source = """\
        class Color(str):
            pass
        """
    assert _run(StrEnumRule, source=source) == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# IntEnumRule
# ---------------------------------------------------------------------------


def test_int_enum_mixin_replaced():
    source = """\
        from enum import Enum

        class Status(int, Enum):
            OK = 1
            FAIL = 2
        """
    expected = """\
        from enum import Enum

        class Status(IntEnum):
            OK = 1
            FAIL = 2
        """
    assert _run(IntEnumRule, source=source) == textwrap.dedent(expected)


def test_int_enum_qualified_style():
    source = """\
        import enum

        class Status(int, enum.Enum):
            OK = 1
        """
    expected = """\
        import enum

        class Status(enum.IntEnum):
            OK = 1
        """
    assert _run(IntEnumRule, source=source) == textwrap.dedent(expected)


def test_int_enum_already_correct_is_noop():
    source = """\
        from enum import IntEnum

        class Status(IntEnum):
            OK = 1
        """
    assert _run(IntEnumRule, source=source) == textwrap.dedent(source)


def test_both_rules_combined():
    source = """\
        from enum import Enum

        class Color(str, Enum):
            RED = "red"

        class Status(int, Enum):
            OK = 1
        """
    expected = """\
        from enum import Enum

        class Color(StrEnum):
            RED = "red"

        class Status(IntEnum):
            OK = 1
        """
    assert _run(StrEnumRule, IntEnumRule, source=source) == textwrap.dedent(expected)
