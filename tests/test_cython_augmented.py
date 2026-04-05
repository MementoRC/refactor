from __future__ import annotations

import textwrap

import pytest

from refactor import Session
from refactor.rules.cython_augmented import (
    ALL_RULES,
    AddCythonImport,
    AddCythonMarkerDecorator,
    ConvertFunctionAnnotations,
    ConvertTypeToCython,
    DeclareClassAttributes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(rule_cls, source: str) -> str:
    return Session([rule_cls]).run(textwrap.dedent(source))


def unchanged(rule_cls, source: str) -> None:
    dedented = textwrap.dedent(source)
    assert Session([rule_cls]).run(dedented) == dedented


# ---------------------------------------------------------------------------
# Rule 1: ConvertTypeToCython
# ---------------------------------------------------------------------------


def test_convert_int_annotation():
    source = """\
        x: int = 0
    """
    result = run(ConvertTypeToCython, source)
    assert "x: cython.int = 0" in result


def test_convert_float_annotation():
    source = """\
        y: float = 1.0
    """
    result = run(ConvertTypeToCython, source)
    assert "y: cython.double = 1.0" in result


def test_convert_bool_annotation():
    source = """\
        flag: bool = True
    """
    result = run(ConvertTypeToCython, source)
    assert "flag: cython.bint = True" in result


def test_convert_complex_annotation():
    source = """\
        z: complex = 1+2j
    """
    result = run(ConvertTypeToCython, source)
    assert "z: cython.doublecomplex = " in result


def test_skip_complex_annotations():
    """Subscript types like list[int] should not be converted."""
    unchanged(
        ConvertTypeToCython,
        """\
        items: list[int] = []
        """,
    )


def test_skip_already_cython():
    """Already-cython annotations must not be double-converted."""
    unchanged(
        ConvertTypeToCython,
        """\
        x: cython.int = 0
        """,
    )


def test_skip_optional_annotation():
    """Optional[int] is not a bare Name — skip."""
    unchanged(
        ConvertTypeToCython,
        """\
        from typing import Optional
        x: Optional[int] = None
        """,
    )


# ---------------------------------------------------------------------------
# Rule 2: ConvertFunctionAnnotations
# ---------------------------------------------------------------------------


def test_convert_function_params():
    source = """\
        def add(x: int, y: int) -> int:
            return x + y
    """
    result = run(ConvertFunctionAnnotations, source)
    assert "x: cython.int" in result
    assert "y: cython.int" in result


def test_convert_function_return():
    source = """\
        def scale(x: float) -> float:
            return x * 2.0
    """
    result = run(ConvertFunctionAnnotations, source)
    assert "@cython.returns(cython.double)" in result
    assert "@cython.ccall" in result


def test_skip_self_annotation():
    """The ``self`` parameter must not be annotated."""
    source = """\
        class Foo:
            def method(self, x: int) -> int:
                return x
    """
    result = run(ConvertFunctionAnnotations, source)
    # self should not get a cython annotation
    assert "self: cython" not in result
    # but x should be converted
    assert "x: cython.int" in result


def test_skip_function_no_convertible_annotations():
    """A function with no convertible annotations stays unchanged."""
    unchanged(
        ConvertFunctionAnnotations,
        """\
        def process(items: list) -> None:
            pass
        """,
    )


def test_no_double_ccall():
    """A function that already has a cython decorator should not get another @cython.ccall."""
    source = """\
        import cython

        @cython.cfunc
        def fast(x: int) -> int:
            return x
    """
    result = run(ConvertFunctionAnnotations, source)
    assert result.count("@cython.ccall") == 0


# ---------------------------------------------------------------------------
# Rule 3: AddCythonMarkerDecorator
# ---------------------------------------------------------------------------


def test_marker_cfunc():
    source = """\
        # cython: cfunc
        def internal_helper(x: int) -> int:
            return x * 2
    """
    result = run(AddCythonMarkerDecorator, source)
    assert "@cython.cfunc" in result


def test_marker_cclass():
    source = """\
        # cython: cclass
        class FastProcessor:
            pass
    """
    result = run(AddCythonMarkerDecorator, source)
    assert "@cython.cclass" in result


def test_marker_ccall():
    source = """\
        # cython: ccall
        def helper():
            pass
    """
    result = run(AddCythonMarkerDecorator, source)
    assert "@cython.ccall" in result


def test_marker_not_applied_without_comment():
    """Functions without a cython marker comment must not get a decorator."""
    unchanged(
        AddCythonMarkerDecorator,
        """\
        def regular_function():
            pass
        """,
    )


def test_marker_unsupported_value():
    """An unsupported marker value must leave the node unchanged."""
    unchanged(
        AddCythonMarkerDecorator,
        """\
        # cython: unknown_marker
        def helper():
            pass
        """,
    )


# ---------------------------------------------------------------------------
# Rule 4: AddCythonImport
# ---------------------------------------------------------------------------


def test_add_cython_import():
    source = """\
        import os

        x: cython.int = 0
    """
    result = run(AddCythonImport, source)
    assert "import cython" in result


def test_no_import_when_no_cython():
    """No ``import cython`` should be added when there is no cython usage."""
    unchanged(
        AddCythonImport,
        """\
        import os

        x: int = 0
        """,
    )


def test_no_duplicate_import():
    """If ``import cython`` already exists, it must not be added again."""
    unchanged(
        AddCythonImport,
        """\
        import cython
        import os

        x: cython.int = 0
        """,
    )


# ---------------------------------------------------------------------------
# Full conversion integration test
# ---------------------------------------------------------------------------


def test_full_conversion():
    """Realistic before/after transformation using all rules in sequence."""
    source = textwrap.dedent("""\
        import os

        counter: int = 0
        ratio: float = 0.0

        # cython: cfunc
        def compute(x: int, y: float) -> float:
            return x * y
    """)

    # Apply rules in sequence: type annotations first, then function annotations,
    # then marker decorators, then import.
    result = source
    for rule_cls in [
        ConvertTypeToCython,
        ConvertFunctionAnnotations,
        AddCythonMarkerDecorator,
        AddCythonImport,
    ]:
        result = Session([rule_cls]).run(result)

    assert "counter: cython.int = 0" in result
    assert "ratio: cython.double = 0.0" in result
    assert "@cython.cfunc" in result
    assert "x: cython.int" in result
    assert "y: cython.double" in result
    assert "@cython.returns(cython.double)" in result
    assert "import cython" in result


# ---------------------------------------------------------------------------
# Rule 5: DeclareClassAttributes
# ---------------------------------------------------------------------------


def test_declare_class_attributes():
    """Basic case: @cython.cclass with typed __init__ params → class-level declares."""
    source = """\
        import cython

        @cython.cclass
        class Point:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y
    """
    result = run(DeclareClassAttributes, source)
    assert "x = cython.declare(cython.int)" in result
    assert "y = cython.declare(cython.int)" in result


def test_skip_non_cclass():
    """A class without @cython.cclass must not be modified."""
    unchanged(
        DeclareClassAttributes,
        """\
        class Point:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y
        """,
    )


def test_skip_existing_declarations():
    """If cython.declare already exists in the class body, don't add more."""
    unchanged(
        DeclareClassAttributes,
        """\
        import cython

        @cython.cclass
        class Point:
            x = cython.declare(cython.int)
            y = cython.declare(cython.int)
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y
        """,
    )


def test_declare_mixed_types():
    """Class with int, float, and bool typed params → correct cython types."""
    source = """\
        import cython

        @cython.cclass
        class Stats:
            def __init__(self, count: int, ratio: float, active: bool):
                self.count = count
                self.ratio = ratio
                self.active = active
    """
    result = run(DeclareClassAttributes, source)
    assert "count = cython.declare(cython.int)" in result
    assert "ratio = cython.declare(cython.double)" in result
    assert "active = cython.declare(cython.bint)" in result


# ---------------------------------------------------------------------------
# Integration: ALL_RULES
# ---------------------------------------------------------------------------


def test_all_rules_list():
    """ALL_RULES must be a non-empty list of Rule subclasses."""
    from refactor import Rule

    assert isinstance(ALL_RULES, list)
    assert len(ALL_RULES) > 0
    for rule_cls in ALL_RULES:
        assert issubclass(rule_cls, Rule)


def test_full_module_conversion():
    """Full module with all patterns: types, functions, markers, classes."""
    source = textwrap.dedent("""\
        import os

        counter: int = 0
        ratio: float = 0.5

        # cython: cfunc
        def _internal(x: int) -> int:
            return x * 2

        def public_api(x: int, y: float) -> float:
            return _internal(x) * y

        # cython: cclass
        class Accumulator:
            def __init__(self, start: int, factor: float):
                self.total = start
                self.factor = factor

            def add(self, value: int) -> float:
                self.total = self.total + value
                return self.total * self.factor
    """)
    session = Session(ALL_RULES)
    result = session.run(source)

    # Verify import added
    assert "import cython" in result

    # Verify type conversions on module-level variables
    assert "counter: cython.int = 0" in result
    assert "ratio: cython.double = 0.5" in result

    # Verify marker decorators applied
    assert "@cython.cfunc" in result
    assert "@cython.cclass" in result

    # Verify class attributes declared
    assert "cython.declare(" in result

    # Verify function annotations converted
    assert "cython.int" in result
    assert "cython.double" in result
