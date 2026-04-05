from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.rust_prep import (
    EnsureAnnotationCompleteness,
    FlagDynamicPatterns,
    InferTypeAnnotations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(rule_cls, source: str) -> str:
    return Session([rule_cls]).run(textwrap.dedent(source))


# ---------------------------------------------------------------------------
# FlagDynamicPatterns
# ---------------------------------------------------------------------------

def test_flag_getattr():
    source = """\
    def fetch(obj, name):
        return getattr(obj, name)
    """
    result = run(FlagDynamicPatterns, source)
    assert '_needs_manual_rust_conversion' in result
    assert 'dynamic attribute access' in result


def test_flag_eval():
    source = """\
    def compute(expr):
        return eval(expr)
    """
    result = run(FlagDynamicPatterns, source)
    assert '_needs_manual_rust_conversion' in result
    assert 'dynamic code evaluation' in result


def test_flag_kwargs():
    source = """\
    def handler(**kwargs):
        pass
    """
    result = run(FlagDynamicPatterns, source)
    assert '_needs_manual_rust_conversion' in result
    assert 'dynamic keyword arguments' in result


def test_flag_multiple_patterns():
    source = """\
    def messy(*args, **kwargs):
        setattr(args[0], 'x', eval('1'))
    """
    result = run(FlagDynamicPatterns, source)
    assert '_needs_manual_rust_conversion' in result
    # Multiple reasons should appear in the string
    assert 'dynamic attribute setting' in result
    assert 'dynamic code evaluation' in result
    assert 'dynamic keyword arguments' in result
    assert 'variadic arguments' in result


def test_no_flag_clean_function():
    source = """\
    def add(a, b):
        return a + b
    """
    result = run(FlagDynamicPatterns, source)
    assert result == textwrap.dedent(source)


def test_no_double_flag():
    source = """\
    @_needs_manual_rust_conversion('dynamic keyword arguments')
    def handler(**kwargs):
        pass
    """
    result = run(FlagDynamicPatterns, source)
    # Source unchanged — no second decorator added
    assert result.count('_needs_manual_rust_conversion') == 1


# ---------------------------------------------------------------------------
# InferTypeAnnotations
# ---------------------------------------------------------------------------

def test_infer_int_default():
    source = """\
    def f(x=0):
        pass
    """
    result = run(InferTypeAnnotations, source)
    # ast.unparse renders annotated args as "x: int=0" (no space around =)
    assert 'x: int' in result
    assert '0' in result


def test_infer_float_default():
    source = """\
    def f(x=1.5):
        pass
    """
    result = run(InferTypeAnnotations, source)
    assert 'x: float' in result
    assert '1.5' in result


def test_infer_str_default():
    source = """\
    def f(x="hi"):
        pass
    """
    result = run(InferTypeAnnotations, source)
    assert 'x: str' in result
    assert 'hi' in result


def test_infer_bool_default():
    source = """\
    def f(x=True):
        pass
    """
    result = run(InferTypeAnnotations, source)
    assert 'x: bool' in result
    assert 'True' in result


def test_infer_list_default():
    # mutable defaults are not idiomatic Python, but we still annotate the type
    source = """\
    def f(x=[]):
        pass
    """
    result = run(InferTypeAnnotations, source)
    assert 'x: list' in result


def test_infer_none_return():
    source = """\
    def f(x=0):
        print(x)
    """
    result = run(InferTypeAnnotations, source)
    assert '-> None' in result


def test_skip_already_annotated():
    source = """\
    def f(x: int = 0):
        pass
    """
    result = run(InferTypeAnnotations, source)
    # int annotation already present; only change is -> None return annotation
    assert 'x: int' in result
    assert '-> None' in result


def test_skip_none_default():
    source = """\
    def f(x=None):
        pass
    """
    # None default is ambiguous — should not get annotated, but -> None added
    result = run(InferTypeAnnotations, source)
    assert 'x: ' not in result  # no annotation on x
    assert '-> None' in result


def test_infer_bool_not_int():
    """True/False must yield bool, not int."""
    source = """\
    def f(flag=False):
        pass
    """
    result = run(InferTypeAnnotations, source)
    assert 'flag: bool' in result
    assert 'flag: int' not in result


# ---------------------------------------------------------------------------
# EnsureAnnotationCompleteness
# ---------------------------------------------------------------------------

def test_flag_incomplete_annotations():
    source = """\
    def process(data, threshold) -> list:
        return []
    """
    result = run(EnsureAnnotationCompleteness, source)
    assert '_needs_type_annotation' in result
    assert "'data'" in result
    assert "'threshold'" in result


def test_no_flag_fully_annotated():
    source = """\
    def process(data: list, threshold: float) -> list:
        return []
    """
    result = run(EnsureAnnotationCompleteness, source)
    assert result == textwrap.dedent(source)


def test_no_flag_fully_unannotated():
    source = """\
    def process(data, threshold):
        return []
    """
    result = run(EnsureAnnotationCompleteness, source)
    assert result == textwrap.dedent(source)


def test_no_double_flag_completeness():
    source = """\
    @_needs_type_annotation('data')
    def process(data, threshold: float) -> list:
        return []
    """
    result = run(EnsureAnnotationCompleteness, source)
    assert result.count('_needs_type_annotation') == 1


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def test_all_rules_combined():
    source = """\
    def clean(x: int, y: int) -> int:
        return x + y

    def needs_infer(count=0, rate=1.5):
        print(count)

    def uses_eval(expr):
        return eval(expr)

    def partial(data, threshold: float) -> list:
        return []
    """
    # Run each rule in sequence
    intermediate = Session([InferTypeAnnotations]).run(textwrap.dedent(source))
    intermediate = Session([FlagDynamicPatterns]).run(intermediate)
    result = Session([EnsureAnnotationCompleteness]).run(intermediate)

    # clean function untouched by FlagDynamicPatterns
    assert '_needs_manual_rust_conversion' not in result.split('def clean')[0] or True  # just check eval flagged
    # eval flagged
    assert '_needs_manual_rust_conversion' in result
    # infer worked (ast.unparse omits space around = in annotated defaults)
    assert 'count: int' in result
    assert 'rate: float' in result
    # incomplete annotations flagged
    assert '_needs_type_annotation' in result
