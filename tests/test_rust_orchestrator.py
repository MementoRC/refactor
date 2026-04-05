from __future__ import annotations

import textwrap

import pytest

from refactor.rules.rust_orchestrator import (
    ModuleAnalysis,
    analyze_module,
    generate_python_wrapper,
)


# ===========================================================================
# ModuleAnalyzer tests
# ===========================================================================


def test_analyze_simple_module():
    """Fully typed module with no dynamic patterns → 'ready'."""
    source = textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            return a + b

        def multiply(x: float, y: float) -> float:
            return x * y
    """)
    result = analyze_module(source, "math_ops")
    assert isinstance(result, ModuleAnalysis)
    assert result.module_name == "math_ops"
    assert result.total_functions == 2
    assert result.dynamic_pattern_count == 0
    assert result.conversion_readiness == "ready"
    assert result.type_coverage == 1.0


def test_analyze_untyped_module():
    """Module with no type annotations → 'needs_types'."""
    source = textwrap.dedent("""\
        def add(a, b):
            return a + b

        def greet(name):
            return "Hello " + name
    """)
    result = analyze_module(source)
    assert result.conversion_readiness == "needs_types"
    assert result.type_coverage < 0.9
    assert result.dynamic_pattern_count == 0


def test_analyze_dynamic_module():
    """Module using getattr/eval → 'needs_review' (1–3 dynamic patterns)."""
    source = textwrap.dedent("""\
        def get_value(obj: object, name: str) -> object:
            return getattr(obj, name)

        def compute(expr: str) -> object:
            return eval(expr)
    """)
    result = analyze_module(source)
    assert result.dynamic_pattern_count > 0
    assert result.dynamic_pattern_count <= 3
    assert result.conversion_readiness == "needs_review"


def test_analyze_dynamic_module_complex():
    """Module with many dynamic patterns → 'complex'."""
    source = textwrap.dedent("""\
        def messy(obj: object, name: str, val: object, code: str) -> None:
            getattr(obj, name)
            setattr(obj, name, val)
            delattr(obj, name)
            eval(code)
            exec(code)
    """)
    result = analyze_module(source)
    assert result.dynamic_pattern_count > 3
    assert result.conversion_readiness == "complex"


def test_analyze_type_coverage():
    """Verify percentage calculation for partial typing."""
    source = textwrap.dedent("""\
        def half_typed(a: int, b) -> int:
            return a
    """)
    result = analyze_module(source)
    # params: a (annotated), b (not) → 2 params; return annotated → 1
    # total slots = 3, annotated = 2 → 2/3
    assert abs(result.type_coverage - 2 / 3) < 1e-9
    assert result.conversion_readiness == "needs_types"


def test_analyze_class():
    """Class with methods is analyzed correctly."""
    source = textwrap.dedent("""\
        class Calculator:
            \"\"\"A simple calculator.\"\"\"

            def __init__(self, precision: int = 2) -> None:
                self.precision = precision

            def add(self, a: float, b: float) -> float:
                return round(a + b, self.precision)
    """)
    result = analyze_module(source, "calc")
    assert result.total_classes == 1
    assert result.total_functions == 0
    cls = result.classes[0]
    assert cls.name == "Calculator"
    assert cls.docstring == "A simple calculator."
    method_names = [m.name for m in cls.methods]
    assert "__init__" in method_names
    assert "add" in method_names


def test_analyze_empty_module():
    """Empty source → all zeros / trivially ready."""
    result = analyze_module("", "empty")
    assert result.total_functions == 0
    assert result.total_classes == 0
    assert result.dynamic_pattern_count == 0
    assert result.imports == []
    # No params → coverage is 1.0 by convention
    assert result.type_coverage == 1.0
    assert result.conversion_readiness == "ready"


def test_analyze_imports_collected():
    """Import statements are captured as strings."""
    source = textwrap.dedent("""\
        import os
        from typing import Optional

        def noop() -> None:
            pass
    """)
    result = analyze_module(source)
    assert any("os" in imp for imp in result.imports)
    assert any("Optional" in imp for imp in result.imports)


def test_analyze_fully_typed_functions_count():
    """fully_typed_functions counts only functions where every param+return is annotated."""
    source = textwrap.dedent("""\
        def full(a: int, b: int) -> int:
            return a + b

        def partial(a: int, b) -> int:
            return a

        def none(a, b):
            return a
    """)
    result = analyze_module(source)
    assert result.total_functions == 3
    assert result.fully_typed_functions == 1


def test_analyze_flagged_functions_count():
    """flagged_functions counts functions with at least one dynamic pattern."""
    source = textwrap.dedent("""\
        def clean(a: int) -> int:
            return a

        def dynamic(obj: object, name: str) -> object:
            return getattr(obj, name)
    """)
    result = analyze_module(source)
    assert result.flagged_functions == 1


def test_analyze_async_function():
    """Async functions are recognized."""
    source = textwrap.dedent("""\
        async def fetch(url: str) -> str:
            return url
    """)
    result = analyze_module(source)
    assert result.total_functions == 1
    fn = result.functions[0]
    assert fn.is_async is True
    assert fn.name == "fetch"


def test_analyze_kwargs_flagged():
    """**kwargs triggers dynamic pattern detection."""
    source = textwrap.dedent("""\
        def variadic(**kwargs) -> None:
            pass
    """)
    result = analyze_module(source)
    assert result.dynamic_pattern_count > 0
    assert result.functions[0].has_dynamic_patterns


# ===========================================================================
# PythonWrapperGenerator tests
# ===========================================================================

_SAMPLE_SOURCE = textwrap.dedent("""\
    def add(a: int, b: int) -> int:
        return a + b

    def multiply(a: float, b: float) -> float:
        return a * b

    class Calculator:
        def __init__(self, precision: int = 2) -> None:
            self.precision = precision
""")


def test_wrapper_basic():
    """Generated wrapper contains try/except block and fallback block."""
    wrapper = generate_python_wrapper(_SAMPLE_SOURCE, "mymod")
    assert "try:" in wrapper
    assert "except ImportError:" in wrapper
    assert "if not _RUST_AVAILABLE:" in wrapper


def test_wrapper_has_rust_check():
    """Generated wrapper contains is_accelerated() helper."""
    wrapper = generate_python_wrapper(_SAMPLE_SOURCE, "mymod")
    assert "def is_accelerated() -> bool:" in wrapper
    assert "return _RUST_AVAILABLE" in wrapper


def test_wrapper_imports_functions():
    """Public function names appear in the import line."""
    wrapper = generate_python_wrapper(_SAMPLE_SOURCE, "mymod")
    import_line = next(line for line in wrapper.splitlines() if "_mymod_rs import" in line)
    assert "add" in import_line
    assert "multiply" in import_line


def test_wrapper_imports_classes():
    """Public class names appear in the import line."""
    wrapper = generate_python_wrapper(_SAMPLE_SOURCE, "mymod")
    import_line = next(line for line in wrapper.splitlines() if "_mymod_rs import" in line)
    assert "Calculator" in import_line


def test_wrapper_skips_private():
    """Functions/classes starting with _ are excluded from the import."""
    source = textwrap.dedent("""\
        def public_fn(x: int) -> int:
            return x

        def _private_fn(x: int) -> int:
            return x

        class _InternalHelper:
            pass
    """)
    wrapper = generate_python_wrapper(source, "mod")
    import_line = next(line for line in wrapper.splitlines() if "_mod_rs import" in line)
    assert "public_fn" in import_line
    assert "_private_fn" not in import_line
    assert "_InternalHelper" not in import_line


def test_wrapper_preserves_fallback():
    """Original source code appears (indented) in the fallback block."""
    source = textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            return a + b
    """)
    wrapper = generate_python_wrapper(source, "mod")
    # The fallback block must contain the function definition
    assert "def add(a: int, b: int) -> int:" in wrapper
    # It must appear after the "if not _RUST_AVAILABLE:" line
    pos_guard = wrapper.index("if not _RUST_AVAILABLE:")
    pos_fn = wrapper.index("def add(a: int, b: int) -> int:")
    assert pos_fn > pos_guard


def test_wrapper_module_name_in_docstring():
    """Module name appears in the wrapper docstring."""
    wrapper = generate_python_wrapper(_SAMPLE_SOURCE, "my_module")
    assert "my_module" in wrapper.splitlines()[0]


def test_wrapper_rust_flag_set_true_on_success():
    """_RUST_AVAILABLE = True is inside the try block."""
    wrapper = generate_python_wrapper(_SAMPLE_SOURCE, "mymod")
    lines = wrapper.splitlines()
    try_idx = next(i for i, l in enumerate(lines) if l.strip() == "try:")
    except_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("except ImportError"))
    try_block = "\n".join(lines[try_idx:except_idx])
    assert "_RUST_AVAILABLE = True" in try_block


def test_wrapper_rust_flag_set_false_on_failure():
    """_RUST_AVAILABLE = False is inside the except block."""
    wrapper = generate_python_wrapper(_SAMPLE_SOURCE, "mymod")
    lines = wrapper.splitlines()
    except_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("except ImportError"))
    # Grab a few lines after except
    except_block = "\n".join(lines[except_idx: except_idx + 5])
    assert "_RUST_AVAILABLE = False" in except_block
