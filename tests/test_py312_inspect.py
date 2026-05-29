from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.inspect_modern import (
    InspectFormatargspecRule,
    InspectGetargspecRule,
)


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


class TestInspectGetargspecRule:
    """Tests for InspectGetargspecRule: inspect.getargspec -> inspect.getfullargspec."""

    def test_basic_replacement(self) -> None:
        """Test basic inspect.getargspec() call is replaced."""
        source = """\
            import inspect
            result = inspect.getargspec(fn)
        """
        expected = """\
            import inspect
            result = inspect.getfullargspec(fn)
        """
        output = _run(InspectGetargspecRule, source=source)
        assert output == textwrap.dedent(expected)

    def test_with_kwargs(self) -> None:
        """Test that keyword arguments are preserved."""
        source = """\
            import inspect
            result = inspect.getargspec(fn, defaults=None)
        """
        expected = """\
            import inspect
            result = inspect.getfullargspec(fn, defaults=None)
        """
        output = _run(InspectGetargspecRule, source=source)
        assert output == textwrap.dedent(expected)

    def test_no_match_getfullargspec(self) -> None:
        """Test that getfullargspec calls are not re-fired."""
        source = """\
            import inspect
            result = inspect.getfullargspec(fn)
        """
        output = _run(InspectGetargspecRule, source=source)
        assert output == textwrap.dedent(source)


class TestInspectFormatargspecRule:
    """Tests for InspectFormatargspecRule: insert TODO marker before formatargspec."""

    def test_basic_call(self) -> None:
        """Test TODO marker inserted before a bare formatargspec call."""
        source = """\
            import inspect
            result = inspect.formatargspec(args)
        """
        output = _run(InspectFormatargspecRule, source=source)
        lines = output.strip().split("\n")
        # Expect: import, empty line, TODO marker, then the original call
        assert "TODO(py312): inspect.formatargspec" in output
        assert "result = inspect.formatargspec(args)" in output
        # The TODO should appear before the call in the output
        assert output.index("TODO(py312)") < output.index("result = inspect.formatargspec")

    def test_in_assignment(self) -> None:
        """Test TODO marker inserted before an assignment containing formatargspec."""
        source = """\
            import inspect
            result = inspect.formatargspec(args, varargs='args')
        """
        output = _run(InspectFormatargspecRule, source=source)
        assert "TODO(py312): inspect.formatargspec" in output
        assert "result = inspect.formatargspec(args, varargs='args')" in output
        # The TODO should appear before the call
        assert output.index("TODO(py312)") < output.index("result = inspect.formatargspec")

    def test_idempotency(self) -> None:
        """Test that running twice produces the same output (no duplicate markers)."""
        source = """\
            import inspect
            result = inspect.formatargspec(args)
        """
        # Run once
        output1 = _run(InspectFormatargspecRule, source=source)
        # Run again on the output
        output2 = _run(InspectFormatargspecRule, source=output1)
        # Count occurrences of the TODO marker
        count1 = output1.count("TODO(py312): inspect.formatargspec")
        count2 = output2.count("TODO(py312): inspect.formatargspec")
        # Should have exactly one marker in both cases
        assert count1 == 1
        assert count2 == 1
