"""Tests for datetime modernization rules (py312_migration.datetime_modern)."""

import textwrap

from refactor import Session
from refactor.rules.py312_migration.datetime_modern import (
    DatetimeUtcfromtimestampRule,
    DatetimeUtcnowRule,
)


def _run(*rules, source: str) -> str:
    """Run refactor Session with given rules on source code."""
    return Session(list(rules)).run(textwrap.dedent(source))


class TestDatetimeUtcnowRule:
    """Tests for DatetimeUtcnowRule."""

    def test_simple_datetime_utcnow(self):
        """Test datetime.utcnow() -> datetime.now(datetime.UTC)."""
        source = """\
        import datetime
        dt = datetime.utcnow()
        """
        expected = """\
        import datetime
        dt = datetime.now(datetime.UTC)
        """
        result = _run(DatetimeUtcnowRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_qualified_datetime_datetime_utcnow(self):
        """Test datetime.datetime.utcnow() -> datetime.datetime.now(datetime.datetime.UTC)."""
        source = """\
        import datetime
        dt = datetime.datetime.utcnow()
        """
        expected = """\
        import datetime
        dt = datetime.datetime.now(datetime.datetime.UTC)
        """
        result = _run(DatetimeUtcnowRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_imported_datetime_class_utcnow(self):
        """Test with 'from datetime import datetime; datetime.utcnow()'."""
        source = """\
        from datetime import datetime
        dt = datetime.utcnow()
        """
        expected = """\
        from datetime import datetime
        dt = datetime.now(datetime.UTC)
        """
        result = _run(DatetimeUtcnowRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_arbitrary_object_utcnow_matches(self):
        """Test that X.utcnow() matches even when X is not datetime (documents behavior)."""
        source = """\
        foo = SomeObject()
        result = foo.utcnow()
        """
        expected = """\
        foo = SomeObject()
        result = foo.now(foo.UTC)
        """
        result = _run(DatetimeUtcnowRule, source=source)
        assert result == textwrap.dedent(expected)


class TestDatetimeUtcfromtimestampRule:
    """Tests for DatetimeUtcfromtimestampRule."""

    def test_simple_datetime_utcfromtimestamp(self):
        """Test datetime.utcfromtimestamp(ts) -> datetime.fromtimestamp(ts, tz=datetime.UTC)."""
        source = """\
        import datetime
        dt = datetime.utcfromtimestamp(1234567890)
        """
        expected = """\
        import datetime
        dt = datetime.fromtimestamp(1234567890, tz=datetime.UTC)
        """
        result = _run(DatetimeUtcfromtimestampRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_qualified_datetime_datetime_utcfromtimestamp(self):
        """Test datetime.datetime.utcfromtimestamp(ts) -> datetime.datetime.fromtimestamp(ts, tz=datetime.datetime.UTC)."""
        source = """\
        import datetime
        dt = datetime.datetime.utcfromtimestamp(1234567890)
        """
        expected = """\
        import datetime
        dt = datetime.datetime.fromtimestamp(1234567890, tz=datetime.datetime.UTC)
        """
        result = _run(DatetimeUtcfromtimestampRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_with_existing_keyword_arg(self):
        """Test that existing keyword args are preserved."""
        source = """\
        import datetime
        dt = datetime.utcfromtimestamp(ts, x=1)
        """
        expected = """\
        import datetime
        dt = datetime.fromtimestamp(ts, x=1, tz=datetime.UTC)
        """
        result = _run(DatetimeUtcfromtimestampRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_idempotency(self):
        """Test that already-converted code does not re-fire the rule."""
        source = """\
        import datetime
        dt = datetime.fromtimestamp(1234567890, tz=datetime.UTC)
        """
        result = _run(DatetimeUtcfromtimestampRule, source=source)
        # Should be unchanged (rule only matches utcfromtimestamp, not fromtimestamp)
        assert result == textwrap.dedent(source)
