from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.stdlib_removed import RemovedStdlibImportRule


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


def test_import_asynchat():
    """Test that importing asynchat (removed module) gets a marker prepended."""
    source = """\
        import asynchat
        """
    result = _run(RemovedStdlibImportRule, source=source)
    assert "TODO(py312): 'asynchat' is removed in Python 3.13" in result
    assert "import asynchat" in result


def test_from_smtpd_import():
    """Test that 'from smtpd import ...' gets a marker prepended."""
    source = """\
        from smtpd import DebuggingServer
        """
    result = _run(RemovedStdlibImportRule, source=source)
    assert "TODO(py312): 'smtpd' is removed in Python 3.13" in result
    assert "from smtpd import DebuggingServer" in result


def test_import_os_not_removed():
    """Test that importing 'os' (not removed) produces no marker."""
    source = """\
        import os
        """
    result = _run(RemovedStdlibImportRule, source=source)
    assert "TODO(py312)" not in result
    assert result == textwrap.dedent(source)


def test_idempotency():
    """Test that running the rule twice does not duplicate markers."""
    source = """\
        import asynchat
        """
    result1 = _run(RemovedStdlibImportRule, source=source)
    # Run again on the output
    result2 = _run(RemovedStdlibImportRule, source=result1)
    # Should be identical; marker should not be duplicated
    assert result1 == result2
    # Count marker occurrences
    marker_count = result2.count("TODO(py312): 'asynchat'")
    assert marker_count == 1, f"Expected 1 marker, got {marker_count}"


def test_multiple_removed_imports():
    """Test that multiple removed imports each get their own marker."""
    source = """\
        import asynchat
        import smtpd
        from aifc import open
        """
    result = _run(RemovedStdlibImportRule, source=source)
    assert "TODO(py312): 'asynchat' is removed in Python 3.13" in result
    assert "TODO(py312): 'smtpd' is removed in Python 3.13" in result
    assert "TODO(py312): 'aifc' is removed in Python 3.13" in result
    assert "import asynchat" in result
    assert "import smtpd" in result
    assert "from aifc import open" in result
