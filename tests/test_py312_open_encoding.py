"""Tests for OpenEncodingRule."""

from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.open_encoding import OpenEncodingRule


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


def test_open_no_mode_gets_encoding():
    source = """\
        f = open('file.txt')
        """
    expected = """\
        f = open('file.txt', encoding='utf-8')
        """
    assert _run(OpenEncodingRule, source=source) == textwrap.dedent(expected)


def test_open_text_mode_gets_encoding():
    source = """\
        f = open('file.txt', 'r')
        """
    expected = """\
        f = open('file.txt', 'r', encoding='utf-8')
        """
    assert _run(OpenEncodingRule, source=source) == textwrap.dedent(expected)


def test_noop_binary_mode_positional():
    source = """\
        f = open('file.txt', 'rb')
        """
    result = _run(OpenEncodingRule, source=source)
    assert result == textwrap.dedent(source)


def test_noop_binary_mode_keyword():
    source = """\
        f = open('file.txt', mode='wb')
        """
    result = _run(OpenEncodingRule, source=source)
    assert result == textwrap.dedent(source)


def test_noop_already_has_encoding():
    source = """\
        f = open('file.txt', encoding='latin-1')
        """
    result = _run(OpenEncodingRule, source=source)
    assert result == textwrap.dedent(source)
