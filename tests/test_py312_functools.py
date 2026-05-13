"""Tests for LruCacheToCacheRule."""

from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.functools_modern import LruCacheToCacheRule


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


def test_lru_cache_no_args_to_cache():
    source = """\
        import functools

        @functools.lru_cache()
        def expensive(x):
            return x * 2
        """
    expected = """\
        import functools

        @functools.cache
        def expensive(x):
            return x * 2
        """
    assert _run(LruCacheToCacheRule, source=source) == textwrap.dedent(expected)


def test_lru_cache_maxsize_none_to_cache():
    source = """\
        import functools

        @functools.lru_cache(maxsize=None)
        def expensive(x):
            return x * 2
        """
    expected = """\
        import functools

        @functools.cache
        def expensive(x):
            return x * 2
        """
    assert _run(LruCacheToCacheRule, source=source) == textwrap.dedent(expected)


def test_lru_cache_explicit_maxsize_no_transform():
    """@functools.lru_cache(maxsize=128) has different semantics — must not transform."""
    source = """\
        import functools

        @functools.lru_cache(maxsize=128)
        def expensive(x):
            return x * 2
        """
    assert _run(LruCacheToCacheRule, source=source) == textwrap.dedent(source)


def test_lru_cache_typed_no_transform():
    """@functools.lru_cache(typed=True) has different equality semantics — must not transform."""
    source = """\
        import functools

        @functools.lru_cache(typed=True)
        def expensive(x):
            return x * 2
        """
    assert _run(LruCacheToCacheRule, source=source) == textwrap.dedent(source)


def test_bare_lru_cache_no_transform():
    """Bare @lru_cache() form is out of scope for Phase 2 — must not transform."""
    source = """\
        from functools import lru_cache

        @lru_cache()
        def expensive(x):
            return x * 2
        """
    assert _run(LruCacheToCacheRule, source=source) == textwrap.dedent(source)


def test_functools_cache_idempotent():
    """@functools.cache must not be modified — the rule should not re-fire."""
    source = """\
        import functools

        @functools.cache
        def expensive(x):
            return x * 2
        """
    assert _run(LruCacheToCacheRule, source=source) == textwrap.dedent(source)
