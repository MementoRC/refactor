"""Tests for PairwiseRule and BatchedRule (Phase 3, opt-in via --enable=itertools-modern)."""

from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.stdlib_additions import BatchedRule, PairwiseRule


def _run(*rules, source: str) -> str:
    """Run refactor rules on source code."""
    return Session(list(rules)).run(textwrap.dedent(source))


class TestPairwiseRule:
    """Test PairwiseRule: zip(seq, seq[1:]) -> itertools.pairwise(seq)."""

    def test_simple_seq_name(self):
        """zip(seq, seq[1:]) with name 'seq' is replaced."""
        source = """\
            import itertools

            result = zip(seq, seq[1:])
        """
        expected = """\
            import itertools

            result = itertools.pairwise(seq)
        """
        result = _run(PairwiseRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_different_variable_name(self):
        """zip(items, items[1:]) with name 'items' is replaced."""
        source = """\
            import itertools

            for a, b in zip(items, items[1:]):
                pass
        """
        expected = """\
            import itertools

            for a, b in itertools.pairwise(items):
                pass
        """
        result = _run(PairwiseRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_noop_different_names(self):
        """zip(a, b) with different names is not transformed."""
        source = """\
            import itertools

            result = zip(a, b)
        """
        result = _run(PairwiseRule, source=source)
        assert result == textwrap.dedent(source)

    def test_noop_wrong_slice_lower(self):
        """zip(seq, seq[2:]) with slice lower=2 is not transformed."""
        source = """\
            import itertools

            result = zip(seq, seq[2:])
        """
        result = _run(PairwiseRule, source=source)
        assert result == textwrap.dedent(source)


class TestBatchedRule:
    """Test BatchedRule: [items[i:i+n] for i in range(0, len(items), n)] -> list(itertools.batched(items, n))."""

    def test_canonical_chunking_pattern(self):
        """Exact canonical chunking list comprehension is replaced."""
        source = """\
            import itertools

            chunks = [items[i:i+n] for i in range(0, len(items), n)]
        """
        expected = """\
            import itertools

            chunks = list(itertools.batched(items, n))
        """
        result = _run(BatchedRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_noop_range_missing_start(self):
        """[items[i:i+n] for i in range(len(items))] (2-arg range) is not transformed."""
        source = """\
            import itertools

            chunks = [items[i:i+n] for i in range(len(items))]
        """
        result = _run(BatchedRule, source=source)
        assert result == textwrap.dedent(source)

    def test_noop_plain_list_comp(self):
        """An unrelated list comprehension is not transformed."""
        source = """\
            result = [x * 2 for x in range(10)]
        """
        result = _run(BatchedRule, source=source)
        assert result == textwrap.dedent(source)
