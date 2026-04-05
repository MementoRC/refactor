from __future__ import annotations

import textwrap
from pathlib import Path

from refactor.__main__ import get_refactors


def test_get_refactors_finds_rules(tmp_path: Path) -> None:
    source = textwrap.dedent("""\
        import ast
        from refactor.core import Rule
        from refactor.actions import ReplacementAction

        class MyRule(Rule):
            def match(self, node):
                raise AssertionError
        """)
    refactor_file = tmp_path / "my_rules.py"
    refactor_file.write_text(source)

    results = list(get_refactors(refactor_file))
    assert len(results) == 1
    assert results[0].__name__ == "MyRule"


def test_get_refactors_skips_private(tmp_path: Path) -> None:
    source = textwrap.dedent("""\
        import ast
        from refactor.core import Rule

        class _PrivateRule(Rule):
            def match(self, node):
                raise AssertionError

        class PublicRule(Rule):
            def match(self, node):
                raise AssertionError
        """)
    refactor_file = tmp_path / "my_rules.py"
    refactor_file.write_text(source)

    results = list(get_refactors(refactor_file))
    assert len(results) == 1
    assert results[0].__name__ == "PublicRule"


def test_get_refactors_skips_non_rules(tmp_path: Path) -> None:
    source = textwrap.dedent("""\
        import ast
        from refactor.core import Rule

        class NotARule:
            pass

        def also_not_a_rule():
            pass

        MY_CONSTANT = 42

        class ActualRule(Rule):
            def match(self, node):
                raise AssertionError
        """)
    refactor_file = tmp_path / "my_rules.py"
    refactor_file.write_text(source)

    results = list(get_refactors(refactor_file))
    assert len(results) == 1
    assert results[0].__name__ == "ActualRule"


def test_get_refactors_skips_builtin_rules(tmp_path: Path) -> None:
    source = textwrap.dedent("""\
        import ast
        from refactor.core import Rule
        """)
    refactor_file = tmp_path / "my_rules.py"
    refactor_file.write_text(source)

    results = list(get_refactors(refactor_file))
    # Rule itself comes from refactor.core, so components[0] == "refactor" -> skipped
    assert results == []
