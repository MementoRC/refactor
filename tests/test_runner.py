from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from refactor.actions import Replace
from refactor.core import Rule, Session
from refactor.runner import (
    _DEFAULT_WORKERS,
    _determine_workers,
    dump_stats,
    expand_paths,
    run_files,
    unbound_main,
)


# ---------------------------------------------------------------------------
# A trivial Rule subclass used by run_files / unbound_main tests
# ---------------------------------------------------------------------------


class RenameFoo(Rule):
    """Replace every use of the name ``foo`` with ``bar``."""

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Name)
        assert node.id == "foo"
        return Replace(node, ast.Name("bar", ast.Load()))


# ---------------------------------------------------------------------------
# expand_paths
# ---------------------------------------------------------------------------


def test_expand_paths_single_file(tmp_path: Path) -> None:
    py_file = tmp_path / "module.py"
    py_file.write_text("x = 1\n")

    result = list(expand_paths(py_file))

    assert result == [py_file]


def test_expand_paths_directory(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    (sub / "c.py").write_text("z = 3\n")
    (tmp_path / "not_python.txt").write_text("ignored\n")

    result = set(expand_paths(tmp_path))

    assert result == {
        tmp_path / "a.py",
        tmp_path / "b.py",
        sub / "c.py",
    }


def test_expand_paths_empty_directory(tmp_path: Path) -> None:
    result = list(expand_paths(tmp_path))

    assert result == []


# ---------------------------------------------------------------------------
# dump_stats
# ---------------------------------------------------------------------------


def test_dump_stats_empty() -> None:
    assert dump_stats({}) == ""


def test_dump_stats_single() -> None:
    result = dump_stats({"reformatted": 1})

    assert result == "1 file reformatted"


def test_dump_stats_plural() -> None:
    result = dump_stats({"reformatted": 3})

    assert result == "3 files reformatted"


def test_dump_stats_multiple_statuses() -> None:
    # Use an ordered dict literal so the output is deterministic.
    result = dump_stats({"reformatted": 2, "left unchanged": 5})

    assert result == "2 files reformatted, 5 files left unchanged"


def test_dump_stats_skips_zero_entries() -> None:
    result = dump_stats({"reformatted": 0, "left unchanged": 4})

    assert result == "4 files left unchanged"


# ---------------------------------------------------------------------------
# _determine_workers
# ---------------------------------------------------------------------------


def test_determine_workers_explicit() -> None:
    assert _determine_workers(4) == 4
    assert _determine_workers(1) == 1


def test_determine_workers_default_debug() -> None:
    # Debug mode must always return 1 to keep execution sequential.
    result = _determine_workers(_DEFAULT_WORKERS, debug_mode=True)

    assert result == 1


def test_determine_workers_invalid() -> None:
    with pytest.raises(ValueError):
        _determine_workers("not-a-number")


# ---------------------------------------------------------------------------
# run_files
# ---------------------------------------------------------------------------


def _write_foo_source(path: Path) -> Path:
    """Write a tiny .py file that contains a ``foo`` name."""
    py_file = path / "sample.py"
    py_file.write_text(
        textwrap.dedent("""\
            foo = 1
            print(foo)
        """)
    )
    return py_file


def test_run_files_no_changes(tmp_path: Path) -> None:
    # A file without "foo" should not be matched by RenameFoo.
    py_file = tmp_path / "clean.py"
    py_file.write_text(
        textwrap.dedent("""\
            bar = 1
            print(bar)
        """)
    )
    session = Session([RenameFoo])

    result = run_files(session, [py_file], apply=False, workers=1)

    assert result == 0


def test_run_files_with_changes_no_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    py_file = _write_foo_source(tmp_path)
    original_text = py_file.read_text()
    session = Session([RenameFoo])

    result = run_files(session, [py_file], apply=False, workers=1)

    # Should report that there is at least one reformatted file.
    assert result == 1
    # The file on disk must NOT be modified.
    assert py_file.read_text() == original_text
    # A diff should have been printed to stdout.
    captured = capsys.readouterr()
    assert captured.out != ""


def test_run_files_with_changes_apply(tmp_path: Path) -> None:
    py_file = _write_foo_source(tmp_path)
    session = Session([RenameFoo])

    result = run_files(session, [py_file], apply=True, workers=1)

    assert result == 1
    new_text = py_file.read_text()
    assert "bar" in new_text
    assert "foo" not in new_text


# ---------------------------------------------------------------------------
# unbound_main
# ---------------------------------------------------------------------------


def test_unbound_main_argv(tmp_path: Path) -> None:
    """The argv parameter must be forwarded to ArgumentParser so that the
    caller can supply an explicit argument list instead of sys.argv."""
    py_file = _write_foo_source(tmp_path)
    session = Session([RenameFoo])

    # Pass argv explicitly — this is the bug-fix behaviour being validated.
    result = unbound_main(session, argv=[str(py_file), "--apply", "--workers", "1"])

    assert result == 1
    new_text = py_file.read_text()
    assert "bar" in new_text
    assert "foo" not in new_text


# ---------------------------------------------------------------------------
# verbose flag
# ---------------------------------------------------------------------------


def test_run_files_verbose_output(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    py_file = _write_foo_source(tmp_path)
    session = Session([RenameFoo])

    result = run_files(session, [py_file], apply=True, workers=1, verbose=True)

    assert result == 1
    captured = capsys.readouterr()
    assert "reformatted" in captured.out
    assert "All done!" in captured.out


def test_run_files_no_verbose_output(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    py_file = _write_foo_source(tmp_path)
    session = Session([RenameFoo])

    result = run_files(session, [py_file], apply=True, workers=1, verbose=False)

    assert result == 1
    captured = capsys.readouterr()
    assert "reformatted" not in captured.out
    assert "All done!" not in captured.out


def test_unbound_main_verbose_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    py_file = _write_foo_source(tmp_path)
    session = Session([RenameFoo])

    result = unbound_main(
        session,
        argv=[str(py_file), "--apply", "--verbose", "--workers", "1"],
    )

    assert result == 1
    captured = capsys.readouterr()
    assert "reformatted" in captured.out
    assert "All done!" in captured.out
