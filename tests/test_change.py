from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from refactor import common
from refactor.change import Change


def test_change_compute_diff(tmp_path):
    file_info = common._FileInfo(path=Path(tmp_path / "test.py"))
    change = Change(
        file_info,
        textwrap.dedent(
            """
        if (
            something
            and something_else
            and something_else
        ):
            ...

        def unchanged():
            return 2 + 2
        """
        ),
        textwrap.dedent(
            """
        if (
            something
            or something_else
            or something_different
        ):
            ...

        def unchanged():
            return 2 + 2
        """
        ),
    )

    assert change.compute_diff().splitlines() == [
        f"--- {os.fspath(file_info.path)}",
        f"+++ {os.fspath(file_info.path)}",
        "@@ -1,8 +1,8 @@",
        " ",
        " if (",
        "     something",
        "-    and something_else",
        "-    and something_else",
        "+    or something_else",
        "+    or something_different",
        " ):",
        "     ...",
        " ",
    ]


def test_change_apply_diff(tmp_path):
    target = tmp_path / "apply_diff.py"
    target.write_text("x = 1\n", encoding="utf-8")
    file_info = common._FileInfo(path=target)
    change = Change(file_info, "x = 1\n", "x = 2\n")
    change.apply_diff()
    assert target.read_text(encoding="utf-8") == "x = 2\n"


def test_change_file_property(tmp_path):
    target = tmp_path / "file_prop.py"
    file_info = common._FileInfo(path=target)
    change = Change(file_info, "a = 1\n", "a = 1\n")
    assert change.file == target


def test_change_no_path_raises():
    file_info = common._FileInfo(path=None)
    with pytest.raises(ValueError, match="Can't apply a change to a string"):
        Change(file_info, "a = 1\n", "a = 2\n")


def test_change_file_no_path_raises(tmp_path):
    # Bypass __post_init__ by constructing with a valid path then replacing path
    target = tmp_path / "bypass.py"
    file_info = common._FileInfo(path=target)
    change = Change(file_info, "a = 1\n", "a = 1\n")
    # Now break the path to simulate the .file guard
    change.file_info = common._FileInfo(path=None)
    with pytest.raises(ValueError, match="Change expects a valid file"):
        _ = change.file


def test_change_compute_diff_no_changes(tmp_path):
    target = tmp_path / "no_changes.py"
    file_info = common._FileInfo(path=target)
    source = "x = 1\ny = 2\n"
    change = Change(file_info, source, source)
    assert change.compute_diff() == ""


def test_change_apply_diff_encoding(tmp_path):
    target = tmp_path / "encoded.py"
    content = "# café\nx = 1\n"
    target.write_text(content, encoding="utf-8")
    file_info = common._FileInfo(path=target, encoding="utf-8")
    refactored = "# café\nx = 2\n"
    change = Change(file_info, content, refactored)
    change.apply_diff()
    assert target.read_bytes() == refactored.encode("utf-8")
