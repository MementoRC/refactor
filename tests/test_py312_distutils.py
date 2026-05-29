from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.distutils import (
    DistutilsCommandRule,
    DistutilsCoreSetupRule,
    DistutilsLogRule,
    DistutilsSpawnRule,
    DistutilsSysconfigRule,
    DistutilsUtilStrtoboolRule,
    DistutilsVersionRule,
)


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


# ---------------------------------------------------------------------------
# DistutilsCoreSetupRule
# ---------------------------------------------------------------------------


def test_distutils_core_setup_replaced():
    source = """\
        from distutils.core import setup
        setup(name="foo")
        """
    expected = """\
        from setuptools import setup
        setup(name="foo")
        """
    assert _run(DistutilsCoreSetupRule, source=source) == textwrap.dedent(expected)


def test_distutils_core_setup_multiple_names():
    source = """\
        from distutils.core import setup, Extension
        """
    expected = """\
        from setuptools import setup, Extension
        """
    assert _run(DistutilsCoreSetupRule, source=source) == textwrap.dedent(expected)


def test_distutils_core_setup_noop_other_module():
    source = """\
        from distutils.util import strtobool
        """
    result = _run(DistutilsCoreSetupRule, source=source)
    assert result == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# DistutilsCommandRule
# ---------------------------------------------------------------------------


def test_distutils_command_replaced():
    source = """\
        from distutils.command.build import build
        """
    expected = """\
        from setuptools.command.build import build
        """
    assert _run(DistutilsCommandRule, source=source) == textwrap.dedent(expected)


def test_distutils_command_install_replaced():
    source = """\
        from distutils.command.install import install
        """
    expected = """\
        from setuptools.command.install import install
        """
    assert _run(DistutilsCommandRule, source=source) == textwrap.dedent(expected)


def test_distutils_command_noop_non_command():
    source = """\
        from distutils.core import setup
        """
    result = _run(DistutilsCommandRule, source=source)
    assert result == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# DistutilsVersionRule
# ---------------------------------------------------------------------------


def test_distutils_version_loose_replaced():
    source = """\
        from distutils.version import LooseVersion
        """
    expected = """\
        from packaging.version import Version as LooseVersion
        """
    assert _run(DistutilsVersionRule, source=source) == textwrap.dedent(expected)


def test_distutils_version_loose_with_alias():
    source = """\
        from distutils.version import LooseVersion as LV
        """
    expected = """\
        from packaging.version import Version as LV
        """
    assert _run(DistutilsVersionRule, source=source) == textwrap.dedent(expected)


def test_distutils_version_noop_other_module():
    source = """\
        from distutils.core import setup
        """
    result = _run(DistutilsVersionRule, source=source)
    assert result == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# DistutilsSpawnRule
# ---------------------------------------------------------------------------


def test_distutils_spawn_find_executable_replaced():
    source = """\
        from distutils.spawn import find_executable
        """
    expected = """\
        from shutil import which as find_executable
        """
    assert _run(DistutilsSpawnRule, source=source) == textwrap.dedent(expected)


def test_distutils_spawn_find_executable_with_alias():
    source = """\
        from distutils.spawn import find_executable as fe
        """
    expected = """\
        from shutil import which as fe
        """
    assert _run(DistutilsSpawnRule, source=source) == textwrap.dedent(expected)


def test_distutils_spawn_noop_other_name():
    source = """\
        from distutils.spawn import spawn
        """
    result = _run(DistutilsSpawnRule, source=source)
    assert result == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# DistutilsUtilStrtoboolRule
# ---------------------------------------------------------------------------


def test_distutils_util_strtobool_replaced():
    source = """\
        from distutils.util import strtobool
        result = strtobool("yes")
        """
    result = _run(DistutilsUtilStrtoboolRule, source=source)
    assert "def strtobool(val: str) -> int:" in result
    assert "from distutils.util import strtobool" not in result


def test_distutils_util_strtobool_body_correct():
    source = """\
        from distutils.util import strtobool
        """
    result = _run(DistutilsUtilStrtoboolRule, source=source)
    assert "raise ValueError" in result
    assert ("'y', 'yes'" in result) or ("'y'" in result)


def test_distutils_util_strtobool_noop_other_name():
    source = """\
        from distutils.util import byte_compile
        """
    result = _run(DistutilsUtilStrtoboolRule, source=source)
    assert result == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# DistutilsSysconfigRule
# ---------------------------------------------------------------------------


def test_distutils_sysconfig_replaced():
    source = """\
        from distutils.sysconfig import get_python_lib
        """
    expected = """\
        from sysconfig import get_python_lib
        """
    assert _run(DistutilsSysconfigRule, source=source) == textwrap.dedent(expected)


def test_distutils_sysconfig_multiple_names():
    source = """\
        from distutils.sysconfig import get_python_lib, get_config_var
        """
    expected = """\
        from sysconfig import get_python_lib, get_config_var
        """
    assert _run(DistutilsSysconfigRule, source=source) == textwrap.dedent(expected)


def test_distutils_sysconfig_noop_other_module():
    source = """\
        from distutils.core import setup
        """
    result = _run(DistutilsSysconfigRule, source=source)
    assert result == textwrap.dedent(source)


# ---------------------------------------------------------------------------
# DistutilsLogRule
# ---------------------------------------------------------------------------


def test_distutils_log_replaced():
    source = """\
        from distutils import log
        log.info("building")
        """
    result = _run(DistutilsLogRule, source=source)
    assert "import logging" in result
    assert "from distutils import log" not in result


def test_distutils_log_noop_other_name():
    source = """\
        from distutils import core
        """
    result = _run(DistutilsLogRule, source=source)
    assert result == textwrap.dedent(source)


def test_distutils_log_noop_distutils_core_module():
    source = """\
        from distutils.core import setup
        """
    result = _run(DistutilsLogRule, source=source)
    assert result == textwrap.dedent(source)
