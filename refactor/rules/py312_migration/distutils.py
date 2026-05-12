from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule


class DistutilsCoreSetupRule(Rule):
    """Replace 'from distutils.core import ...' with 'from setuptools import ...'."""

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ImportFrom)
        assert node.module == "distutils.core"
        new_node = clone(node)
        new_node.module = "setuptools"
        return Replace(node, new_node)


class DistutilsCommandRule(Rule):
    """Replace 'from distutils.command.X import ...' with 'from setuptools.command.X import ...'."""

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ImportFrom)
        assert node.module is not None
        assert node.module.startswith("distutils.command.")
        new_node = clone(node)
        new_node.module = "setuptools.command." + node.module[len("distutils.command.") :]
        return Replace(node, new_node)


class DistutilsVersionRule(Rule):
    """Replace 'from distutils.version import LooseVersion/StrictVersion' with 'from packaging.version import Version'.

    Preserves callsite compatibility by keeping the original name as an alias.
    If an 'as' alias is already present, keeps it unchanged on Version.
    """

    _VERSION_NAMES = frozenset({"LooseVersion", "StrictVersion"})

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ImportFrom)
        assert node.module == "distutils.version"
        imported = {alias.name for alias in node.names}
        assert imported & self._VERSION_NAMES
        new_aliases: list[ast.alias] = []
        for alias in node.names:
            if alias.name in self._VERSION_NAMES:
                # Preserve the effective name at the callsite
                effective_name = alias.asname if alias.asname else alias.name
                new_aliases.append(ast.alias(name="Version", asname=effective_name))
            else:
                new_aliases.append(clone(alias))
        new_node = clone(node)
        new_node.module = "packaging.version"
        new_node.names = new_aliases
        return Replace(node, new_node)


class DistutilsSpawnRule(Rule):
    """Replace 'from distutils.spawn import find_executable' with 'from shutil import which as find_executable'."""

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ImportFrom)
        assert node.module == "distutils.spawn"
        spawn_aliases = [alias for alias in node.names if alias.name == "find_executable"]
        assert len(spawn_aliases) > 0
        new_aliases: list[ast.alias] = []
        for alias in node.names:
            if alias.name == "find_executable":
                # Preserve explicit alias if present; otherwise keep find_executable as alias
                effective_name = alias.asname if alias.asname else "find_executable"
                new_aliases.append(ast.alias(name="which", asname=effective_name))
            else:
                new_aliases.append(clone(alias))
        new_node = clone(node)
        new_node.module = "shutil"
        new_node.names = new_aliases
        return Replace(node, new_node)


class DistutilsUtilStrtoboolRule(Rule):
    """Replace 'from distutils.util import strtobool' with an inline strtobool() definition."""

    _STRTOBOOL_SOURCE = '''\
def strtobool(val: str) -> int:
    """Replacement for removed distutils.util.strtobool."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(f"invalid truth value {val!r}")
'''

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ImportFrom)
        assert node.module == "distutils.util"
        strtobool_aliases = [alias for alias in node.names if alias.name == "strtobool"]
        assert len(strtobool_aliases) > 0
        func_def = ast.parse(self._STRTOBOOL_SOURCE).body[0]
        return Replace(node, func_def)


class DistutilsSysconfigRule(Rule):
    """Replace 'from distutils.sysconfig import ...' with 'from sysconfig import ...'."""

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ImportFrom)
        assert node.module == "distutils.sysconfig"
        new_node = clone(node)
        new_node.module = "sysconfig"
        return Replace(node, new_node)


class DistutilsLogRule(Rule):
    """Replace 'from distutils import log' with 'import logging'."""

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ImportFrom)
        assert node.module == "distutils"
        log_aliases = [alias for alias in node.names if alias.name == "log"]
        assert len(log_aliases) > 0
        new_node = ast.Import(names=[ast.alias(name="logging", asname=None)])
        return Replace(node, new_node)
