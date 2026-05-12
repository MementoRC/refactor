from __future__ import annotations

import ast

from refactor import Replace
from refactor.actions import InsertBefore
from refactor.common import clone
from refactor.core import Rule

# PEP 594 modules removed in Python 3.12 or 3.13
_REMOVED_MODULES = frozenset(
    {
        "asynchat",
        "asyncore",
        "smtpd",
        "imghdr",
        "sndhdr",
        "audioop",
        "aifc",
        "sunau",
        "chunk",
        "cgi",
        "cgitb",
        "crypt",
        "spwd",
        "nis",
        "pipes",
        "mailcap",
        "nntplib",
        "telnetlib",
        "uu",
        "ossaudiodev",
        "xdrlib",
        "msilib",
    }
)


def _marker_already_exists(tree: ast.Module, node: ast.stmt, module_name: str) -> bool:
    """Check if a marker comment for this module already exists immediately before the import."""
    # Find the index of the current node in the tree body
    try:
        node_idx = tree.body.index(node)
    except ValueError:
        return False

    if node_idx == 0:
        return False

    prev_stmt = tree.body[node_idx - 1]
    # Check if the previous statement is an Expr with a constant string marker
    if isinstance(prev_stmt, ast.Expr) and isinstance(prev_stmt.value, ast.Constant):
        marker_value = prev_stmt.value.value
        if isinstance(marker_value, str) and f"TODO(py312): '{module_name}'" in marker_value:
            return True

    return False


class RemovedStdlibImportRule(Rule):
    """Insert a warning comment before imports of PEP 594 removed stdlib modules.

    When a module from _REMOVED_MODULES is detected via `import X` or `from X import ...`,
    insert a marker expression (string constant) on the line immediately before the import
    to warn the developer that the module has been removed in Python 3.12 or 3.13.

    Example:
        import asynchat
        ->
        "TODO(py312): 'asynchat' is removed in Python 3.13"
        import asynchat
    """

    def match(self, node: ast.AST) -> InsertBefore | None:
        assert isinstance(node, (ast.Import, ast.ImportFrom))

        tree = self.context.tree
        assert isinstance(tree, ast.Module)

        removed_module = None

        if isinstance(node, ast.Import):
            # Check if any of the imported names are in _REMOVED_MODULES
            for alias in node.names:
                if alias.name in _REMOVED_MODULES:
                    removed_module = alias.name
                    break
        elif isinstance(node, ast.ImportFrom):
            # Check if the module being imported from is in _REMOVED_MODULES
            if node.module and node.module in _REMOVED_MODULES:
                removed_module = node.module

        assert removed_module is not None

        # Avoid duplicating markers: skip if one already exists immediately before
        if _marker_already_exists(tree, node, removed_module):
            return None

        # Create a marker expression (string constant) to prepend
        marker_msg = f"TODO(py312): '{removed_module}' is removed in Python 3.13"
        marker_node = ast.Expr(value=ast.Constant(value=marker_msg))
        ast.fix_missing_locations(marker_node)

        return InsertBefore(node, marker_node)
