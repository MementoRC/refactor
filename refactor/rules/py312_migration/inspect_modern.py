from __future__ import annotations

import ast

from refactor import Replace
from refactor.actions import InsertBefore
from refactor.common import clone
from refactor.core import Rule


class InspectGetargspecRule(Rule):
    """Replace inspect.getargspec(fn) with inspect.getfullargspec(fn).

    inspect.getargspec was deprecated in Python 3.0 and removed in 3.11.
    The direct replacement is inspect.getfullargspec.

    Example:
        inspect.getargspec(fn)  ->  inspect.getfullargspec(fn)
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Call)
        assert isinstance(node.func, ast.Attribute)
        assert isinstance(node.func.value, ast.Name)
        assert node.func.value.id == "inspect"
        assert node.func.attr == "getargspec"

        new_node = clone(node)
        new_node.func = clone(node.func)
        new_node.func.attr = "getfullargspec"
        return Replace(node, new_node)


class InspectFormatargspecRule(Rule):
    """Flag inspect.formatargspec calls with a TODO comment.

    inspect.formatargspec has no direct 1:1 replacement (removed in 3.11).
    The recommended approach is to use inspect.signature(fn) and str(sig),
    or iterate parameters manually.

    This rule inserts a TODO comment before the statement containing the call.

    Example:
        result = inspect.formatargspec(args)
        # becomes:
        TODO(py312): inspect.formatargspec removed; use inspect.signature(fn) and str(sig)
        result = inspect.formatargspec(args)
    """

    def match(self, node: ast.AST) -> InsertBefore | None:
        assert isinstance(node, ast.stmt)

        # Walk the statement to find any inspect.formatargspec calls
        found_formatargspec = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        if child.func.value.id == "inspect" and child.func.attr == "formatargspec":
                            found_formatargspec = True
                            break

        assert found_formatargspec

        # Check if the immediately preceding statement is already the TODO marker
        tree = self.context.tree
        assert isinstance(tree, ast.Module)

        # Find the index of this statement in the module body
        stmt_index = None
        for i, stmt in enumerate(tree.body):
            if stmt is node:
                stmt_index = i
                break

        # If this is not the first statement, check if the previous one is our TODO marker
        if stmt_index is not None and stmt_index > 0:
            prev_stmt = tree.body[stmt_index - 1]
            if isinstance(prev_stmt, ast.Expr):
                if isinstance(prev_stmt.value, ast.Constant):
                    if isinstance(prev_stmt.value.value, str):
                        if "TODO(py312): inspect.formatargspec" in prev_stmt.value.value:
                            # Marker already exists, don't add again
                            return None

        # Create the TODO marker as a string constant expression
        marker = ast.Expr(
            value=ast.Constant(
                value="TODO(py312): inspect.formatargspec removed; use inspect.signature(fn) and str(sig)"
            )
        )
        ast.fix_missing_locations(marker)

        return InsertBefore(node, marker)
