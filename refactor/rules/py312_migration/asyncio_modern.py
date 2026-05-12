from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule


class AsyncioGetEventLoopRule(Rule):
    """Replace asyncio.get_event_loop() with asyncio.get_running_loop() in async contexts.

    This rule only transforms calls inside async def function bodies. Calls in sync
    functions or at module level are left unchanged.

    Example:
        async def foo():
            loop = asyncio.get_event_loop()  # -> asyncio.get_running_loop()
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Call)

        # Check if this is asyncio.get_event_loop()
        func = node.func
        assert isinstance(func, ast.Attribute)
        assert func.attr == "get_event_loop"
        assert isinstance(func.value, ast.Name)
        assert func.value.id == "asyncio"

        # Walk up the ancestry chain to find the enclosing function
        parent_field, parent_node = self.context.ancestry.infer(node)

        while parent_node is not None:
            if isinstance(parent_node, ast.AsyncFunctionDef):
                # Found an async def — this is a valid context for get_running_loop()
                break
            if isinstance(parent_node, (ast.FunctionDef, ast.Lambda)):
                # Found a sync function or lambda — not a valid context
                assert False
            # Continue walking up
            parent_field, parent_node = self.context.ancestry.infer(parent_node)
        else:
            # Reached module level without finding an async def
            assert False

        # Transform: replace get_event_loop with get_running_loop
        new_node = clone(node)
        new_func = clone(func)
        new_func.attr = "get_running_loop"
        new_node.func = new_func
        return Replace(node, new_node)
