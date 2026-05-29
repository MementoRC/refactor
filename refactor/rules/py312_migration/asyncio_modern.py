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
                return None
            # Continue walking up
            parent_field, parent_node = self.context.ancestry.infer(parent_node)
        else:
            # Reached module level without finding an async def
            return None

        # Transform: replace get_event_loop with get_running_loop
        new_node = clone(node)
        new_func = clone(func)
        new_func.attr = "get_running_loop"
        new_node.func = new_func
        return Replace(node, new_node)


class AsyncioEnsureFutureRule(Rule):
    """Replace asyncio.ensure_future(coro) with asyncio.create_task(coro) in async contexts.

    This rule only transforms calls inside async def function bodies. Calls in sync
    functions or at module level are left unchanged, since ensure_future also accepts
    Futures (not just coroutines) and the substitution is only safe for async contexts.

    Example:
        async def foo():
            task = asyncio.ensure_future(coro())  # -> asyncio.create_task(coro())
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Call)

        # Check if this is asyncio.ensure_future(...)
        func = node.func
        assert isinstance(func, ast.Attribute)
        assert func.attr == "ensure_future"
        assert isinstance(func.value, ast.Name)
        assert func.value.id == "asyncio"

        # Walk up the ancestry chain to find the enclosing function
        parent_field, parent_node = self.context.ancestry.infer(node)

        while parent_node is not None:
            if isinstance(parent_node, ast.AsyncFunctionDef):
                # Found an async def — safe to substitute create_task
                break
            if isinstance(parent_node, (ast.FunctionDef, ast.Lambda)):
                # Found a sync function or lambda — not safe to transform
                return None
            # Continue walking up
            parent_field, parent_node = self.context.ancestry.infer(parent_node)
        else:
            # Reached module level without finding an async def
            return None

        # Transform: replace ensure_future with create_task
        new_node = clone(node)
        new_func = clone(func)
        new_func.attr = "create_task"
        new_node.func = new_func
        return Replace(node, new_node)


class AsyncioWaitForToTimeoutRule(Rule):
    """Replace ``await asyncio.wait_for(coro, timeout=T)`` with the modern context-manager form.

    Transforms:

        async with asyncio.timeout(T):
            await coro

    **Opt-in only** (``--enable=asyncio-timeout``).  The two APIs have subtly
    different cancellation semantics: ``wait_for`` cancels the inner task on
    timeout whereas ``asyncio.timeout`` propagates a ``TimeoutError`` and
    relies on structured concurrency; callers that catch
    ``asyncio.TimeoutError`` / ``concurrent.futures.TimeoutError`` explicitly
    may need manual adjustment after this transform.

    Conservatively matches ONLY bare expression-statement awaits — i.e. the
    ``await asyncio.wait_for(...)`` must be the entire statement, not the
    right-hand side of an assignment or part of a larger expression.  That
    guarantees the replacement (which produces a block, not an expression) is
    always syntactically valid.
    """

    def match(self, node: ast.AST) -> Replace | None:
        # We match the enclosing ast.Expr statement, not the Call itself,
        # because the replacement is a block statement (ast.AsyncWith) and
        # must replace the whole statement node.
        assert isinstance(node, ast.Expr)

        # The statement value must be an Await expression.
        value = node.value
        assert isinstance(value, ast.Await)

        # The awaited expression must be a Call.
        call = value.value
        assert isinstance(call, ast.Call)

        # The call must be asyncio.wait_for(...)
        func = call.func
        assert isinstance(func, ast.Attribute)
        assert func.attr == "wait_for"
        assert isinstance(func.value, ast.Name)
        assert func.value.id == "asyncio"

        # Resolve positional args and the timeout value.
        # Accepted forms:
        #   wait_for(coro, timeout=T)  -> 1 positional arg, 1 keyword
        #   wait_for(coro, T)          -> 2 positional args, 0 keywords
        args = call.args
        kwargs = call.keywords

        if len(args) == 1 and len(kwargs) == 1 and kwargs[0].arg == "timeout":
            coro_expr = args[0]
            timeout_expr = kwargs[0].value
        elif len(args) == 2 and len(kwargs) == 0:
            coro_expr = args[0]
            timeout_expr = args[1]
        else:
            return None

        # Must be inside an async def — walk ancestry.
        parent_field, parent_node = self.context.ancestry.infer(node)

        while parent_node is not None:
            if isinstance(parent_node, ast.AsyncFunctionDef):
                break
            if isinstance(parent_node, (ast.FunctionDef, ast.Lambda)):
                return None
            parent_field, parent_node = self.context.ancestry.infer(parent_node)
        else:
            return None

        # Build: async with asyncio.timeout(T): await coro
        new_stmt = ast.AsyncWith(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="asyncio", ctx=ast.Load()),
                            attr="timeout",
                            ctx=ast.Load(),
                        ),
                        args=[timeout_expr],
                        keywords=[],
                    ),
                    optional_vars=None,
                )
            ],
            body=[ast.Expr(value=ast.Await(value=coro_expr))],
        )
        ast.fix_missing_locations(new_stmt)
        return Replace(node, new_stmt)
