"""Tests for AsyncioGetEventLoopRule (Phase 1), AsyncioEnsureFutureRule (Phase 2), and AsyncioWaitForToTimeoutRule (Phase 3)."""

from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.py312_migration.asyncio_modern import (
    AsyncioEnsureFutureRule,
    AsyncioGetEventLoopRule,
    AsyncioWaitForToTimeoutRule,
)


def _run(*rules, source: str) -> str:
    """Run refactor rules on source code."""
    return Session(list(rules)).run(textwrap.dedent(source))


class TestAsyncioGetEventLoopRule:
    """Test AsyncioGetEventLoopRule transformations."""

    def test_inside_async_def(self):
        """Test: asyncio.get_event_loop() inside async def should be transformed."""
        source = """\
            import asyncio

            async def foo():
                loop = asyncio.get_event_loop()
                return loop
        """
        expected = """\
            import asyncio

            async def foo():
                loop = asyncio.get_running_loop()
                return loop
        """
        result = _run(AsyncioGetEventLoopRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_inside_sync_def_no_transform(self):
        """Test: asyncio.get_event_loop() inside sync def should NOT be transformed."""
        source = """\
            import asyncio

            def foo():
                loop = asyncio.get_event_loop()
                return loop
        """
        result = _run(AsyncioGetEventLoopRule, source=source)
        assert result == textwrap.dedent(source)

    def test_at_module_level_no_transform(self):
        """Test: asyncio.get_event_loop() at module level should NOT be transformed."""
        source = """\
            import asyncio

            loop = asyncio.get_event_loop()
        """
        result = _run(AsyncioGetEventLoopRule, source=source)
        assert result == textwrap.dedent(source)

    def test_nested_sync_in_async_no_transform(self):
        """Test: nested sync def inside async def — call in sync def should NOT be transformed."""
        source = """\
            import asyncio

            async def outer():
                def inner():
                    loop = asyncio.get_event_loop()
                    return loop
                return inner()
        """
        result = _run(AsyncioGetEventLoopRule, source=source)
        assert result == textwrap.dedent(source)

    def test_nested_async_in_sync_should_transform(self):
        """Test: nested async def inside sync def — call in async def SHOULD be transformed."""
        source = """\
            import asyncio

            def outer():
                async def inner():
                    loop = asyncio.get_event_loop()
                    return loop
                return inner()
        """
        expected = """\
            import asyncio

            def outer():
                async def inner():
                    loop = asyncio.get_running_loop()
                    return loop
                return inner()
        """
        result = _run(AsyncioGetEventLoopRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_get_running_loop_unchanged(self):
        """Test: asyncio.get_running_loop() in async def should NOT be transformed (idempotent)."""
        source = """\
            import asyncio

            async def foo():
                loop = asyncio.get_running_loop()
                return loop
        """
        result = _run(AsyncioGetEventLoopRule, source=source)
        assert result == textwrap.dedent(source)


class TestAsyncioEnsureFutureRule:
    """Test AsyncioEnsureFutureRule transformations."""

    def test_inside_async_def(self):
        """Test: asyncio.ensure_future(coro) inside async def should be transformed."""
        source = """\
            import asyncio

            async def foo():
                task = asyncio.ensure_future(coro())
                return task
        """
        expected = """\
            import asyncio

            async def foo():
                task = asyncio.create_task(coro())
                return task
        """
        result = _run(AsyncioEnsureFutureRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_inside_sync_def_no_transform(self):
        """Test: asyncio.ensure_future(coro) inside sync def should NOT be transformed."""
        source = """\
            import asyncio

            def foo():
                task = asyncio.ensure_future(coro())
                return task
        """
        result = _run(AsyncioEnsureFutureRule, source=source)
        assert result == textwrap.dedent(source)

    def test_at_module_level_no_transform(self):
        """Test: asyncio.ensure_future(coro) at module level should NOT be transformed."""
        source = """\
            import asyncio

            task = asyncio.ensure_future(coro())
        """
        result = _run(AsyncioEnsureFutureRule, source=source)
        assert result == textwrap.dedent(source)

    def test_create_task_unchanged(self):
        """Test: asyncio.create_task(coro) in async def should NOT be re-transformed (idempotent)."""
        source = """\
            import asyncio

            async def foo():
                task = asyncio.create_task(coro())
                return task
        """
        result = _run(AsyncioEnsureFutureRule, source=source)
        assert result == textwrap.dedent(source)

    def test_both_rules_together(self):
        """Test: file with both get_event_loop and ensure_future in async def — both transform."""
        source = """\
            import asyncio

            async def foo():
                loop = asyncio.get_event_loop()
                task = asyncio.ensure_future(coro())
                return loop, task
        """
        expected = """\
            import asyncio

            async def foo():
                loop = asyncio.get_running_loop()
                task = asyncio.create_task(coro())
                return loop, task
        """
        result = _run(AsyncioGetEventLoopRule, AsyncioEnsureFutureRule, source=source)
        assert result == textwrap.dedent(expected)


class TestAsyncioWaitForToTimeoutRule:
    """Test AsyncioWaitForToTimeoutRule transformations (Phase 3, opt-in)."""

    def test_kwarg_timeout_form(self):
        """Test: await asyncio.wait_for(coro, timeout=T) -> async with asyncio.timeout(T): await coro."""
        source = """\
            import asyncio

            async def foo():
                await asyncio.wait_for(coro, timeout=5)
        """
        expected = """\
            import asyncio

            async def foo():
                async with asyncio.timeout(5):
                    await coro
        """
        result = _run(AsyncioWaitForToTimeoutRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_positional_timeout_form(self):
        """Test: await asyncio.wait_for(coro, T) -> async with asyncio.timeout(T): await coro."""
        source = """\
            import asyncio

            async def foo():
                await asyncio.wait_for(coro, 5)
        """
        expected = """\
            import asyncio

            async def foo():
                async with asyncio.timeout(5):
                    await coro
        """
        result = _run(AsyncioWaitForToTimeoutRule, source=source)
        assert result == textwrap.dedent(expected)

    def test_assigned_await_no_transform(self):
        """Test: result = await asyncio.wait_for(coro, timeout=5) should NOT be transformed.

        The await is inside an ast.Assign, not an ast.Expr statement, so the
        rule does not match and the source is returned unchanged.
        """
        source = """\
            import asyncio

            async def foo():
                result = await asyncio.wait_for(coro, timeout=5)
        """
        result = _run(AsyncioWaitForToTimeoutRule, source=source)
        assert result == textwrap.dedent(source)

    def test_inside_sync_def_no_transform(self):
        """Test: wait_for in sync def (no await statement) should NOT be transformed."""
        source = """\
            import asyncio

            def foo():
                result = asyncio.run(asyncio.wait_for(coro, timeout=5))
        """
        result = _run(AsyncioWaitForToTimeoutRule, source=source)
        assert result == textwrap.dedent(source)

    def test_idempotency(self):
        """Test: already-transformed code (async with asyncio.timeout) is not re-fired."""
        source = """\
            import asyncio

            async def foo():
                async with asyncio.timeout(5):
                    await coro
        """
        result = _run(AsyncioWaitForToTimeoutRule, source=source)
        assert result == textwrap.dedent(source)
