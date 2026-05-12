from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule


def _resolve_datetime_chain(node: ast.expr) -> list[str] | None:
    """
    Resolve a chain of attribute accesses to check if it represents datetime or datetime.datetime.

    Returns a list of identifiers in the chain (e.g., ['datetime'] or ['datetime', 'datetime']),
    or None if the chain cannot be resolved to datetime.

    Examples:
        Name('datetime')                             -> ['datetime']
        Attribute(Name('datetime'), 'datetime')      -> ['datetime', 'datetime']
        Name('foo')                                  -> ['foo'] (we accept this; caller disambiguates)
    """
    chain = []
    current = node

    while isinstance(current, ast.Attribute):
        chain.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        chain.append(current.id)
        chain.reverse()
        return chain

    return None


class DatetimeUtcnowRule(Rule):
    """Replace datetime.utcnow() with datetime.now(datetime.UTC).

    Handles:
    - datetime.utcnow()              -> datetime.now(datetime.UTC)
    - datetime.datetime.utcnow()     -> datetime.datetime.now(datetime.datetime.UTC)
    - from datetime import datetime; datetime.utcnow()  -> datetime.now(datetime.UTC)

    NOTE: This rule matches ANY X.utcnow() pattern based on AST structure alone, not semantic
    analysis. While utcnow() is almost exclusively a datetime method, false positives are possible
    (e.g., foo.utcnow() where foo is some other object). This is acceptable because:
    1. utcnow is overwhelmingly a datetime method in practice
    2. The replacement pattern is idiomatic for all datetime variations
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Call)
        assert isinstance(node.func, ast.Attribute)
        assert node.func.attr == "utcnow"

        # Extract the object chain (e.g., datetime or datetime.datetime)
        obj_chain = _resolve_datetime_chain(node.func.value)
        assert obj_chain is not None

        # Clone the Call node and build the replacement
        new_node = clone(node)
        new_node.func = clone(new_node.func)

        # Change method name from utcnow to now
        new_node.func.attr = "now"

        # Build the UTC argument: <chain>.UTC
        # Reconstruct the same chain as the object, then append .UTC
        utc_expr: ast.expr = ast.Name(id=obj_chain[0], ctx=ast.Load())
        for attr in obj_chain[1:]:
            utc_expr = ast.Attribute(value=utc_expr, attr=attr, ctx=ast.Load())
        utc_expr = ast.Attribute(value=utc_expr, attr="UTC", ctx=ast.Load())

        # Add UTC as positional argument
        new_node.args = [utc_expr] + list(new_node.args)

        ast.fix_missing_locations(new_node)
        return Replace(node, new_node)


class DatetimeUtcfromtimestampRule(Rule):
    """Replace datetime.utcfromtimestamp(ts) with datetime.fromtimestamp(ts, tz=datetime.UTC).

    Handles:
    - datetime.utcfromtimestamp(ts)              -> datetime.fromtimestamp(ts, tz=datetime.UTC)
    - datetime.datetime.utcfromtimestamp(ts)     -> datetime.datetime.fromtimestamp(ts, tz=datetime.datetime.UTC)

    Preserves existing positional and keyword arguments, inserting tz= after positional args.

    NOTE: Like DatetimeUtcnowRule, this matches any X.utcfromtimestamp() pattern. False positives
    are unlikely but theoretically possible.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Call)
        assert isinstance(node.func, ast.Attribute)
        assert node.func.attr == "utcfromtimestamp"

        # Extract the object chain
        obj_chain = _resolve_datetime_chain(node.func.value)
        assert obj_chain is not None

        # Clone the Call node
        new_node = clone(node)
        new_node.func = clone(new_node.func)

        # Change method name from utcfromtimestamp to fromtimestamp
        new_node.func.attr = "fromtimestamp"

        # Build the UTC argument: <chain>.UTC
        utc_expr: ast.expr = ast.Name(id=obj_chain[0], ctx=ast.Load())
        for attr in obj_chain[1:]:
            utc_expr = ast.Attribute(value=utc_expr, attr=attr, ctx=ast.Load())
        utc_expr = ast.Attribute(value=utc_expr, attr="UTC", ctx=ast.Load())

        # Add tz=<chain>.UTC keyword argument
        tz_keyword = ast.keyword(arg="tz", value=utc_expr)

        # Preserve existing keywords and append tz=
        new_node.keywords = list(new_node.keywords) + [tz_keyword]

        ast.fix_missing_locations(new_node)
        return Replace(node, new_node)
