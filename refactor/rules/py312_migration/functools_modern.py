from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule


class LruCacheToCacheRule(Rule):
    """Replace @functools.lru_cache() and @functools.lru_cache(maxsize=None) with @functools.cache.

    @functools.cache was added in Python 3.9 and is the idiomatic, zero-overhead
    equivalent of @functools.lru_cache(maxsize=None).

    Transforms only the QUALIFIED form (@functools.lru_cache(...) -> @functools.cache).
    The bare-name form (@lru_cache(...) after `from functools import lru_cache`) is
    intentionally NOT transformed in this phase, because adding a new `cache` import
    is out of scope. That enhancement can be addressed in a future phase.

    Conditions for transformation (ALL must hold):
    - The decorator is a Call node whose func is functools.lru_cache (Attribute form).
    - No positional arguments.
    - Zero keyword arguments, OR exactly one keyword: maxsize=None.
    - No typed=True kwarg (typed=True changes equality semantics).

    Cases that return None (no transformation):
    - @functools.lru_cache(maxsize=128) — explicit non-None maxsize.
    - @functools.lru_cache(typed=True) — different caching semantics.
    - @functools.lru_cache(maxsize=None, typed=True) — typed flag present.
    - @lru_cache(...) bare-name form — scope limitation, see docstring.
    - Any positional arguments.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Call)

        func = node.func

        # Only handle qualified functools.lru_cache(...) form.
        assert isinstance(func, ast.Attribute)
        assert func.attr == "lru_cache"
        assert isinstance(func.value, ast.Name)
        assert func.value.id == "functools"

        # Reject any positional arguments.
        if node.args:
            return None

        kwargs = {kw.arg: kw.value for kw in node.keywords}

        # Reject if typed=True is present (different equality semantics).
        if "typed" in kwargs:
            return None

        # Accept only zero kwargs OR maxsize=None.
        if kwargs:
            if set(kwargs.keys()) != {"maxsize"}:
                return None
            maxsize_val = kwargs["maxsize"]
            if not (isinstance(maxsize_val, ast.Constant) and maxsize_val.value is None):
                return None

        # Build replacement: ast.Attribute(value=Name('functools'), attr='cache')
        new_attr = clone(func)
        new_attr.attr = "cache"

        # The decorator becomes @functools.cache (an Attribute, no Call wrapper).
        return Replace(node, new_attr)
