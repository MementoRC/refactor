from __future__ import annotations

import ast

from refactor import Replace
from refactor.common import clone
from refactor.core import Rule


class PairwiseRule(Rule):
    """Replace ``zip(seq, seq[1:])`` with ``itertools.pairwise(seq)``.

    ``itertools.pairwise`` was added in Python 3.10 and is the idiomatic way to
    iterate over consecutive pairs from a sequence.

    Conservative scope: only matches the simple NAME form, i.e. the first argument
    is a bare ``Name`` node and the second argument is ``Name[1:]`` — the SAME
    name in both positions.  More complex expressions (e.g. function calls or
    attribute access as the iterable) are not matched to avoid false positives.

    Opt-in only (``--enable=itertools-modern``).

    NOTE: The rule does NOT add ``import itertools``.  The developer must ensure
    that ``import itertools`` is present in the target file.

    Matches:
        zip(seq, seq[1:])           -> itertools.pairwise(seq)
        zip(seq, seq[1:], strict=…) -> itertools.pairwise(seq)  (strict kwarg dropped)

    Does NOT match:
        zip(a, b)         — different names
        zip(seq, seq[2:]) — wrong slice lower bound
        zip(seq, seq[:-1], seq[1:]) — more than two args
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Call)

        # Must be a bare zip(…) call.
        assert isinstance(node.func, ast.Name)
        assert node.func.id == "zip"

        # Exactly 2 positional arguments; only the optional strict= kwarg is allowed.
        assert len(node.args) == 2
        for kw in node.keywords:
            assert kw.arg == "strict"

        first, second = node.args

        # First arg must be a plain Name.
        assert isinstance(first, ast.Name)
        seq_name = first.id

        # Second arg must be Name[1:] with the SAME name.
        assert isinstance(second, ast.Subscript)
        assert isinstance(second.value, ast.Name)
        assert second.value.id == seq_name

        slc = second.slice
        assert isinstance(slc, ast.Slice)
        assert isinstance(slc.lower, ast.Constant)
        assert slc.lower.value == 1
        assert slc.upper is None
        assert slc.step is None

        # Build itertools.pairwise(seq).
        new_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="itertools", ctx=ast.Load()),
                attr="pairwise",
                ctx=ast.Load(),
            ),
            args=[clone(first)],
            keywords=[],
        )
        ast.fix_missing_locations(new_node)
        return Replace(node, new_node)


class BatchedRule(Rule):
    """Replace the canonical manual-chunking list comprehension with ``itertools.batched``.

    ``itertools.batched`` was added in Python 3.12 and is the idiomatic way to
    split an iterable into fixed-size chunks.

    Conservative scope: only matches the exact canonical form::

        [iterable[i:i+n] for i in range(0, len(iterable), n)]

    where ``iterable``, ``i``, and ``n`` are all plain ``Name`` nodes, and the
    same names are used consistently between the element expression and the
    comprehension iterator.  Any deviation — different slice form, extra
    generators, non-Name nodes — is left untouched.

    Opt-in only (``--enable=itertools-modern``).

    NOTE: The rule does NOT add ``import itertools``.  The developer must ensure
    that ``import itertools`` is present in the target file.

    Produces::

        list(itertools.batched(iterable, n))
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ListComp)

        # Exactly one generator.
        assert len(node.generators) == 1
        gen = node.generators[0]

        # Generator target must be a plain Name (the loop variable i).
        assert isinstance(gen.target, ast.Name)
        i_name = gen.target.id

        # Generator iter must be range(0, len(iterable), n).
        assert isinstance(gen.iter, ast.Call)
        assert isinstance(gen.iter.func, ast.Name)
        assert gen.iter.func.id == "range"
        assert len(gen.iter.args) == 3
        assert not gen.iter.keywords

        range_start, range_stop, range_step = gen.iter.args

        # range start must be Constant(0).
        assert isinstance(range_start, ast.Constant)
        assert range_start.value == 0

        # range stop must be len(iterable) — len is a bare Name call.
        assert isinstance(range_stop, ast.Call)
        assert isinstance(range_stop.func, ast.Name)
        assert range_stop.func.id == "len"
        assert len(range_stop.args) == 1
        assert not range_stop.keywords
        assert isinstance(range_stop.args[0], ast.Name)
        iterable_name = range_stop.args[0].id

        # range step must be a plain Name (the chunk size n).
        assert isinstance(range_step, ast.Name)
        n_name = range_step.id

        # No comprehension conditions (no `if` clauses).
        assert not gen.ifs

        # Element must be iterable[i:i+n].
        elt = node.elt
        assert isinstance(elt, ast.Subscript)
        assert isinstance(elt.value, ast.Name)
        assert elt.value.id == iterable_name

        slc = elt.slice
        assert isinstance(slc, ast.Slice)

        # Slice lower must be the same loop variable i.
        assert isinstance(slc.lower, ast.Name)
        assert slc.lower.id == i_name

        # Slice upper must be i + n (BinOp with Add).
        assert isinstance(slc.upper, ast.BinOp)
        assert isinstance(slc.upper.op, ast.Add)
        assert isinstance(slc.upper.left, ast.Name)
        assert slc.upper.left.id == i_name
        assert isinstance(slc.upper.right, ast.Name)
        assert slc.upper.right.id == n_name

        assert slc.step is None

        # Build list(itertools.batched(iterable, n)).
        new_node = ast.Call(
            func=ast.Name(id="list", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="itertools", ctx=ast.Load()),
                        attr="batched",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id=iterable_name, ctx=ast.Load()),
                        ast.Name(id=n_name, ctx=ast.Load()),
                    ],
                    keywords=[],
                )
            ],
            keywords=[],
        )
        ast.fix_missing_locations(new_node)
        return Replace(node, new_node)
