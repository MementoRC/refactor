from __future__ import annotations

import ast

from refactor import Replace
from refactor.actions import Erase, InsertAfter, InsertBefore
from refactor.common import clone
from refactor.core import Rule

# Names that indicate a unittest base class
_UNITTEST_BASES = frozenset({"TestCase", "IsolatedAsyncioWrapperTestCase"})

# Qualified names: unittest.TestCase, unittest.IsolatedAsyncioTestCase, etc.
_UNITTEST_MODULE = "unittest"

# The specific wrapper import module path
_WRAPPER_MODULE = "test.isolated_asyncio_wrapper_test_case"
_WRAPPER_CLASS = "IsolatedAsyncioWrapperTestCase"


def _is_unittest_base(node: ast.expr, unittest_names: set[str]) -> bool:
    """Return True if the expression is a known unittest base class reference."""
    if isinstance(node, ast.Attribute):
        # unittest.TestCase style
        return isinstance(node.value, ast.Name) and node.value.id == _UNITTEST_MODULE
    if isinstance(node, ast.Name):
        return node.id in unittest_names
    return False


def _get_unittest_names_from_imports(tree: ast.Module) -> set[str]:
    """Collect names imported from unittest (e.g. TestCase from 'from unittest import TestCase')."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == _UNITTEST_MODULE:
                for alias in node.names:
                    imported_name = alias.asname or alias.name
                    names.add(imported_name)
            elif node.module == _WRAPPER_MODULE:
                for alias in node.names:
                    if alias.name == _WRAPPER_CLASS:
                        imported_name = alias.asname or alias.name
                        names.add(imported_name)
    return names


class RemoveUnittestInheritance(Rule):
    """Remove unittest base classes from test class definitions.

    Handles:
    - class Foo(unittest.TestCase): -> class Foo:
    - class Foo(TestCase): -> class Foo:  (when TestCase imported from unittest)
    - class Foo(IsolatedAsyncioWrapperTestCase): -> class Foo:

    Does NOT modify classes that have non-unittest bases (mixins) remaining
    after removing unittest bases, to avoid breaking pytest class instantiation.
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.ClassDef)
        assert len(node.bases) > 0

        # Collect names that represent unittest base classes in this module
        tree = self.context.tree
        assert isinstance(tree, ast.Module)
        unittest_names = _get_unittest_names_from_imports(tree)

        # Check if any base is a unittest base
        has_unittest_base = any(_is_unittest_base(base, unittest_names) for base in node.bases)
        assert has_unittest_base

        # Filter out unittest bases, keeping non-unittest ones
        non_unittest_bases = [
            base for base in node.bases if not _is_unittest_base(base, unittest_names)
        ]

        # Only remove unittest bases when ALL bases are unittest bases (result is bare class).
        # If non-unittest bases (mixins) remain, keep the class unchanged to avoid
        # TypeError when pytest tries to instantiate the class.
        assert len(non_unittest_bases) == 0

        new_node = clone(node)
        new_node.bases = []
        return Replace(node, new_node)


def _unittest_is_still_used(tree: ast.Module) -> bool:
    """Return True if 'unittest' is used as a name in expressions (e.g. unittest.mock)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == _UNITTEST_MODULE:
                return True
        if isinstance(node, ast.Name):
            if node.id == _UNITTEST_MODULE:
                return True
    return False


def _classdef_still_uses_name(tree: ast.Module, name: str) -> bool:
    """Return True if any ClassDef base still references the given name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == name:
                    return True
    return False


class RemoveUnittestImport(Rule):
    """Remove 'import unittest' or 'from unittest import TestCase' when no longer needed.

    - Removes 'import unittest' if unittest is not used anywhere else in the module.
    - Removes 'from unittest import TestCase' if TestCase is not used as a class base.
    """

    def match(self, node: ast.AST) -> Erase | None:
        assert isinstance(node, (ast.Import, ast.ImportFrom))

        tree = self.context.tree
        assert isinstance(tree, ast.Module)

        if isinstance(node, ast.Import):
            # Handle 'import unittest'
            assert any(alias.name == _UNITTEST_MODULE for alias in node.names)

            # Check if unittest is still referenced anywhere (excluding this import)
            used = _unittest_is_still_used_excluding_import(tree, node)
            assert not used
            return Erase(node)

        if isinstance(node, ast.ImportFrom):
            # Handle 'from unittest import TestCase [, ...]'
            assert node.module == _UNITTEST_MODULE

            # Find which names from this import are no longer used as class bases
            unused_names = []
            for alias in node.names:
                imported_name = alias.asname or alias.name
                if not _classdef_still_uses_name(tree, imported_name):
                    unused_names.append(alias)

            # Only act if ALL names in this import are unused
            assert len(unused_names) == len(node.names)
            return Erase(node)

        return None


def _unittest_is_still_used_excluding_import(tree: ast.Module, import_node: ast.Import) -> bool:
    """Return True if 'unittest' is used in any node other than the given import."""
    for node in ast.walk(tree):
        if node is import_node:
            continue
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == _UNITTEST_MODULE:
                return True
        if isinstance(node, ast.Name):
            if node.id == _UNITTEST_MODULE:
                return True
    return False


class RemoveTestWrapperImport(Rule):
    """Remove 'from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase'
    when the class is no longer used as a base class.
    """

    def match(self, node: ast.AST) -> Erase | None:
        assert isinstance(node, ast.ImportFrom)
        assert node.module == _WRAPPER_MODULE

        # Check that this import includes IsolatedAsyncioWrapperTestCase
        wrapper_aliases = [alias for alias in node.names if alias.name == _WRAPPER_CLASS]
        assert len(wrapper_aliases) > 0

        tree = self.context.tree
        assert isinstance(tree, ast.Module)

        # Check if any of the imported names are still used
        for alias in wrapper_aliases:
            imported_name = alias.asname or alias.name
            if _classdef_still_uses_name(tree, imported_name):
                return None

        return Erase(node)


def _make_name(id: str) -> ast.Name:
    """Create an ast.Name node with Load context."""
    return ast.Name(id=id, ctx=ast.Load())


def _make_attr_call(obj: str, attr: str, args: list[ast.expr]) -> ast.Call:
    """Create ast.Call for obj.attr(args...)."""
    return ast.Call(
        func=ast.Attribute(
            value=_make_name(obj),
            attr=attr,
            ctx=ast.Load(),
        ),
        args=args,
        keywords=[],
    )


def _extract_msg(call: ast.Call, msg_index: int) -> ast.expr | None:
    """Extract optional message argument at msg_index or from 'msg' keyword."""
    if len(call.args) > msg_index:
        return call.args[msg_index]
    for kw in call.keywords:
        if kw.arg == "msg":
            return kw.value
    return None


# Map from assertion method name to a callable that takes the Call node and
# returns the test expression for ast.Assert.
def _conv_equal(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.Eq()], comparators=[call.args[1]])


def _conv_not_equal(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.NotEq()], comparators=[call.args[1]])


def _conv_true(call: ast.Call) -> ast.expr:
    return call.args[0]


def _conv_false(call: ast.Call) -> ast.expr:
    return ast.UnaryOp(op=ast.Not(), operand=call.args[0])


def _conv_is(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.Is()], comparators=[call.args[1]])


def _conv_is_not(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.IsNot()], comparators=[call.args[1]])


def _conv_is_none(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.Is()], comparators=[ast.Constant(value=None)])


def _conv_is_not_none(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.IsNot()], comparators=[ast.Constant(value=None)])


def _conv_in(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.In()], comparators=[call.args[1]])


def _conv_not_in(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.NotIn()], comparators=[call.args[1]])


def _conv_is_instance(call: ast.Call) -> ast.expr:
    return ast.Call(
        func=_make_name("isinstance"),
        args=[call.args[0], call.args[1]],
        keywords=[],
    )


def _conv_almost_equal(call: ast.Call) -> ast.expr:
    approx = _make_attr_call("pytest", "approx", [call.args[1]])
    return ast.Compare(left=call.args[0], ops=[ast.Eq()], comparators=[approx])


def _conv_greater(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.Gt()], comparators=[call.args[1]])


def _conv_greater_equal(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.GtE()], comparators=[call.args[1]])


def _conv_less(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.Lt()], comparators=[call.args[1]])


def _conv_less_equal(call: ast.Call) -> ast.expr:
    return ast.Compare(left=call.args[0], ops=[ast.LtE()], comparators=[call.args[1]])


def _conv_regex(call: ast.Call) -> ast.expr:
    # assertRegex(s, r) -> re.search(r, s)  — note argument order swap
    return _make_attr_call("re", "search", [call.args[1], call.args[0]])


def _conv_count_equal(call: ast.Call) -> ast.expr:
    sorted_a = _make_attr_call("", "sorted", [call.args[0]])
    sorted_a = ast.Call(func=_make_name("sorted"), args=[call.args[0]], keywords=[])
    sorted_b = ast.Call(func=_make_name("sorted"), args=[call.args[1]], keywords=[])
    return ast.Compare(left=sorted_a, ops=[ast.Eq()], comparators=[sorted_b])


# Maps method name → (converter_fn, msg_arg_index)
_ASSERT_CONVERTERS: dict[str, tuple[object, int]] = {
    "assertEqual": (_conv_equal, 2),
    "assertNotEqual": (_conv_not_equal, 2),
    "assertTrue": (_conv_true, 1),
    "assertFalse": (_conv_false, 1),
    "assertIs": (_conv_is, 2),
    "assertIsNot": (_conv_is_not, 2),
    "assertIsNone": (_conv_is_none, 1),
    "assertIsNotNone": (_conv_is_not_none, 1),
    "assertIn": (_conv_in, 2),
    "assertNotIn": (_conv_not_in, 2),
    "assertIsInstance": (_conv_is_instance, 2),
    "assertAlmostEqual": (_conv_almost_equal, 2),
    "assertGreater": (_conv_greater, 2),
    "assertGreaterEqual": (_conv_greater_equal, 2),
    "assertLess": (_conv_less, 2),
    "assertLessEqual": (_conv_less_equal, 2),
    "assertRegex": (_conv_regex, 2),
    "assertCountEqual": (_conv_count_equal, 2),
}


class ConvertAssertions(Rule):
    """Convert unittest self.assertX() calls to plain pytest assert statements.

    Examples:
        self.assertEqual(a, b)      -> assert a == b
        self.assertTrue(x)          -> assert x
        self.assertAlmostEqual(a,b) -> assert a == pytest.approx(b)
        self.assertRegex(s, r)      -> assert re.search(r, s)
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.Expr)
        assert isinstance(node.value, ast.Call)

        call = node.value
        assert isinstance(call.func, ast.Attribute)
        assert isinstance(call.func.value, ast.Name)
        assert call.func.value.id == "self"

        method_name = call.func.attr
        assert method_name in _ASSERT_CONVERTERS

        converter_fn, msg_index = _ASSERT_CONVERTERS[method_name]

        test_expr = converter_fn(call)  # type: ignore[operator]
        msg_expr = _extract_msg(call, msg_index)

        new_node = ast.Assert(test=test_expr, msg=msg_expr)
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)

        return Replace(node, new_node)


# ---------------------------------------------------------------------------
# ConvertSetUpTearDown
# ---------------------------------------------------------------------------


def _make_fixture_decorator(autouse: bool = True) -> ast.expr:
    """Create @pytest.fixture(autouse=True) decorator node."""
    return ast.Call(
        func=ast.Attribute(
            value=_make_name("pytest"),
            attr="fixture",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[ast.keyword(arg="autouse", value=ast.Constant(value=autouse))],
    )


class ConvertSetUpTearDown(Rule):
    """Convert unittest setUp/tearDown methods to pytest fixtures.

    - setUp(self) -> @pytest.fixture(autouse=True) / def setup(self)
    - asyncSetUp(self) -> @pytest.fixture(autouse=True) / async def setup(self)
    - tearDown(self) -> @pytest.fixture(autouse=True) / def teardown_fixture(self) with yield prepended
    - asyncTearDown(self) -> @pytest.fixture(autouse=True) / async def teardown_fixture(self) with yield
    - super().setUp() calls -> Erase
    - self.maxDiff = None -> Erase
    """

    _SETUP_NAMES = frozenset({"setUp", "asyncSetUp"})
    _TEARDOWN_NAMES = frozenset({"tearDown", "asyncTearDown"})

    def match(self, node: ast.AST) -> Replace | Erase | None:
        # --- Handle super().setUp() removal ---
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Call)
            call = node.value
            assert isinstance(call.func, ast.Attribute)
            assert call.func.attr in ("setUp", "asyncSetUp", "tearDown", "asyncTearDown")
            assert isinstance(call.func.value, ast.Call)
            inner = call.func.value
            assert isinstance(inner.func, ast.Name)
            assert inner.func.id == "super"
            return Erase(node)

        # --- Handle self.maxDiff = None removal ---
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1
            target = node.targets[0]
            assert isinstance(target, ast.Attribute)
            assert isinstance(target.value, ast.Name)
            assert target.value.id == "self"
            assert target.attr == "maxDiff"
            assert isinstance(node.value, ast.Constant)
            assert node.value.value is None
            return Erase(node)

        # --- Handle setUp / asyncSetUp / tearDown / asyncTearDown ---
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        assert node.name in (self._SETUP_NAMES | self._TEARDOWN_NAMES)

        new_node = clone(node)
        fixture_dec = _make_fixture_decorator(autouse=True)
        ast.fix_missing_locations(fixture_dec)

        # Prepend the fixture decorator (keep existing decorators after it)
        new_node.decorator_list = [fixture_dec] + new_node.decorator_list

        if node.name in self._TEARDOWN_NAMES:
            new_node.name = "teardown_fixture"
            # Prepend a bare yield statement to the body
            yield_stmt = ast.Expr(value=ast.Yield(value=None))
            ast.fix_missing_locations(yield_stmt)
            new_node.body = [yield_stmt] + new_node.body
        else:
            # setUp / asyncSetUp -> rename to setup
            new_node.name = "setup"

        ast.fix_missing_locations(new_node)
        return Replace(node, new_node)


# ---------------------------------------------------------------------------
# ConvertAssertRaises
# ---------------------------------------------------------------------------


class ConvertAssertRaises(Rule):
    """Convert self.assertRaises(...) context managers to pytest.raises(...).

    Example:
        with self.assertRaises(ValueError):   ->   with pytest.raises(ValueError):
            ...                                        ...
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, ast.With)

        # Find the first item whose context_expr is self.assertRaises(...)
        new_items = list(node.items)
        changed = False

        for i, item in enumerate(new_items):
            ctx = item.context_expr
            assert isinstance(ctx, ast.Call)
            assert isinstance(ctx.func, ast.Attribute)
            assert isinstance(ctx.func.value, ast.Name)
            assert ctx.func.value.id == "self"
            assert ctx.func.attr == "assertRaises"

            # Build pytest.raises(...) call with the same args/keywords
            new_call = ast.Call(
                func=ast.Attribute(
                    value=_make_name("pytest"),
                    attr="raises",
                    ctx=ast.Load(),
                ),
                args=list(ctx.args),
                keywords=list(ctx.keywords),
            )
            ast.copy_location(new_call, ctx)
            ast.fix_missing_locations(new_call)

            new_item = ast.withitem(
                context_expr=new_call,
                optional_vars=item.optional_vars,
            )
            new_items[i] = new_item
            changed = True
            break  # only transform the first matching item per With node

        assert changed

        new_node = clone(node)
        new_node.items = new_items
        ast.fix_missing_locations(new_node)
        return Replace(node, new_node)


# ---------------------------------------------------------------------------
# ConvertUnittestDecorators
# ---------------------------------------------------------------------------


def _build_pytest_mark_skip(args: list[ast.expr]) -> ast.expr:
    """Build @pytest.mark.skip(reason=<reason>) from @unittest.skip(<reason>)."""
    reason_arg = args[0] if args else ast.Constant(value="")
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=_make_name("pytest"),
                attr="mark",
                ctx=ast.Load(),
            ),
            attr="skip",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[ast.keyword(arg="reason", value=reason_arg)],
    )


def _build_pytest_mark_skipif(condition: ast.expr, reason: ast.expr) -> ast.expr:
    """Build @pytest.mark.skipif(condition, reason=<reason>)."""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=_make_name("pytest"),
                attr="mark",
                ctx=ast.Load(),
            ),
            attr="skipif",
            ctx=ast.Load(),
        ),
        args=[condition],
        keywords=[ast.keyword(arg="reason", value=reason)],
    )


def _match_unittest_skip_decorator(
    dec: ast.expr,
) -> tuple[str, list[ast.expr]] | None:
    """Return (decorator_attr_name, args) if dec is unittest.skip/skipIf/skipUnless."""
    _SKIP_NAMES = ("skip", "skipIf", "skipUnless")

    if isinstance(dec, ast.Call):
        func = dec.func
        if isinstance(func, ast.Attribute) and func.attr in _SKIP_NAMES:
            if isinstance(func.value, ast.Name) and func.value.id == "unittest":
                return func.attr, dec.args
        # bare name (from unittest import skip)
        if isinstance(func, ast.Name) and func.id in _SKIP_NAMES:
            return func.id, dec.args

    return None


class ConvertUnittestDecorators(Rule):
    """Convert unittest skip decorators to pytest equivalents.

    - @unittest.skip("reason")        -> @pytest.mark.skip(reason="reason")
    - @unittest.skipIf(cond, "r")     -> @pytest.mark.skipif(cond, reason="r")
    - @unittest.skipUnless(cond, "r") -> @pytest.mark.skipif(not cond, reason="r")
    """

    def match(self, node: ast.AST) -> Replace | None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        assert len(node.decorator_list) > 0

        new_decorators = list(node.decorator_list)
        changed = False

        for i, dec in enumerate(new_decorators):
            result = _match_unittest_skip_decorator(dec)
            if result is None:
                continue

            attr_name, args = result
            if attr_name == "skip":
                new_dec = _build_pytest_mark_skip(args)
            elif attr_name == "skipIf":
                # skipIf(condition, reason)
                assert len(args) >= 2
                new_dec = _build_pytest_mark_skipif(args[0], args[1])
            elif attr_name == "skipUnless":
                # skipUnless(condition, reason) -> skipif(not condition, reason)
                assert len(args) >= 2
                negated = ast.UnaryOp(op=ast.Not(), operand=args[0])
                ast.fix_missing_locations(negated)
                new_dec = _build_pytest_mark_skipif(negated, args[1])
            else:
                continue

            ast.fix_missing_locations(new_dec)
            new_decorators[i] = new_dec
            changed = True

        assert changed

        new_node = clone(node)
        new_node.decorator_list = new_decorators
        ast.fix_missing_locations(new_node)
        return Replace(node, new_node)


def _has_pytest_reference(tree: ast.Module) -> bool:
    """Return True if the tree has any reference to pytest (pytest.* or bare pytest name)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "pytest":
                return True
        if isinstance(node, ast.Name) and node.id == "pytest":
            return True
    return False


def _has_pytest_import(tree: ast.Module) -> bool:
    """Return True if ``import pytest`` already exists in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest" and alias.asname is None:
                    return True
    return False


class AddPytestImport(Rule):
    """Insert ``import pytest`` when pytest references exist but no import is present yet.

    This fires after all other rules have added pytest references (e.g. pytest.raises,
    pytest.approx, pytest.fixture, pytest.mark.*).

    - When import statements are present: fires on the *last* import and inserts after it.
    - When no import statements remain (e.g. after RemoveUnittestImport erased the only
      import): fires on the first statement of the module and inserts before it.
    """

    def match(self, node: ast.AST) -> InsertAfter | InsertBefore | None:
        assert isinstance(node, ast.stmt)

        tree = self.context.tree
        assert isinstance(tree, ast.Module)
        assert tree.body

        # Nothing to do if import pytest is already present
        assert not _has_pytest_import(tree)

        # Only fire if there are actual pytest.* / pytest references in the tree
        assert _has_pytest_reference(tree)

        import_node = ast.Import(names=[ast.alias(name="pytest")])
        ast.fix_missing_locations(import_node)

        # Prefer inserting after the last top-level import
        last_import: ast.stmt | None = None
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                last_import = stmt

        if last_import is not None:
            # Only fire on the last import so insertion happens exactly once
            assert last_import is node, "Not the last import statement"
            return InsertAfter(node, import_node)

        # No imports remain — insert before the first statement, firing only once
        assert tree.body[0] is node, "Not the first statement"
        return InsertBefore(node, import_node)


# All migration rules in application order
ALL_RULES = [
    RemoveUnittestInheritance,
    RemoveUnittestImport,
    RemoveTestWrapperImport,
    ConvertAssertions,
    ConvertSetUpTearDown,
    ConvertAssertRaises,
    ConvertUnittestDecorators,
    AddPytestImport,  # Must be last: adds import pytest after all pytest refs are created
]

if __name__ == "__main__":
    from functools import partial

    import refactor
    from refactor.runner import unbound_main

    session = refactor.Session(ALL_RULES)
    session.config.debug_mode = True  # Force single-worker to avoid pickle errors (#9)
    main = partial(unbound_main, session=session)
    main()
