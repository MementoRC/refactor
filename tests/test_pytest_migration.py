from __future__ import annotations

import textwrap

from refactor import Session
from refactor.rules.pytest_migration import (
    ALL_RULES,
    AddPytestImport,
    ConvertAssertions,
    ConvertAssertRaises,
    ConvertExceptionToValue,
    ConvertSetUpTearDown,
    ConvertUnittestDecorators,
    RemoveTestWrapperImport,
    RemoveUnittestImport,
    RemoveUnittestInheritance,
)


def _run(*rules, source: str) -> str:
    return Session(list(rules)).run(textwrap.dedent(source))


def test_remove_simple_testcase():
    source = """\
        import unittest

        class TestFoo(unittest.TestCase):
            def test_something(self):
                pass
        """
    expected = """\
        import unittest

        class TestFoo:
            def test_something(self):
                pass
        """
    result = _run(RemoveUnittestInheritance, source=source)
    assert result == textwrap.dedent(expected)


def test_remove_imported_testcase():
    source = """\
        from unittest import TestCase

        class TestFoo(TestCase):
            def test_something(self):
                pass
        """
    expected = """\
        from unittest import TestCase

        class TestFoo:
            def test_something(self):
                pass
        """
    result = _run(RemoveUnittestInheritance, source=source)
    assert result == textwrap.dedent(expected)


def test_remove_async_wrapper():
    source = """\
        from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase

        class TestFoo(IsolatedAsyncioWrapperTestCase):
            async def test_something(self):
                pass
        """
    expected = """\
        from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase

        class TestFoo:
            async def test_something(self):
                pass
        """
    result = _run(RemoveUnittestInheritance, source=source)
    assert result == textwrap.dedent(expected)


def test_multiple_inheritance_keep_mixin():
    source = """\
        from unittest import TestCase

        class TestFoo(SomeMixin, TestCase):
            def test_something(self):
                pass
        """
    result = _run(RemoveUnittestInheritance, source=source)
    # When mixin bases remain, TestCase is also kept to avoid pytest TypeError
    assert "class TestFoo(SomeMixin, TestCase):" in result
    assert "__init__ = None" not in result
    assert "def test_something(self):" in result


def test_multiple_inheritance_all_unittest():
    source = """\
        from unittest import TestCase

        class TestFoo(TestCase):
            def test_something(self):
                pass
        """
    expected = """\
        from unittest import TestCase

        class TestFoo:
            def test_something(self):
                pass
        """
    result = _run(RemoveUnittestInheritance, source=source)
    assert result == textwrap.dedent(expected)


def test_remove_unused_unittest_import():
    source = """\
        import unittest

        class TestFoo:
            def test_something(self):
                pass
        """
    # Erase leaves a blank line where the import was
    expected = """\

        class TestFoo:
            def test_something(self):
                pass
        """
    result = _run(RemoveUnittestImport, source=source)
    assert result == textwrap.dedent(expected)


def test_keep_unittest_import_for_mock():
    source = """\
        import unittest

        class TestFoo:
            @unittest.mock.patch("foo.bar")
            def test_something(self, mock_bar):
                pass
        """
    result = _run(RemoveUnittestImport, source=source)
    assert result == textwrap.dedent(source)


def test_remove_wrapper_import():
    source = """\
        from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase

        class TestFoo:
            async def test_something(self):
                pass
        """
    # Erase leaves a blank line where the import was
    expected = """\

        class TestFoo:
            async def test_something(self):
                pass
        """
    result = _run(RemoveTestWrapperImport, source=source)
    assert result == textwrap.dedent(expected)


def test_no_change_non_unittest():
    source = """\
        class TestFoo(SomeMixin):
            def test_something(self):
                pass
        """
    result = _run(RemoveUnittestInheritance, source=source)
    assert result == textwrap.dedent(source)


def test_full_file_transformation():
    """Complete before/after with multiple classes and all three rules combined."""
    source = """\
        import unittest
        from unittest import TestCase
        from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase

        class TestSimple(unittest.TestCase):
            def test_a(self):
                pass

        class TestImported(TestCase):
            def test_b(self):
                pass

        class TestAsync(IsolatedAsyncioWrapperTestCase):
            async def test_c(self):
                pass

        class TestMixed(SomeMixin, TestCase):
            def test_d(self):
                pass
        """
    result = _run(
        RemoveUnittestInheritance,
        RemoveUnittestImport,
        RemoveTestWrapperImport,
        source=source,
    )
    # unittest module import removed (TestSimple no longer uses unittest.TestCase)
    assert "import unittest" not in result
    # 'from unittest import TestCase' kept because TestMixed still uses TestCase
    assert "from unittest import TestCase" in result
    # IsolatedAsyncioWrapperTestCase import removed (TestAsync no longer uses it)
    assert "IsolatedAsyncioWrapperTestCase" not in result
    # Pure TestCase classes become bare classes
    assert "class TestSimple:" in result
    assert "class TestImported:" in result
    assert "class TestAsync:" in result
    # Mixin class is left unchanged (both bases kept) to avoid pytest TypeError
    assert "class TestMixed(SomeMixin, TestCase):" in result
    assert "__init__ = None" not in result


def test_remove_from_unittest_import_testcase_when_unused():
    """'from unittest import TestCase' is removed after classes are migrated."""
    source = """\
        from unittest import TestCase

        class TestFoo:
            def test_something(self):
                pass
        """
    # Erase leaves a blank line where the import was
    expected = """\

        class TestFoo:
            def test_something(self):
                pass
        """
    result = _run(RemoveUnittestImport, source=source)
    assert result == textwrap.dedent(expected)


def test_combined_inheritance_and_import_removal():
    """Running all rules together removes both bases and then the now-unused imports."""
    source = """\
        import unittest
        from unittest import TestCase

        class TestFoo(unittest.TestCase):
            def test_a(self):
                pass

        class TestBar(TestCase):
            def test_b(self):
                pass
        """
    # Each erased import leaves a blank line; the Session collapses them to one leading blank line
    expected = """\

        class TestFoo:
            def test_a(self):
                pass

        class TestBar:
            def test_b(self):
                pass
        """
    result = _run(
        RemoveUnittestInheritance,
        RemoveUnittestImport,
        source=source,
    )
    assert result == textwrap.dedent(expected)


def test_multiple_inheritance_wrapper():
    source = textwrap.dedent("""\
        from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase

        class TestFoo(SomeMixin, IsolatedAsyncioWrapperTestCase):
            pass
    """)
    result = _run(RemoveUnittestInheritance, RemoveTestWrapperImport, source=source)
    # When mixin bases remain, IsolatedAsyncioWrapperTestCase is also kept
    assert "class TestFoo(SomeMixin, IsolatedAsyncioWrapperTestCase):" in result
    assert "IsolatedAsyncioWrapperTestCase" in result


def test_nested_class_definition():
    source = textwrap.dedent("""\
        import unittest

        class OuterTest(unittest.TestCase):
            class InnerTest(unittest.TestCase):
                def test_inner(self):
                    pass
            def test_outer(self):
                pass
    """)
    result = _run(RemoveUnittestInheritance, RemoveUnittestImport, source=source)
    assert "class OuterTest:" in result
    assert "class InnerTest:" in result
    assert "import unittest" not in result


# ---------------------------------------------------------------------------
# ConvertAssertions tests
# ---------------------------------------------------------------------------


def _assert(source: str) -> str:
    """Run only ConvertAssertions on a single dedented statement."""
    return _run(ConvertAssertions, source=textwrap.dedent(source))


def test_convert_assert_equal():
    result = _assert("self.assertEqual(a, b)\n")
    assert result == "assert a == b\n"


def test_convert_assert_not_equal():
    result = _assert("self.assertNotEqual(a, b)\n")
    assert result == "assert a != b\n"


def test_convert_assert_true():
    result = _assert("self.assertTrue(x)\n")
    assert result == "assert x\n"


def test_convert_assert_false():
    result = _assert("self.assertFalse(x)\n")
    assert result == "assert not x\n"


def test_convert_assert_is():
    result = _assert("self.assertIs(a, b)\n")
    assert result == "assert a is b\n"


def test_convert_assert_is_not():
    result = _assert("self.assertIsNot(a, b)\n")
    assert result == "assert a is not b\n"


def test_convert_assert_is_none():
    result = _assert("self.assertIsNone(x)\n")
    assert result == "assert x is None\n"


def test_convert_assert_is_not_none():
    result = _assert("self.assertIsNotNone(x)\n")
    assert result == "assert x is not None\n"


def test_convert_assert_in():
    result = _assert("self.assertIn(a, b)\n")
    assert result == "assert a in b\n"


def test_convert_assert_not_in():
    result = _assert("self.assertNotIn(a, b)\n")
    assert result == "assert a not in b\n"


def test_convert_assert_is_instance():
    result = _assert("self.assertIsInstance(a, MyType)\n")
    assert result == "assert isinstance(a, MyType)\n"


def test_convert_assert_almost_equal():
    result = _assert("self.assertAlmostEqual(a, b)\n")
    assert result == "assert a == pytest.approx(b)\n"


def test_convert_assert_greater():
    result = _assert("self.assertGreater(a, b)\n")
    assert result == "assert a > b\n"


def test_convert_assert_greater_equal():
    result = _assert("self.assertGreaterEqual(a, b)\n")
    assert result == "assert a >= b\n"


def test_convert_assert_less():
    result = _assert("self.assertLess(a, b)\n")
    assert result == "assert a < b\n"


def test_convert_assert_less_equal():
    result = _assert("self.assertLessEqual(a, b)\n")
    assert result == "assert a <= b\n"


def test_convert_assert_regex():
    result = _assert("self.assertRegex(s, r)\n")
    assert result == "assert re.search(r, s)\n"


def test_convert_assert_count_equal():
    result = _assert("self.assertCountEqual(a, b)\n")
    assert result == "assert sorted(a) == sorted(b)\n"


def test_convert_assert_equal_with_message():
    result = _assert('self.assertEqual(a, b, "mismatch")\n')
    assert result == 'assert a == b, "mismatch"\n'


def test_convert_assert_true_with_msg_kwarg():
    result = _assert('self.assertTrue(x, msg="should be true")\n')
    assert result == 'assert x, "should be true"\n'


def test_convert_assert_not_equal_with_message():
    result = _assert('self.assertNotEqual(x, y, "values differ")\n')
    assert result == 'assert x != y, "values differ"\n'


def test_convert_assertions_full_method():
    """A method with multiple assertion types is fully transformed."""
    source = """\
        def test_example(self):
            self.assertEqual(result, 42)
            self.assertTrue(flag)
            self.assertIsNone(value)
            self.assertIn(item, collection)
            self.assertGreater(score, 0)
        """
    expected = """\
        def test_example(self):
            assert result == 42
            assert flag
            assert value is None
            assert item in collection
            assert score > 0
        """
    result = _run(ConvertAssertions, source=source)
    assert result == textwrap.dedent(expected)


def test_no_change_non_self_call():
    """Calls not on self are not converted."""
    source = "other.assertEqual(a, b)\n"
    result = _assert(source)
    assert result == source


def test_no_change_non_assert_method():
    """Non-assertion self methods are not converted."""
    source = "self.setUp()\n"
    result = _assert(source)
    assert result == source


# ---------------------------------------------------------------------------
# ConvertSetUpTearDown tests
# ---------------------------------------------------------------------------


def _setup(source: str) -> str:
    """Run only ConvertSetUpTearDown on a dedented source."""
    return _run(ConvertSetUpTearDown, source=textwrap.dedent(source))


def test_convert_setup():
    source = """\
        def setUp(self):
            self.foo = 1
        """
    result = _setup(source)
    assert "@pytest.fixture(autouse=True)" in result
    assert "def setup(self):" in result
    assert "self.foo = 1" in result
    assert "def setUp" not in result


def test_convert_async_setup():
    source = """\
        async def asyncSetUp(self):
            self.foo = await something()
        """
    result = _setup(source)
    assert "@pytest.fixture(autouse=True)" in result
    assert "async def setup(self):" in result
    assert "def asyncSetUp" not in result


def test_convert_teardown():
    source = """\
        def tearDown(self):
            self.cleanup()
        """
    result = _setup(source)
    assert "@pytest.fixture(autouse=True)" in result
    assert "def teardown_fixture(self):" in result
    assert "yield" in result
    assert "self.cleanup()" in result
    assert "def tearDown" not in result


def test_remove_super_setup():
    source = """\
        def setUp(self):
            super().setUp()
            self.foo = 1
        """
    result = _setup(source)
    assert "super().setUp()" not in result
    assert "self.foo = 1" in result


def test_remove_maxdiff():
    source = """\
        def setUp(self):
            self.maxDiff = None
            self.foo = 1
        """
    result = _setup(source)
    assert "self.maxDiff" not in result
    assert "self.foo = 1" in result


# ---------------------------------------------------------------------------
# ConvertAssertRaises tests
# ---------------------------------------------------------------------------


def _raises(source: str) -> str:
    """Run only ConvertAssertRaises on a dedented source."""
    return _run(ConvertAssertRaises, source=textwrap.dedent(source))


def test_convert_assert_raises_basic():
    source = """\
        with self.assertRaises(ValueError):
            do_something()
        """
    result = _raises(source)
    assert "pytest.raises(ValueError)" in result
    assert "self.assertRaises" not in result
    assert "do_something()" in result


def test_convert_assert_raises_with_as():
    source = """\
        with self.assertRaises(TypeError) as cm:
            do_something()
        """
    result = _raises(source)
    assert "pytest.raises(TypeError)" in result
    assert "as cm:" in result
    assert "self.assertRaises" not in result


# ---------------------------------------------------------------------------
# ConvertUnittestDecorators tests
# ---------------------------------------------------------------------------


def _decorators(source: str) -> str:
    """Run only ConvertUnittestDecorators on a dedented source."""
    return _run(ConvertUnittestDecorators, source=textwrap.dedent(source))


def test_convert_skip_decorator():
    source = """\
        @unittest.skip("not ready")
        def test_foo(self):
            pass
        """
    result = _decorators(source)
    assert "@pytest.mark.skip(reason=" in result
    assert '"not ready"' in result
    assert "@unittest.skip" not in result


def test_convert_skipif_decorator():
    source = """\
        @unittest.skipIf(sys.version_info < (3, 10), "needs 3.10")
        def test_foo(self):
            pass
        """
    result = _decorators(source)
    assert "pytest.mark.skipif" in result
    assert "reason=" in result
    assert "@unittest.skipIf" not in result


def test_full_unittest_to_pytest_migration():
    source = textwrap.dedent("""\
        import unittest
        from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase


        class TestCalculator(IsolatedAsyncioWrapperTestCase):

            def setUp(self):
                self.calc = Calculator()

            def tearDown(self):
                self.calc.close()

            def test_add(self):
                self.assertEqual(self.calc.add(1, 2), 3)
                self.assertTrue(self.calc.is_ready)

            def test_divide_by_zero(self):
                with self.assertRaises(ZeroDivisionError):
                    self.calc.divide(1, 0)

            def test_precision(self):
                self.assertAlmostEqual(self.calc.pi(), 3.14159)

            @unittest.skip("not implemented")
            def test_future_feature(self):
                pass
    """)

    session = Session(ALL_RULES)
    result = session.run(source)

    # Class inheritance removed
    assert "class TestCalculator:" in result
    assert "IsolatedAsyncioWrapperTestCase" not in result
    assert "import unittest" not in result

    # setUp/tearDown converted
    assert "@pytest.fixture(autouse=True)" in result
    assert "def setup(self):" in result
    assert "def teardown_fixture(self):" in result
    assert "yield" in result

    # Assertions converted
    assert "assert self.calc.add(1, 2) == 3" in result
    assert "assert self.calc.is_ready" in result

    # assertRaises converted
    assert "with pytest.raises(ZeroDivisionError):" in result

    # assertAlmostEqual converted
    assert "pytest.approx" in result

    # Decorator converted
    assert "@pytest.mark.skip" in result
    assert "unittest.skip" not in result


def test_convert_skipunless_decorator():
    source = """\
        @unittest.skipUnless(HAS_FEATURE, "needs feature")
        def test_foo(self):
            pass
        """
    result = _decorators(source)
    assert "pytest.mark.skipif" in result
    assert "not HAS_FEATURE" in result
    assert "reason=" in result
    assert "@unittest.skipUnless" not in result


# ---------------------------------------------------------------------------
# Issue #10: AddPytestImport tests
# ---------------------------------------------------------------------------


def test_add_pytest_import():
    """After converting setUp and assertRaises, 'import pytest' should be present."""
    source = textwrap.dedent("""\
        import unittest

        class TestFoo(unittest.TestCase):
            def setUp(self):
                self.value = 42

            def test_raises(self):
                with self.assertRaises(ValueError):
                    raise ValueError("oops")
    """)

    result = _run(
        RemoveUnittestInheritance,
        RemoveUnittestImport,
        ConvertSetUpTearDown,
        ConvertAssertRaises,
        AddPytestImport,
        source=source,
    )

    assert "import pytest" in result
    assert "pytest.fixture" in result
    assert "pytest.raises" in result


def test_add_pytest_import_not_added_when_already_present():
    """AddPytestImport must not duplicate an existing 'import pytest'."""
    source = textwrap.dedent("""\
        import pytest
        import unittest

        class TestFoo(unittest.TestCase):
            def test_raises(self):
                with self.assertRaises(ValueError):
                    raise ValueError("oops")
    """)

    result = _run(
        RemoveUnittestInheritance,
        RemoveUnittestImport,
        ConvertAssertRaises,
        AddPytestImport,
        source=source,
    )

    assert result.count("import pytest") == 1


def test_add_pytest_import_not_added_when_no_pytest_refs():
    """AddPytestImport must not fire when there are no pytest references."""
    source = textwrap.dedent("""\
        import unittest

        class TestFoo(unittest.TestCase):
            def test_eq(self):
                self.assertEqual(1, 1)
    """)

    # Only run RemoveUnittestInheritance + ConvertAssertions (no pytest refs generated)
    result = _run(
        RemoveUnittestInheritance,
        RemoveUnittestImport,
        ConvertAssertions,
        AddPytestImport,
        source=source,
    )

    assert "import pytest" not in result


# ---------------------------------------------------------------------------
# Issue #15: RemoveUnittestInheritance mixin __init__ tests
# ---------------------------------------------------------------------------


def test_keep_testcase_when_mixin_present():
    """When mixin bases remain, the class is NOT modified to avoid pytest TypeError."""
    source = textwrap.dedent("""\
        from unittest import TestCase

        class TestFoo(SomeMixin, TestCase):
            def test_something(self):
                pass
    """)

    result = _run(RemoveUnittestInheritance, source=source)

    # Class is left entirely unchanged: both bases kept, no __init__ = None injected
    assert "class TestFoo(SomeMixin, TestCase):" in result
    assert "__init__ = None" not in result


def test_keep_testcase_when_mixin_and_own_init_present():
    """When mixin bases remain (even with own __init__), the class is NOT modified."""
    source = textwrap.dedent("""\
        from unittest import TestCase

        class TestFoo(SomeMixin, TestCase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def test_something(self):
                pass
    """)

    result = _run(RemoveUnittestInheritance, source=source)

    # Class is left entirely unchanged
    assert "class TestFoo(SomeMixin, TestCase):" in result
    assert "__init__ = None" not in result


def test_no_mixin_init_override_for_pure_testcase():
    """Pure TestCase inheritance (no other bases) must NOT get __init__ = None."""
    source = textwrap.dedent("""\
        from unittest import TestCase

        class TestFoo(TestCase):
            def test_something(self):
                pass
    """)

    result = _run(RemoveUnittestInheritance, source=source)

    assert "class TestFoo:" in result
    assert "__init__ = None" not in result


# ---------------------------------------------------------------------------
# Issue #27: assertIsNone / assertIsNotNone operator tests
# ---------------------------------------------------------------------------


def test_assert_is_none_uses_is():
    """assertIsNone(x) must produce 'assert x is None', not 'assert x == None'."""
    result = _assert("self.assertIsNone(x)\n")
    assert result == "assert x is None\n"


def test_assert_is_not_none_uses_is_not():
    """assertIsNotNone(x) must produce 'assert x is not None', not 'assert x != None'."""
    result = _assert("self.assertIsNotNone(x)\n")
    assert result == "assert x is not None\n"


# ---------------------------------------------------------------------------
# Issue #26: super().asyncSetUp() removal tests
# ---------------------------------------------------------------------------


def test_remove_super_async_setup():
    """await super().asyncSetUp() must be removed by ConvertSetUpTearDown."""
    source = """\
        async def asyncSetUp(self):
            await super().asyncSetUp()
            self.foo = 1
        """
    result = _setup(source)
    assert "super().asyncSetUp()" not in result
    assert "self.foo = 1" in result


# ---------------------------------------------------------------------------
# Issue #25: cm.exception -> cm.value tests
# ---------------------------------------------------------------------------


def test_convert_exception_to_value():
    """cm.exception must be converted to cm.value."""
    source = """\
        with self.assertRaises(ValueError) as cm:
            do_something()
        assert cm.exception is not None
        """
    result = _run(ConvertAssertRaises, ConvertExceptionToValue, source=textwrap.dedent(source))
    assert "cm.value" in result
    assert "cm.exception" not in result
    assert "pytest.raises(ValueError)" in result
