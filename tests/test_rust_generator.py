from __future__ import annotations

import ast
import textwrap

import pytest

from refactor.rules.rust_generator import (
    generate_cargo_toml,
    generate_module,
    generate_pyclass,
    generate_pyfunction,
    python_type_to_rust,
)


# ---------------------------------------------------------------------------
# python_type_to_rust
# ---------------------------------------------------------------------------


def test_type_map_int():
    node = ast.Name(id="int", ctx=ast.Load())
    assert python_type_to_rust(node) == "i64"


def test_type_map_float():
    node = ast.Name(id="float", ctx=ast.Load())
    assert python_type_to_rust(node) == "f64"


def test_type_map_str():
    node = ast.Name(id="str", ctx=ast.Load())
    assert python_type_to_rust(node) == "String"


def test_type_map_bool():
    node = ast.Name(id="bool", ctx=ast.Load())
    assert python_type_to_rust(node) == "bool"


def test_type_map_bytes():
    node = ast.Name(id="bytes", ctx=ast.Load())
    assert python_type_to_rust(node) == "Vec<u8>"


def test_type_map_none():
    node = ast.Name(id="None", ctx=ast.Load())
    assert python_type_to_rust(node) == "()"


def test_type_map_any():
    node = ast.Name(id="Any", ctx=ast.Load())
    assert python_type_to_rust(node) == "PyObject"


def test_type_map_unknown():
    node = ast.Name(id="SomeCustomClass", ctx=ast.Load())
    assert python_type_to_rust(node) == "PyObject"


def test_type_map_list_int():
    # list[int] → Vec<i64>
    tree = ast.parse("def f(x: list[int]): pass")
    ann = tree.body[0].args.args[0].annotation
    assert python_type_to_rust(ann) == "Vec<i64>"


def test_type_map_list_str():
    tree = ast.parse("def f(x: list[str]): pass")
    ann = tree.body[0].args.args[0].annotation
    assert python_type_to_rust(ann) == "Vec<String>"


def test_type_map_dict():
    # dict[str, int] → HashMap<String, i64>
    tree = ast.parse("def f(x: dict[str, int]): pass")
    ann = tree.body[0].args.args[0].annotation
    assert python_type_to_rust(ann) == "HashMap<String, i64>"


def test_type_map_set():
    tree = ast.parse("def f(x: set[float]): pass")
    ann = tree.body[0].args.args[0].annotation
    assert python_type_to_rust(ann) == "HashSet<f64>"


def test_type_map_tuple():
    tree = ast.parse("def f(x: tuple[int, str]): pass")
    ann = tree.body[0].args.args[0].annotation
    result = python_type_to_rust(ann)
    assert "i64" in result
    assert "String" in result


def test_type_map_optional():
    # Optional[float] → Option<f64>
    tree = ast.parse("from typing import Optional\ndef f(x: Optional[float]): pass")
    ann = tree.body[1].args.args[0].annotation
    assert python_type_to_rust(ann) == "Option<f64>"


def test_type_map_optional_subscript():
    # typing.Optional[str] via Attribute subscript
    source = textwrap.dedent("""\
        import typing
        def f(x: typing.Optional[str]): pass
    """)
    tree = ast.parse(source)
    ann = tree.body[1].args.args[0].annotation
    assert python_type_to_rust(ann) == "Option<String>"


def test_type_map_nested_generics():
    # list[dict[str, list[int]]] → Vec<HashMap<String, Vec<i64>>>
    tree = ast.parse("def f(x: list[dict[str, list[int]]]): pass")
    ann = tree.body[0].args.args[0].annotation
    assert python_type_to_rust(ann) == "Vec<HashMap<String, Vec<i64>>>"


def test_type_map_none_constant():
    node = ast.Constant(value=None)
    assert python_type_to_rust(node) == "()"


def test_type_map_union_pep604():
    # int | None → Option<i64>
    tree = ast.parse("def f(x: int | None): pass")
    ann = tree.body[0].args.args[0].annotation
    assert python_type_to_rust(ann) == "Option<i64>"


# ---------------------------------------------------------------------------
# generate_pyfunction
# ---------------------------------------------------------------------------


def test_generate_simple_function():
    source = "def add(a: int, b: int) -> int:\n    return a + b"
    tree = ast.parse(source)
    func = tree.body[0]
    result = generate_pyfunction(func)
    assert "#[pyfunction]" in result
    assert "fn add(a: i64, b: i64) -> PyResult<i64>" in result
    assert "todo!" in result


def test_generate_function_with_docstring():
    source = textwrap.dedent("""\
        def calc(x: float) -> float:
            \"\"\"Calculate something.\"\"\"
            return x
    """)
    tree = ast.parse(source)
    func = tree.body[0]
    result = generate_pyfunction(func)
    assert "/// Calculate something." in result
    assert "#[pyfunction]" in result
    assert "fn calc" in result


def test_generate_function_no_annotations():
    source = "def foo(x, y):\n    pass"
    tree = ast.parse(source)
    func = tree.body[0]
    result = generate_pyfunction(func)
    assert "#[pyfunction]" in result
    assert "fn foo" in result
    assert "PyObject" in result


def test_generate_function_no_return():
    source = "def nothing(x: int):\n    pass"
    tree = ast.parse(source)
    func = tree.body[0]
    result = generate_pyfunction(func)
    assert "PyResult<()>" in result


def test_generate_function_skips_self():
    source = textwrap.dedent("""\
        class Foo:
            def method(self, x: int) -> str:
                pass
    """)
    tree = ast.parse(source)
    method = tree.body[0].body[0]
    result = generate_pyfunction(method)
    assert "self" not in result
    assert "x: i64" in result


def test_generate_function_with_default():
    source = "def greet(name: str, times: int = 1) -> str:\n    pass"
    tree = ast.parse(source)
    func = tree.body[0]
    result = generate_pyfunction(func)
    assert "Option<i64>" in result
    assert "name: String" in result


def test_generate_async_function():
    # async def should still generate (PyO3 supports async via pyo3-asyncio)
    source = "async def fetch(url: str) -> str:\n    pass"
    tree = ast.parse(source)
    result = generate_pyfunction(tree.body[0])
    assert "fn fetch" in result
    assert "String" in result


# ---------------------------------------------------------------------------
# generate_pyclass
# ---------------------------------------------------------------------------


def test_generate_class():
    source = textwrap.dedent("""\
        class Point:
            def __init__(self, x: float, y: float):
                self.x = x
                self.y = y
            def distance(self) -> float:
                return (self.x**2 + self.y**2)**0.5
    """)
    tree = ast.parse(source)
    cls = tree.body[0]
    result = generate_pyclass(cls)
    assert "#[pyclass]" in result
    assert "struct Point" in result
    assert "x: f64" in result
    assert "y: f64" in result
    assert "#[pymethods]" in result
    assert "#[new]" in result
    assert "fn distance(&self)" in result


def test_generate_class_with_docstring():
    source = textwrap.dedent("""\
        class Calculator:
            \"\"\"A simple calculator.\"\"\"
            def __init__(self, precision: int = 2):
                self.precision = precision
            def add(self, a: float, b: float) -> float:
                return a + b
    """)
    tree = ast.parse(source)
    cls = tree.body[0]
    result = generate_pyclass(cls)
    assert "/// A simple calculator." in result
    assert "#[pyclass]" in result
    assert "precision: i64" in result
    assert "fn add(&self, a: f64, b: f64) -> PyResult<f64>" in result


def test_generate_class_init_with_default_wraps_option():
    source = textwrap.dedent("""\
        class Counter:
            def __init__(self, start: int = 0):
                self.count = start
    """)
    tree = ast.parse(source)
    cls = tree.body[0]
    result = generate_pyclass(cls)
    assert "Option<i64>" in result


def test_generate_class_with_typed_list_field():
    source = textwrap.dedent("""\
        class History:
            def __init__(self):
                self.items: list[float] = []
    """)
    tree = ast.parse(source)
    cls = tree.body[0]
    result = generate_pyclass(cls)
    assert "items: Vec<f64>" in result


def test_generate_class_no_init():
    source = textwrap.dedent("""\
        class Empty:
            def do_thing(self) -> int:
                pass
    """)
    tree = ast.parse(source)
    cls = tree.body[0]
    result = generate_pyclass(cls)
    assert "#[pyclass]" in result
    assert "struct Empty {" in result
    assert "fn do_thing(&self)" in result


# ---------------------------------------------------------------------------
# generate_module
# ---------------------------------------------------------------------------


def test_generate_full_module():
    source = textwrap.dedent("""\
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        class Counter:
            def __init__(self, start: int = 0):
                self.count = start
            def increment(self) -> int:
                self.count += 1
                return self.count
    """)
    result = generate_module(source, "mymod")
    assert "use pyo3::prelude::*;" in result
    assert "#[pymodule]" in result
    assert "fn mymod" in result
    assert "wrap_pyfunction!(greet" in result
    assert "add_class::<Counter>" in result


def test_generate_module_uses_hashmap():
    source = "def f(x: dict[str, int]) -> dict[str, int]:\n    pass"
    result = generate_module(source, "mymod")
    assert "use std::collections::HashMap;" in result


def test_generate_module_uses_hashset():
    source = "def f(x: set[int]) -> set[int]:\n    pass"
    result = generate_module(source, "mymod")
    assert "use std::collections::HashSet;" in result


def test_generate_module_no_hashmap_if_unused():
    source = "def f(x: int) -> str:\n    pass"
    result = generate_module(source, "mymod")
    assert "HashMap" not in result


def test_generate_module_ok_no_return():
    source = "def f() -> None:\n    pass"
    result = generate_module(source, "mod")
    assert "Ok(())" in result


def test_generate_module_async_functions():
    # async def should still generate (PyO3 supports async via pyo3-asyncio)
    source = "async def fetch(url: str) -> str:\n    pass"
    tree = ast.parse(source)
    result = generate_pyfunction(tree.body[0])
    assert "fn fetch" in result


# ---------------------------------------------------------------------------
# generate_cargo_toml
# ---------------------------------------------------------------------------


def test_generate_cargo_toml():
    result = generate_cargo_toml("mymod")
    assert 'name = "mymod"' in result
    assert "pyo3" in result
    assert "cdylib" in result


def test_generate_cargo_toml_version():
    result = generate_cargo_toml("mymod", version="0.2.1")
    assert 'version = "0.2.1"' in result


def test_generate_cargo_toml_edition():
    result = generate_cargo_toml("mymod")
    assert 'edition = "2021"' in result


def test_generate_cargo_toml_lib_name():
    result = generate_cargo_toml("my_lib")
    assert '[lib]' in result
    assert 'name = "my_lib"' in result


def test_generate_cargo_toml_extension_module():
    result = generate_cargo_toml("mymod")
    assert "extension-module" in result
