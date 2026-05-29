from __future__ import annotations

import textwrap

from refactor.rules.rust_scaffold import (
    generate_implementation_prompt,
    scaffold_rust_project,
    write_scaffold,
)

# ---------------------------------------------------------------------------
# generate_implementation_prompt
# ---------------------------------------------------------------------------


def test_implementation_prompt_has_type_reference():
    source = "def add(a: int, b: int) -> int:\n    return a + b"
    result = generate_implementation_prompt(source, "math_utils")
    assert "Type Reference" in result
    assert "i64" in result


def test_implementation_prompt_has_python_source():
    source = "def add(a: int, b: int) -> int:\n    return a + b"
    result = generate_implementation_prompt(source, "math_utils")
    assert "Original Python" in result
    assert "return a + b" in result


def test_implementation_prompt_has_rust_skeleton():
    source = "def add(a: int, b: int) -> int:\n    return a + b"
    result = generate_implementation_prompt(source, "math_utils")
    assert "#[pyfunction]" in result
    assert "fn add" in result


def test_implementation_prompt_flags_dynamic():
    source = "def fetch(url: str, **kwargs) -> str:\n    return getattr(obj, url)"
    result = generate_implementation_prompt(source, "fetcher")
    assert "kwargs" in result.lower() or "dynamic" in result.lower()


def test_implementation_prompt_module_name_in_header():
    source = "def noop() -> None:\n    pass"
    result = generate_implementation_prompt(source, "my_module")
    assert "my_module" in result


def test_implementation_prompt_includes_class():
    source = textwrap.dedent("""\
        class Box:
            def __init__(self, value: int) -> None:
                self.value = value
            def get(self) -> int:
                return self.value
    """)
    result = generate_implementation_prompt(source, "containers")
    assert "struct Box" in result
    assert "Classes to Implement" in result


def test_implementation_prompt_edge_cases_section():
    source = "def add(a: int, b: int) -> int:\n    return a + b"
    result = generate_implementation_prompt(source, "math_utils")
    assert "Edge Cases" in result


def test_implementation_prompt_no_unsafe():
    source = "def add(a: int, b: int) -> int:\n    return a + b"
    result = generate_implementation_prompt(source, "math_utils")
    assert "unsafe" in result  # warns that none are needed


def test_implementation_prompt_varargs_flagged():
    source = "def variadic(*args: int) -> int:\n    return sum(args)"
    result = generate_implementation_prompt(source, "varmod")
    # vararg detected as dynamic pattern
    assert "args" in result


# ---------------------------------------------------------------------------
# scaffold_rust_project
# ---------------------------------------------------------------------------


def test_scaffold_has_all_files():
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
    result = scaffold_rust_project(source, "mymod")
    assert "src/lib.rs" in result
    assert "Cargo.toml" in result
    assert "IMPLEMENTATION_GUIDE.md" in result

    # Verify content
    assert "use pyo3::prelude::*;" in result["src/lib.rs"]
    assert 'name = "mymod"' in result["Cargo.toml"]
    assert "fn greet" in result["src/lib.rs"]


def test_scaffold_lib_rs_valid():
    source = "def compute(x: float, y: float) -> float:\n    return x * y"
    result = scaffold_rust_project(source, "compute_mod")
    lib_rs = result["src/lib.rs"]
    assert "#[pymodule]" in lib_rs
    assert "fn compute_mod" in lib_rs


def test_scaffold_cargo_toml_valid():
    source = "def noop() -> None:\n    pass"
    result = scaffold_rust_project(source, "test_mod")
    cargo = result["Cargo.toml"]
    assert "pyo3" in cargo
    assert "cdylib" in cargo


def test_scaffold_readme_present():
    source = "def noop() -> None:\n    pass"
    result = scaffold_rust_project(source, "mymod")
    assert "README.md" in result
    assert "mymod" in result["README.md"]


def test_scaffold_python_wrapper_present():
    source = "def noop() -> None:\n    pass"
    result = scaffold_rust_project(source, "mymod")
    assert "python_wrapper.py" in result


def test_scaffold_implementation_guide_content():
    source = "def add(a: int, b: int) -> int:\n    return a + b"
    result = scaffold_rust_project(source, "addmod")
    guide = result["IMPLEMENTATION_GUIDE.md"]
    assert "Type Reference" in guide
    assert "fn add" in guide


def test_scaffold_class_in_lib_rs():
    source = textwrap.dedent("""\
        class Point:
            def __init__(self, x: float, y: float) -> None:
                self.x = x
                self.y = y
    """)
    result = scaffold_rust_project(source, "geometry")
    lib_rs = result["src/lib.rs"]
    assert "struct Point" in lib_rs
    assert "#[pyclass]" in lib_rs


def test_scaffold_returns_dict_of_strings():
    source = "def noop() -> None:\n    pass"
    result = scaffold_rust_project(source, "check")
    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


# ---------------------------------------------------------------------------
# write_scaffold
# ---------------------------------------------------------------------------


def test_write_scaffold_creates_files(tmp_path):
    source = "def add(a: int, b: int) -> int:\n    return a + b"
    scaffold = scaffold_rust_project(source, "mymod")
    write_scaffold(scaffold, str(tmp_path))

    for rel_path in scaffold:
        full_path = tmp_path / rel_path
        assert full_path.exists(), f"Expected {rel_path} to exist"
        assert full_path.read_text() == scaffold[rel_path]


def test_write_scaffold_creates_src_dir(tmp_path):
    source = "def noop() -> None:\n    pass"
    scaffold = scaffold_rust_project(source, "mymod")
    write_scaffold(scaffold, str(tmp_path))
    assert (tmp_path / "src").is_dir()
    assert (tmp_path / "src" / "lib.rs").exists()
