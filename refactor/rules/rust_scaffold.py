from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from refactor.rules.rust_generator import (
    generate_cargo_toml,
    generate_module,
    generate_pyfunction,
    generate_pyclass,
)

# ---------------------------------------------------------------------------
# Minimal inline wrapper generator (used if rust_orchestrator is unavailable)
# ---------------------------------------------------------------------------

def _generate_python_wrapper_inline(source: str, module_name: str) -> str:
    """Minimal Python wrapper that tries the Rust extension, falls back to pure Python."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    all_names = ", ".join(names)
    return textwrap.dedent(f"""\
        # Auto-generated Python wrapper for {module_name}
        # Tries to import the compiled Rust extension; falls back to pure Python.
        try:
            from ._{module_name}_rs import {all_names}  # noqa: F401
        except ImportError:
            # Rust extension not built yet — falling back to pure Python implementation.
            from .{module_name}_pure import {all_names}  # noqa: F401
        """)


def _try_import_wrapper(source: str, module_name: str) -> str:
    """Import generate_python_wrapper from rust_orchestrator if available."""
    try:
        from refactor.rules.rust_orchestrator import generate_python_wrapper  # type: ignore[import]
        return generate_python_wrapper(source, module_name)
    except ImportError:
        return _generate_python_wrapper_inline(source, module_name)


# ---------------------------------------------------------------------------
# Helper: detect dynamic patterns in a function/method
# ---------------------------------------------------------------------------

def _has_dynamic_patterns(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return a list of dynamic pattern descriptions found in the function."""
    issues: list[str] = []
    if func.args.vararg:
        issues.append(f"`*{func.args.vararg.arg}` (var-positional args)")
    if func.args.kwarg:
        issues.append(f"`**{func.args.kwarg.arg}` (keyword args — requires manual adaptation)")
    # Detect getattr / setattr / eval / exec calls
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "setattr", "eval", "exec"):
                issues.append(f"`{node.func.id}()` call (dynamic attribute access)")
                break
    return issues


def _body_complexity_notes(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return implementation hints based on the function body."""
    notes: list[str] = []
    body = func.body
    # Skip pure docstring bodies
    effective = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]

    if len(effective) == 0:
        notes.append("Empty body — implement as no-op or return default value")
        return notes

    if len(effective) == 1:
        stmt = effective[0]
        if isinstance(stmt, ast.Return):
            notes.append("Direct arithmetic/value translation")
            notes.append("No error handling needed (pure computation)")
            return notes

    # Check for loops
    has_loop = any(isinstance(s, (ast.For, ast.While)) for s in ast.walk(func))
    if has_loop:
        notes.append("Contains loop — translate to Rust iterator or explicit loop")

    # Check for try/except
    has_try = any(isinstance(s, ast.Try) for s in ast.walk(func))
    if has_try:
        notes.append("Contains exception handling — map to `Result`/`PyErr` in Rust")

    # Check for comprehensions
    has_comp = any(isinstance(s, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)) for s in ast.walk(func))
    if has_comp:
        notes.append("Contains comprehension — translate to `.iter().map().collect()`")

    if not notes:
        notes.append("General translation — implement step by step")

    return notes


# ---------------------------------------------------------------------------
# TYPE_REFERENCE table
# ---------------------------------------------------------------------------

_TYPE_REFERENCE = [
    ("int", "i64", ""),
    ("float", "f64", ""),
    ("str", "String", "Use `&str` for borrowed"),
    ("bool", "bool", ""),
    ("bytes", "Vec<u8>", ""),
    ("None", "()", "Return type only"),
    ("list[T]", "Vec<T>", "e.g. `list[int]` → `Vec<i64>`"),
    ("dict[K, V]", "HashMap<K, V>", "Requires `use std::collections::HashMap`"),
    ("set[T]", "HashSet<T>", "Requires `use std::collections::HashSet`"),
    ("tuple[A, B]", "(A, B,)", ""),
    ("Optional[T]", "Option<T>", "Also `T | None`"),
    ("Any", "PyObject", "Dynamic — avoid if possible"),
]


def _type_reference_table() -> str:
    rows = ["| Python | Rust | Notes |", "|--------|------|-------|"]
    for py, rs, note in _TYPE_REFERENCE:
        rows.append(f"| `{py}` | `{rs}` | {note} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Component 1: AgentPromptGenerator
# ---------------------------------------------------------------------------

def generate_implementation_prompt(source: str, module_name: str) -> str:
    """Generate a detailed prompt for Claude to implement Rust logic."""
    tree = ast.parse(source)

    sections: list[str] = []

    # Header
    sections.append(f"# Rust Implementation Guide: {module_name}\n")

    # Type reference
    sections.append("## Type Reference\n")
    sections.append(_type_reference_table())
    sections.append("")

    # Collect warnings for dynamic patterns
    warnings: list[str] = []

    # Functions section
    func_nodes = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if func_nodes:
        sections.append("## Functions to Implement\n")
        for func in func_nodes:
            rust_skeleton = generate_pyfunction(func)
            # Extract the fn signature line for the heading
            sig_line = next(
                (ln.strip() for ln in rust_skeleton.splitlines() if ln.strip().startswith("fn ")),
                f"fn {func.name}(...)",
            )
            sections.append(f"### `{sig_line}`\n")

            # Original Python
            py_src = ast.get_source_segment(source, func) or "(source unavailable)"
            sections.append("**Original Python:**")
            sections.append(f"```python\n{py_src}\n```\n")

            # Rust skeleton
            sections.append("**Rust skeleton:**")
            sections.append(f"```rust\n{rust_skeleton}\n```\n")

            # Implementation notes
            notes = _body_complexity_notes(func)
            dynamic = _has_dynamic_patterns(func)
            sections.append("**Implementation notes:**")
            for note in notes:
                sections.append(f"- {note}")
            for d in dynamic:
                sections.append(f"- Dynamic pattern detected: {d}")
                warnings.append(f"Function `{func.name}` uses {d}")
            sections.append("")

    # Classes section
    class_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if class_nodes:
        sections.append("## Classes to Implement\n")
        for cls in class_nodes:
            sections.append(f"### `struct {cls.name}`\n")

            # Collect fields and methods
            from refactor.rules.rust_generator import _extract_init_fields  # noqa: PLC0415
            fields = _extract_init_fields(cls)
            field_str = ", ".join(f"{n}: {t}" for n, t in fields) if fields else "(none detected)"
            method_names = [
                n.name for n in cls.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            sections.append(f"**Fields:** {field_str}")
            sections.append(f"**Methods:** {', '.join(method_names) if method_names else '(none)'}\n")

            py_src = ast.get_source_segment(source, cls) or "(source unavailable)"
            sections.append("**Original Python:**")
            sections.append(f"```python\n{py_src}\n```\n")

            rust_skeleton = generate_pyclass(cls)
            sections.append("**Rust skeleton:**")
            sections.append(f"```rust\n{rust_skeleton}\n```\n")

            # Per-method dynamic warnings
            for node in cls.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dynamic = _has_dynamic_patterns(node)
                    for d in dynamic:
                        warnings.append(f"Method `{cls.name}::{node.name}` uses {d}")

    # Edge cases & warnings
    sections.append("## Edge Cases & Warnings\n")
    if warnings:
        for w in warnings:
            sections.append(f"- {w}")
    else:
        sections.append("- No dynamic patterns detected")
    sections.append("- No `unsafe` blocks should be needed for this module")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Component 2: ProjectScaffolder
# ---------------------------------------------------------------------------

def _generate_readme(module_name: str) -> str:
    return textwrap.dedent(f"""\
        # {module_name}

        Rust-accelerated Python module generated by `rust_scaffold`.

        ## Building

        ```bash
        maturin develop          # development build
        maturin build --release  # release wheel
        ```

        ## Usage

        ```python
        import {module_name}
        ```

        ## Development

        After building, the compiled extension is importable as `_{module_name}_rs`.
        The `python_wrapper.py` file provides a transparent fallback to a pure-Python
        implementation while the Rust code is being developed.

        See `IMPLEMENTATION_GUIDE.md` for instructions on filling in the `todo!()` bodies.
        """)


def scaffold_rust_project(
    source: str,
    module_name: str,
    output_dir: str | None = None,
) -> dict[str, str]:
    """Generate complete Rust project scaffold from Python source.

    Returns dict mapping file paths (relative) to file contents:
    {
        "src/lib.rs": "use pyo3::prelude::*; ...",
        "Cargo.toml": "[package] ...",
        "python_wrapper.py": "try: from ._mod_rs import ...",
        "IMPLEMENTATION_GUIDE.md": "# Rust Implementation Guide ...",
        "README.md": "# module_name\\n\\nRust-accelerated ...",
    }
    """
    scaffold: dict[str, str] = {
        "src/lib.rs": generate_module(source, module_name),
        "Cargo.toml": generate_cargo_toml(module_name),
        "python_wrapper.py": _try_import_wrapper(source, module_name),
        "IMPLEMENTATION_GUIDE.md": generate_implementation_prompt(source, module_name),
        "README.md": _generate_readme(module_name),
    }
    return scaffold


def write_scaffold(scaffold: dict[str, str], output_dir: str) -> None:
    """Write all scaffold files to disk under output_dir."""
    base = Path(output_dir)
    for rel_path, content in scaffold.items():
        full_path = base / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    source_file = sys.argv[1]
    module_name = sys.argv[2] if len(sys.argv) > 2 else Path(source_file).stem
    with open(source_file) as f:
        source = f.read()
    scaffold = scaffold_rust_project(source, module_name)
    for path, content in scaffold.items():
        print(f"=== {path} ===")
        print(content[:200] + "..." if len(content) > 200 else content)
        print()
