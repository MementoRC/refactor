from __future__ import annotations

import ast
import textwrap
from typing import Union

# Python annotation → Rust type (PyO3-compatible)
TYPE_MAP: dict[str, str] = {
    # Primitives
    "int": "i64",
    "float": "f64",
    "str": "String",
    "bool": "bool",
    "bytes": "Vec<u8>",
    "complex": "(f64, f64)",
    "None": "()",
    # Containers (simple, unparameterized)
    "list": "Vec<PyObject>",
    "dict": "HashMap<PyObject, PyObject>",
    "set": "HashSet<PyObject>",
    "tuple": "(PyObject,)",
    # Special
    "Any": "PyObject",
    "object": "PyObject",
}


def python_type_to_rust(annotation: ast.expr) -> str:
    """Convert a Python type annotation AST node to a Rust type string."""
    if annotation is None:
        return "PyObject"

    if isinstance(annotation, ast.Name):
        return TYPE_MAP.get(annotation.id, "PyObject")

    if isinstance(annotation, ast.Constant):
        # e.g. None literal
        if annotation.value is None:
            return "()"
        return "PyObject"

    if isinstance(annotation, ast.Attribute):
        # e.g. typing.Optional, typing.List, etc.
        attr = annotation.attr
        return TYPE_MAP.get(attr, "PyObject")

    if isinstance(annotation, ast.Subscript):
        return _handle_subscript(annotation)

    # BinOp handles `X | Y` (PEP 604 union syntax, Python 3.10+)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        left = python_type_to_rust(annotation.left)
        right = python_type_to_rust(annotation.right)
        # If right is "()" this is an Optional equivalent
        if right == "()":
            return f"Option<{left}>"
        if left == "()":
            return f"Option<{right}>"
        return "PyObject"

    return "PyObject"


def _get_name(node: ast.expr) -> str:
    """Extract simple name string from Name or Attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _handle_subscript(node: ast.Subscript) -> str:
    """Handle generic subscript types like list[int], Optional[X], etc."""
    container_name = _get_name(node.value)
    slice_node = node.slice

    # Unwrap Index node (Python 3.8 compatibility)
    if isinstance(slice_node, ast.Index):
        slice_node = slice_node.value  # type: ignore[attr-defined]

    if container_name in ("Optional",):
        inner = python_type_to_rust(slice_node)
        return f"Option<{inner}>"

    if container_name == "Union":
        # Union[X, None] → Option<X>
        elts = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
        non_none = [e for e in elts if not (_is_none(e))]
        if len(non_none) == 1 and len(elts) == 2:
            inner = python_type_to_rust(non_none[0])
            return f"Option<{inner}>"
        return "PyObject"

    if container_name == "list":
        inner = python_type_to_rust(slice_node)
        return f"Vec<{inner}>"

    if container_name == "set":
        inner = python_type_to_rust(slice_node)
        return f"HashSet<{inner}>"

    if container_name == "dict":
        if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
            k = python_type_to_rust(slice_node.elts[0])
            v = python_type_to_rust(slice_node.elts[1])
            return f"HashMap<{k}, {v}>"
        return "HashMap<PyObject, PyObject>"

    if container_name == "tuple":
        if isinstance(slice_node, ast.Tuple):
            parts = ", ".join(python_type_to_rust(e) for e in slice_node.elts)
            return f"({parts},)"
        inner = python_type_to_rust(slice_node)
        return f"({inner},)"

    if container_name in ("List",):
        inner = python_type_to_rust(slice_node)
        return f"Vec<{inner}>"

    if container_name in ("Dict",):
        if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
            k = python_type_to_rust(slice_node.elts[0])
            v = python_type_to_rust(slice_node.elts[1])
            return f"HashMap<{k}, {v}>"
        return "HashMap<PyObject, PyObject>"

    if container_name in ("Set",):
        inner = python_type_to_rust(slice_node)
        return f"HashSet<{inner}>"

    if container_name in ("Tuple",):
        if isinstance(slice_node, ast.Tuple):
            parts = ", ".join(python_type_to_rust(e) for e in slice_node.elts)
            return f"({parts},)"
        inner = python_type_to_rust(slice_node)
        return f"({inner},)"

    return "PyObject"


def _is_none(node: ast.expr) -> bool:
    """Return True if node represents None."""
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name) and node.id == "None":
        return True
    return False


def _extract_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> str | None:
    """Extract docstring from a function or class node."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value
    return None


def _has_default(func: Union[ast.FunctionDef, ast.AsyncFunctionDef], param_name: str) -> bool:
    """Check whether a parameter has a default value."""
    args = func.args
    all_params = args.args + args.kwonlyargs
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        if arg.arg == param_name:
            default_idx = i - defaults_offset
            return default_idx >= 0
    for i, arg in enumerate(args.kwonlyargs):
        if arg.arg == param_name:
            kw_default = args.kw_defaults[i]
            return kw_default is not None
    return False


def generate_pyfunction(func: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    """Generate a #[pyfunction] Rust function from a Python function def."""
    lines: list[str] = []

    # Docstring as /// comment
    docstring = _extract_docstring(func)
    if docstring:
        for doc_line in docstring.strip().splitlines():
            lines.append(f"/// {doc_line.strip()}")

    lines.append("#[pyfunction]")

    # Build parameter list, skipping self/cls
    params: list[str] = []
    for arg in func.args.args:
        if arg.arg in ("self", "cls"):
            continue
        rust_type = python_type_to_rust(arg.annotation) if arg.annotation else "PyObject"
        # Wrap in Option if has default
        if _has_default(func, arg.arg):
            if not rust_type.startswith("Option<"):
                rust_type = f"Option<{rust_type}>"
        params.append(f"{arg.arg}: {rust_type}")

    # Return type
    if func.returns:
        ret = python_type_to_rust(func.returns)
    else:
        ret = "()"

    ret_wrapped = f"PyResult<{ret}>"
    param_str = ", ".join(params)
    lines.append(f"fn {func.name}({param_str}) -> {ret_wrapped} {{")
    lines.append(f'    todo!("Implement: {func.name}")')
    lines.append("}")

    return "\n".join(lines)


def _extract_init_fields(cls: ast.ClassDef) -> list[tuple[str, str]]:
    """
    Extract (field_name, rust_type) pairs from self.attr assignments in __init__.

    Prefers type annotations on assignments (self.x: int = ...) but falls back
    to the __init__ parameter annotation for the same name.
    """
    init_method: ast.FunctionDef | None = None
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            init_method = node  # type: ignore[assignment]
            break

    if init_method is None:
        return []

    # Build param annotation map (excluding self)
    param_annotations: dict[str, ast.expr | None] = {}
    for arg in init_method.args.args:
        if arg.arg == "self":
            continue
        param_annotations[arg.arg] = arg.annotation

    fields: list[tuple[str, str]] = []
    seen: set[str] = set()

    for stmt in ast.walk(init_method):
        # self.x = ... or self.x: T = ...
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            if isinstance(stmt, ast.AnnAssign):
                target = stmt.target
                annotation = stmt.annotation
            else:
                if len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                annotation = None

            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"):
                continue

            field_name = target.attr
            if field_name in seen:
                continue
            seen.add(field_name)

            if annotation is not None:
                rust_type = python_type_to_rust(annotation)
            elif field_name in param_annotations and param_annotations[field_name] is not None:
                rust_type = python_type_to_rust(param_annotations[field_name])  # type: ignore[arg-type]
            else:
                rust_type = "PyObject"

            fields.append((field_name, rust_type))

    return fields


def generate_pyclass(cls: ast.ClassDef) -> str:
    """Generate a #[pyclass] Rust struct + #[pymethods] impl from a Python class def."""
    lines: list[str] = []

    # Class docstring
    docstring = _extract_docstring(cls)
    if docstring:
        for doc_line in docstring.strip().splitlines():
            lines.append(f"/// {doc_line.strip()}")

    lines.append("#[pyclass]")
    lines.append(f"struct {cls.name} {{")

    fields = _extract_init_fields(cls)
    for field_name, rust_type in fields:
        lines.append(f"    {field_name}: {rust_type},")

    lines.append("}")
    lines.append("")
    lines.append("#[pymethods]")
    lines.append(f"impl {cls.name} {{")

    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        method_lines: list[str] = []

        method_doc = _extract_docstring(node)
        if method_doc:
            for doc_line in method_doc.strip().splitlines():
                method_lines.append(f"    /// {doc_line.strip()}")

        if node.name == "__init__":
            method_lines.append("    #[new]")
            params: list[str] = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                rust_type = python_type_to_rust(arg.annotation) if arg.annotation else "PyObject"
                if _has_default(node, arg.arg):
                    if not rust_type.startswith("Option<"):
                        rust_type = f"Option<{rust_type}>"
                params.append(f"{arg.arg}: {rust_type}")
            param_str = ", ".join(params)
            method_lines.append(f"    fn new({param_str}) -> Self {{")
            method_lines.append(f'        todo!("Implement: {cls.name}::new")')
            method_lines.append("    }")
        elif node.name.startswith("__") and node.name.endswith("__"):
            # Skip other dunder methods (could be expanded later)
            continue
        else:
            params = []
            has_self = any(arg.arg == "self" for arg in node.args.args)
            if has_self:
                params.append("&self")
            for arg in node.args.args:
                if arg.arg in ("self", "cls"):
                    continue
                rust_type = python_type_to_rust(arg.annotation) if arg.annotation else "PyObject"
                if _has_default(node, arg.arg):
                    if not rust_type.startswith("Option<"):
                        rust_type = f"Option<{rust_type}>"
                params.append(f"{arg.arg}: {rust_type}")

            ret = python_type_to_rust(node.returns) if node.returns else "()"
            ret_wrapped = f"PyResult<{ret}>"
            param_str = ", ".join(params)
            method_lines.append(f"    fn {node.name}({param_str}) -> {ret_wrapped} {{")
            method_lines.append(f'        todo!("Implement: {cls.name}::{node.name}")')
            method_lines.append("    }")

        lines.extend(method_lines)

    lines.append("}")

    return "\n".join(lines)


def generate_module(source: str, module_name: str) -> str:
    """Parse Python source and generate a complete Rust module with PyO3 bindings."""
    tree = ast.parse(source)

    uses_hashmap = False
    uses_hashset = False

    function_names: list[str] = []
    class_names: list[str] = []
    body_parts: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rust_code = generate_pyfunction(node)
            body_parts.append(rust_code)
            function_names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            rust_code = generate_pyclass(node)
            body_parts.append(rust_code)
            class_names.append(node.name)

    # Detect HashMap/HashSet usage
    all_code = "\n".join(body_parts)
    if "HashMap" in all_code:
        uses_hashmap = True
    if "HashSet" in all_code:
        uses_hashset = True

    lines: list[str] = []
    lines.append("use pyo3::prelude::*;")
    if uses_hashmap:
        lines.append("use std::collections::HashMap;")
    if uses_hashset:
        lines.append("use std::collections::HashSet;")

    if body_parts:
        lines.append("")
        lines.append("\n\n".join(body_parts))

    lines.append("")
    lines.append("#[pymodule]")
    lines.append(f"fn {module_name}(m: &Bound<'_, PyModule>) -> PyResult<()> {{")
    for fn_name in function_names:
        lines.append(f"    m.add_function(wrap_pyfunction!({fn_name}, m)?)?;")
    for cls_name in class_names:
        lines.append(f"    m.add_class::<{cls_name}>()?;")
    lines.append("    Ok(())")
    lines.append("}")

    return "\n".join(lines)


def generate_cargo_toml(module_name: str, version: str = "0.1.0") -> str:
    """Generate a minimal Cargo.toml for a PyO3 crate."""
    return textwrap.dedent(f"""\
        [package]
        name = "{module_name}"
        version = "{version}"
        edition = "2021"

        [lib]
        name = "{module_name}"
        crate-type = ["cdylib"]

        [dependencies]
        pyo3 = {{ version = "0.22", features = ["extension-module"] }}
        """)
