"""AST-based documentation extraction engine.

Reads Python source files via AST and produces structured JSON suitable
for downstream template engines.  No runtime imports, no dynamic analysis.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_OPTIONS: dict[str, bool] = {
    "include_private": False,
    "include_dunder": False,
    "include_inherited": False,
}

# ---------------------------------------------------------------------------
# Docstring helpers
# ---------------------------------------------------------------------------

_PARAM_RE = re.compile(r":param\s+(\w+):\s*(.+?)(?=\n\s*:|$)", re.DOTALL)
_RETURNS_RE = re.compile(r":returns?:\s*(.+?)(?=\n\s*:|$)", re.DOTALL)


def _extract_docstring(body: list[ast.stmt]) -> str | None:
    """Extract a docstring from the first statement of a body."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value.value
    return None


def _parse_docstring_params(docstring: str) -> list[dict]:
    """Parse ``:param name: description`` entries from a docstring."""
    results: list[dict] = []
    for match in _PARAM_RE.finditer(docstring):
        results.append({"name": match.group(1), "description": match.group(2).strip()})
    return results


def _parse_docstring_returns(docstring: str) -> str | None:
    """Parse ``:returns:`` from a docstring."""
    match = _RETURNS_RE.search(docstring)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# AST annotation helpers
# ---------------------------------------------------------------------------


def _unparse_annotation(node: ast.expr | None) -> str | None:
    """Return a string representation of a type annotation node."""
    if node is None:
        return None
    return ast.unparse(node)


def _unparse_default(node: ast.expr | None) -> str | None:
    """Return a string representation of a default-value node."""
    if node is None:
        return None
    return ast.unparse(node)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _base_names(node: ast.ClassDef) -> list[str]:
    """Return simple string names for all bases of a class."""
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(ast.unparse(base))
    return names


def _decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return string representations of all decorators."""
    results: list[str] = []
    for dec in node.decorator_list:
        results.append(ast.unparse(dec))
    return results


def _is_pydantic_model(node: ast.ClassDef) -> bool:
    bases = _base_names(node)
    return any(b in ("BaseModel", "BaseClientModel") for b in bases)


def _is_protocol(node: ast.ClassDef) -> bool:
    bases = _base_names(node)
    return any(b == "Protocol" for b in bases)


def _is_enum(node: ast.ClassDef) -> bool:
    bases = _base_names(node)
    return any(b in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag") for b in bases)


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any("dataclass" in ast.unparse(d) for d in node.decorator_list)


def _is_abstract(node: ast.ClassDef) -> bool:
    bases = _base_names(node)
    if any(b in ("ABC", "ABCMeta") for b in bases):
        return True
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in item.decorator_list:
                if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
                    return True
                if isinstance(dec, ast.Attribute) and dec.attr == "abstractmethod":
                    return True
    return False


def _is_runtime_checkable(node: ast.ClassDef) -> bool:
    return any("runtime_checkable" in ast.unparse(d) for d in node.decorator_list)


# ---------------------------------------------------------------------------
# Pydantic / Enum field extraction
# ---------------------------------------------------------------------------


def _extract_pydantic_fields(node: ast.ClassDef) -> list[dict]:
    """Extract Pydantic model fields from annotated assignments."""
    fields: list[dict] = []
    # Collect validator names targeting specific fields
    validators: dict[str, list[str]] = {}
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in item.decorator_list:
                if isinstance(dec, ast.Call):
                    dec_name = ast.unparse(dec.func) if isinstance(dec, ast.Call) else ""
                    if "validator" in dec_name:
                        for arg in dec.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                validators.setdefault(arg.value, []).append(item.name)

    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            field: dict = {
                "name": item.target.id,
                "type_annotation": _unparse_annotation(item.annotation),
                "default": None,
                "field_info": {},
                "validators": validators.get(item.target.id, []),
            }
            if item.value is not None:
                if isinstance(item.value, ast.Call):
                    func_name = ast.unparse(item.value.func)
                    if "Field" in func_name:
                        # Extract Field() kwargs
                        for kw in item.value.keywords:
                            if kw.arg == "default":
                                field["default"] = ast.unparse(kw.value)
                            elif kw.arg == "description":
                                field["field_info"]["description"] = (
                                    kw.value.value
                                    if isinstance(kw.value, ast.Constant)
                                    else ast.unparse(kw.value)
                                )
                            else:
                                field["field_info"][kw.arg] = ast.unparse(kw.value)
                        # Positional default in Field(default_value)
                        if item.value.args:
                            field["default"] = ast.unparse(item.value.args[0])
                    else:
                        field["default"] = ast.unparse(item.value)
                else:
                    field["default"] = ast.unparse(item.value)
            fields.append(field)
    return fields


def _extract_enum_members(node: ast.ClassDef) -> list[dict]:
    """Extract enum member assignments."""
    members: list[dict] = []
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    members.append(
                        {
                            "name": target.id,
                            "value": ast.unparse(item.value),
                        }
                    )
    return members


# ---------------------------------------------------------------------------
# Function / method extraction
# ---------------------------------------------------------------------------


def extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    """Extract structured information from a function/method node."""
    decorators = _decorator_names(node)
    is_classmethod = any(d in ("classmethod",) for d in decorators)
    is_staticmethod = any(d in ("staticmethod",) for d in decorators)
    is_property = any(d in ("property",) for d in decorators)
    is_abstract = any("abstractmethod" in d for d in decorators)

    params: list[dict] = []
    args = node.args

    # Build defaults mapping: positional defaults are right-aligned
    num_args = len(args.args)
    num_defaults = len(args.defaults)
    default_offset = num_args - num_defaults

    for i, arg in enumerate(args.args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        default_idx = i - default_offset
        default = _unparse_default(args.defaults[default_idx]) if default_idx >= 0 else None
        params.append(
            {
                "name": arg.arg,
                "type_annotation": _unparse_annotation(arg.annotation),
                "default": default,
            }
        )

    # keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        default = _unparse_default(args.kw_defaults[i]) if args.kw_defaults[i] else None
        params.append(
            {
                "name": arg.arg,
                "type_annotation": _unparse_annotation(arg.annotation),
                "default": default,
            }
        )

    # *args
    if args.vararg:
        params.append(
            {
                "name": f"*{args.vararg.arg}",
                "type_annotation": _unparse_annotation(args.vararg.annotation),
                "default": None,
            }
        )

    # **kwargs
    if args.kwarg:
        params.append(
            {
                "name": f"**{args.kwarg.arg}",
                "type_annotation": _unparse_annotation(args.kwarg.annotation),
                "default": None,
            }
        )

    docstring = _extract_docstring(node.body)

    return {
        "name": node.name,
        "docstring": docstring,
        "decorators": decorators,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "is_classmethod": is_classmethod,
        "is_staticmethod": is_staticmethod,
        "is_property": is_property,
        "is_abstract": is_abstract,
        "params": params,
        "return_type": _unparse_annotation(node.returns),
    }


# ---------------------------------------------------------------------------
# Property extraction
# ---------------------------------------------------------------------------


def extract_property(
    name: str,
    getter: ast.FunctionDef,
    setter: ast.FunctionDef | None,
    deleter: ast.FunctionDef | None,
) -> dict:
    """Extract property info from getter/setter/deleter nodes."""
    docstring = _extract_docstring(getter.body)
    return {
        "name": name,
        "docstring": docstring,
        "type_annotation": _unparse_annotation(getter.returns),
        "has_setter": setter is not None,
        "has_deleter": deleter is not None,
    }


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def _should_include_method(name: str, options: dict) -> bool:
    """Determine whether a method should be included based on options."""
    if name.startswith("__") and name.endswith("__"):
        return options.get("include_dunder", False) or name == "__init__"
    if name.startswith("_"):
        return options.get("include_private", False)
    return True


def extract_class(node: ast.ClassDef, source: str, options: dict | None = None) -> dict:
    """Extract structured information from a class node."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    bases = _base_names(node)
    decorators = _decorator_names(node)
    docstring = _extract_docstring(node.body)

    # Classify the class
    is_pydantic = _is_pydantic_model(node)
    is_proto = _is_protocol(node)
    is_en = _is_enum(node)
    is_dc = _is_dataclass(node)
    is_abs = _is_abstract(node)

    # Class variables
    class_variables: list[dict] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name = item.target.id
            if not _should_include_method(name, opts) and name.startswith("_"):
                continue
            class_variables.append(
                {
                    "name": name,
                    "type_annotation": _unparse_annotation(item.annotation),
                    "default_repr": _unparse_default(item.value),
                    "is_class_var": True,
                }
            )

    # Collect methods, properties
    methods: list[dict] = []
    property_map: dict[str, dict] = {}  # name -> {"getter", "setter", "deleter"}
    init_params: list[dict] = []

    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        decs = _decorator_names(item)

        # Property handling
        is_prop_getter = "property" in decs
        is_prop_setter = any(d.endswith(".setter") for d in decs)
        is_prop_deleter = any(d.endswith(".deleter") for d in decs)

        if is_prop_getter:
            property_map.setdefault(item.name, {})["getter"] = item
        elif is_prop_setter:
            prop_name = item.name
            property_map.setdefault(prop_name, {})["setter"] = item
        elif is_prop_deleter:
            prop_name = item.name
            property_map.setdefault(prop_name, {})["deleter"] = item
        else:
            if not _should_include_method(item.name, opts):
                continue
            func_info = extract_function(item)
            if item.name == "__init__":
                # Extract init params with docstring info
                init_docstring = _extract_docstring(item.body)
                param_docs = _parse_docstring_params(init_docstring) if init_docstring else []
                param_doc_map = {p["name"]: p["description"] for p in param_docs}
                for p in func_info["params"]:
                    init_params.append(
                        {
                            "name": p["name"],
                            "type_annotation": p["type_annotation"],
                            "default": p["default"],
                            "docstring": param_doc_map.get(p["name"]),
                        }
                    )
            methods.append(func_info)

    # Build properties list
    properties: list[dict] = []
    for prop_name, parts in sorted(property_map.items()):
        if not _should_include_method(prop_name, opts):
            continue
        getter = parts.get("getter")
        if getter is None:
            continue
        properties.append(
            extract_property(
                prop_name,
                getter,
                parts.get("setter"),
                parts.get("deleter"),
            )
        )

    # Protocol-specific
    required_methods: list[str] = []
    required_attributes: list[str] = []
    if is_proto:
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                required_methods.append(item.name)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                required_attributes.append(item.target.id)

    result: dict = {
        "name": node.name,
        "docstring": docstring,
        "bases": bases,
        "decorators": decorators,
        "is_abstract": is_abs,
        "is_dataclass": is_dc,
        "is_enum": is_en,
        "is_pydantic_model": is_pydantic,
        "is_protocol": is_proto,
        "class_variables": class_variables,
        "init_params": init_params,
        "methods": methods,
        "properties": properties,
    }

    if is_pydantic:
        result["pydantic_fields"] = _extract_pydantic_fields(node)

    if is_en:
        result["enum_members"] = _extract_enum_members(node)

    if is_proto:
        result["is_runtime_checkable"] = _is_runtime_checkable(node)
        result["required_methods"] = required_methods
        result["required_attributes"] = required_attributes

    return result


# ---------------------------------------------------------------------------
# Module extraction
# ---------------------------------------------------------------------------


def _extract_imports(tree: ast.Module) -> list[dict]:
    """Extract import statements from a module."""
    imports: list[dict] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"name": alias.name, "from": None, "alias": alias.asname})
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({"name": alias.name, "from": module, "alias": alias.asname})
    return imports


def _extract_module_assignments(tree: ast.Module) -> tuple[list[dict], list[dict]]:
    """Extract module-level assignments, split into constants and other."""
    constants: list[dict] = []
    assignments: list[dict] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    entry = {
                        "name": target.id,
                        "type_annotation": None,
                        "value_repr": ast.unparse(node.value),
                    }
                    if target.id.isupper():
                        constants.append(entry)
                    else:
                        assignments.append(entry)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            entry = {
                "name": node.target.id,
                "type_annotation": _unparse_annotation(node.annotation),
                "value_repr": _unparse_default(node.value),
            }
            if node.target.id.isupper():
                constants.append(entry)
            else:
                assignments.append(entry)
    return constants, assignments


def extract_module(
    source: str,
    file_path: str = "",
    options: dict | None = None,
) -> dict:
    """Parse source code and extract structured documentation IR.

    :param source: Python source code string
    :param file_path: Optional file path for metadata
    :param options: Extraction options (see DEFAULT_OPTIONS)
    :returns: JSON-serializable dict of module documentation
    """
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    tree = ast.parse(source)

    docstring = _extract_docstring(tree.body)
    imports = _extract_imports(tree)
    constants, assignments = _extract_module_assignments(tree)

    # Convert file path to module path
    module_path = ""
    if file_path:
        module_path = file_path.replace(os.sep, ".").removesuffix(".py").removeprefix(".")

    classes: list[dict] = []
    functions: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if not _should_include_method(node.name, opts):
                continue
            classes.append(extract_class(node, source, opts))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _should_include_method(node.name, opts):
                continue
            functions.append(extract_function(node))

    return {
        "module_path": module_path,
        "file_path": file_path,
        "docstring": docstring,
        "imports": imports,
        "module_level_assignments": assignments,
        "constants": constants,
        "classes": classes,
        "functions": functions,
    }


# ---------------------------------------------------------------------------
# Package extraction
# ---------------------------------------------------------------------------


def extract_package(
    package_path: str,
    options: dict | None = None,
) -> dict:
    """Recursively scan a package directory and build documentation IR.

    :param package_path: Path to the package directory
    :param options: Extraction options (see DEFAULT_OPTIONS)
    :returns: JSON-serializable dict with modules, dependency graph, inheritance tree
    """
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    pkg = Path(package_path).resolve()
    package_name = pkg.name

    modules_data: dict[str, dict] = {}
    all_classes: dict[str, list[str]] = {}  # class_name -> bases
    sub_packages: list[str] = []

    # Discover sub-packages
    for child in sorted(pkg.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            sub_packages.append(child.name)

    # Collect all .py files
    py_files = sorted(pkg.rglob("*.py"))

    for py_file in py_files:
        rel = py_file.relative_to(pkg)
        # Build module name relative to package
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            if len(parts) == 1:
                mod_name = "__init__"
            else:
                parts = parts[:-1]
                mod_name = ".".join(parts)
        else:
            parts[-1] = parts[-1].removesuffix(".py")
            mod_name = ".".join(parts)

        source = py_file.read_text(encoding="utf-8")
        mod_data = extract_module(source, str(rel), opts)
        modules_data[mod_name] = mod_data

        # Collect class inheritance info
        for cls in mod_data["classes"]:
            all_classes[cls["name"]] = cls["bases"]

    # Build dependency graph (intra-package only)
    dependency_graph: dict[str, list[str]] = {}
    module_names = set(modules_data.keys())

    for mod_name, mod_data in modules_data.items():
        deps: list[str] = []
        for imp in mod_data["imports"]:
            from_module = imp.get("from") or ""
            # Check relative imports (starting with .)
            if from_module.startswith("."):
                # Resolve relative import
                target = from_module.lstrip(".")
                if target in module_names:
                    deps.append(target)
                # Also check just the module part
                parts = target.split(".")
                if parts[0] in module_names:
                    deps.append(parts[0])
            # Check if the imported module matches a package module
            for mn in module_names:
                if from_module.endswith(mn) or from_module.endswith(f".{mn}"):
                    deps.append(mn)
        # Deduplicate and remove self-references
        deps = sorted(set(d for d in deps if d != mod_name))
        if deps:
            dependency_graph[mod_name] = deps

    # Build inheritance tree
    inheritance_tree: dict[str, list[str]] = {}
    for class_name, bases in sorted(all_classes.items()):
        inheritance_tree[class_name] = bases

    return {
        "package_path": package_name,
        "modules": sorted(m for m in modules_data.keys() if m != "__init__"),
        "sub_packages": sorted(sub_packages),
        "dependency_graph": dict(sorted(dependency_graph.items())),
        "inheritance_tree": dict(sorted(inheritance_tree.items())),
        "module_data": {k: v for k, v in sorted(modules_data.items())},
    }


# ---------------------------------------------------------------------------
# Public JSON serialization helper
# ---------------------------------------------------------------------------


def to_json(data: dict, indent: int = 2) -> str:
    """Serialize extraction result to deterministic JSON."""
    return json.dumps(data, indent=indent, sort_keys=True, ensure_ascii=False)
