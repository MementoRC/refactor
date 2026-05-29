from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FunctionInfo:
    name: str
    is_async: bool
    params: list[tuple[str, str | None]]  # (name, type_annotation_str or None)
    return_type: str | None
    has_dynamic_patterns: list[str]  # e.g., ["getattr", "**kwargs"]
    line_count: int
    docstring: str | None


@dataclass
class ClassInfo:
    name: str
    methods: list[FunctionInfo]
    attributes: list[tuple[str, str | None]]  # (name, type or None)
    bases: list[str]
    docstring: str | None


@dataclass
class ModuleAnalysis:
    module_name: str
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    imports: list[str]

    # Readiness metrics
    type_coverage: float  # 0.0 to 1.0
    dynamic_pattern_count: int
    conversion_readiness: str  # "ready", "needs_types", "needs_review", "complex"

    # Summary
    total_functions: int
    total_classes: int
    fully_typed_functions: int
    flagged_functions: int  # functions with dynamic patterns


# ---------------------------------------------------------------------------
# Dynamic pattern detection (mirrors FlagDynamicPatterns in rust_prep.py)
# ---------------------------------------------------------------------------

_DYNAMIC_CALLS: dict[str, str] = {
    "getattr": "dynamic attribute access",
    "setattr": "dynamic attribute setting",
    "delattr": "dynamic attribute deletion",
    "exec": "dynamic code execution",
    "eval": "dynamic code evaluation",
    "globals": "runtime introspection",
    "locals": "runtime introspection",
    "__import__": "dynamic import",
}


def _extract_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str | None:
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value
    return None


def _annotation_to_str(node: ast.expr | None) -> str | None:
    """Convert an annotation AST node to a source string (best-effort)."""
    if node is None:
        return None
    return ast.unparse(node)


def _analyze_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> FunctionInfo:
    """Build a FunctionInfo from a function/async-function AST node."""
    params: list[tuple[str, str | None]] = []
    for arg in node.args.args:
        if arg.arg in ("self", "cls"):
            continue
        params.append((arg.arg, _annotation_to_str(arg.annotation)))

    dynamic_patterns: list[str] = []

    # Signature-level patterns
    if node.args.kwarg:
        dynamic_patterns.append("**kwargs")
    if node.args.vararg:
        dynamic_patterns.append("*args")

    # Walk body for dynamic calls
    seen_reasons: set[str] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in _DYNAMIC_CALLS
        ):
            reason = _DYNAMIC_CALLS[child.func.id]
            if reason not in seen_reasons:
                seen_reasons.add(reason)
                dynamic_patterns.append(child.func.id)

    line_count = (node.end_lineno or node.lineno) - node.lineno + 1

    return FunctionInfo(
        name=node.name,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        params=params,
        return_type=_annotation_to_str(node.returns),
        has_dynamic_patterns=dynamic_patterns,
        line_count=line_count,
        docstring=_extract_docstring(node),
    )


def _analyze_class(node: ast.ClassDef) -> ClassInfo:
    """Build a ClassInfo from a class AST node."""
    methods: list[FunctionInfo] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_analyze_function(item))

    # Collect class-level annotated attributes (ClassVar / AnnAssign at class body level)
    attributes: list[tuple[str, str | None]] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            attributes.append((item.target.id, _annotation_to_str(item.annotation)))

    bases = [ast.unparse(b) for b in node.bases]

    return ClassInfo(
        name=node.name,
        methods=methods,
        attributes=attributes,
        bases=bases,
        docstring=_extract_docstring(node),
    )


def _import_to_str(node: ast.stmt) -> str:
    """Convert an import AST node to a source string."""
    return ast.unparse(node)


def _compute_type_coverage(
    functions: list[FunctionInfo],
    classes: list[ClassInfo],
) -> float:
    """Return fraction of params+returns that carry type annotations."""
    total = 0
    annotated = 0

    def _tally(fn: FunctionInfo) -> None:
        nonlocal total, annotated
        for _name, ann in fn.params:
            total += 1
            if ann is not None:
                annotated += 1
        # return type counts as one slot
        total += 1
        if fn.return_type is not None:
            annotated += 1

    for fn in functions:
        _tally(fn)

    for cls in classes:
        for method in cls.methods:
            # Skip self/cls — already excluded in params list
            _tally(method)

    if total == 0:
        return 1.0  # nothing to annotate → trivially covered
    return annotated / total


def _count_dynamic_patterns(
    functions: list[FunctionInfo],
    classes: list[ClassInfo],
) -> int:
    total = sum(len(fn.has_dynamic_patterns) for fn in functions)
    for cls in classes:
        total += sum(len(m.has_dynamic_patterns) for m in cls.methods)
    return total


def _determine_readiness(type_coverage: float, dynamic_pattern_count: int) -> str:
    if type_coverage >= 0.9 and dynamic_pattern_count == 0:
        return "ready"
    if type_coverage < 0.9 and dynamic_pattern_count == 0:
        return "needs_types"
    if 0 < dynamic_pattern_count <= 3:
        return "needs_review"
    return "complex"


# ---------------------------------------------------------------------------
# Public API — ModuleAnalyzer
# ---------------------------------------------------------------------------


def analyze_module(source: str, module_name: str = "module") -> ModuleAnalysis:
    """Analyze a Python module for Rust conversion readiness."""
    tree = ast.parse(source)

    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []
    imports: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_analyze_function(node))
        elif isinstance(node, ast.ClassDef):
            classes.append(_analyze_class(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(_import_to_str(node))

    type_coverage = _compute_type_coverage(functions, classes)
    dynamic_pattern_count = _count_dynamic_patterns(functions, classes)
    conversion_readiness = _determine_readiness(type_coverage, dynamic_pattern_count)

    fully_typed_functions = sum(
        1
        for fn in functions
        if all(ann is not None for _, ann in fn.params) and fn.return_type is not None
    )
    flagged_functions = sum(1 for fn in functions if fn.has_dynamic_patterns)

    return ModuleAnalysis(
        module_name=module_name,
        functions=functions,
        classes=classes,
        imports=imports,
        type_coverage=type_coverage,
        dynamic_pattern_count=dynamic_pattern_count,
        conversion_readiness=conversion_readiness,
        total_functions=len(functions),
        total_classes=len(classes),
        fully_typed_functions=fully_typed_functions,
        flagged_functions=flagged_functions,
    )


# ---------------------------------------------------------------------------
# Public API — PythonWrapperGenerator
# ---------------------------------------------------------------------------


def generate_python_wrapper(source: str, module_name: str) -> str:
    """Generate Python wrapper that imports from Rust extension with fallback."""
    tree = ast.parse(source)

    public_functions: list[str] = []
    public_classes: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                public_functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                public_classes.append(node.name)

    all_public = public_functions + public_classes
    import_names = ", ".join(all_public)

    # Indent the original source for use inside the `if not _RUST_AVAILABLE:` block
    indented_fallback = textwrap.indent(source.rstrip(), "    ")

    lines: list[str] = []
    lines.append(f'"""Python wrapper for {module_name} with Rust acceleration.')
    lines.append("")
    lines.append("Imports from the compiled Rust extension when available,")
    lines.append("falls back to pure Python implementation otherwise.")
    lines.append('"""')
    lines.append("try:")
    lines.append(f"    from ._{module_name}_rs import {import_names}")
    lines.append("    _RUST_AVAILABLE = True")
    lines.append("except ImportError:")
    lines.append("    _RUST_AVAILABLE = False")
    lines.append("")
    lines.append("if not _RUST_AVAILABLE:")
    lines.append("    # Pure Python fallback")
    lines.append(indented_fallback)
    lines.append("")
    lines.append("")
    lines.append("def is_accelerated() -> bool:")
    lines.append('    """Return True if Rust acceleration is available."""')
    lines.append("    return _RUST_AVAILABLE")

    return "\n".join(lines)
