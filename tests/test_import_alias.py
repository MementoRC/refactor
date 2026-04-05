"""Test for import alias handling in ScopeInfo definitions."""

import ast
import textwrap

from refactor.context import Configuration, Context, Scope, _resolve_dependencies


def get_context(source, *representatives, **kwargs):
    tree = ast.parse(textwrap.dedent(source))
    config = Configuration(**kwargs)
    return Context._from_dependencies(
        _resolve_dependencies(representatives),
        config=config,
        tree=tree,
        source=source,
    )


def test_scope_definitions_import_alias():
    """Test that import aliases are correctly registered in scope definitions.

    This test validates the change from:
        local_definitions[alias.name].append(node)
    to:
        local_definitions[(alias.asname or alias.name)].append(node)
    """
    context = get_context(
        textwrap.dedent(
            """
            import module_a
            import module_b as alias_b
            from module_c import name_c
            from module_d import name_d as alias_d

            def func():
                import module_e
                import module_f as alias_f
                from module_g import name_g
                from module_h import name_h as alias_h
                pass
            """
        ),
        Scope,
    )

    tree = context.tree
    scope = context.metadata["scope"]

    # Find the function 'func'
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "func":
            func_node = node
            break
    assert func_node is not None

    # Get global scope and function scope
    global_scope = scope.resolve(tree.body[0])
    func_scope = scope.resolve(func_node.body[0])

    # Test global scope imports - module name only
    assert "module_a" in global_scope.definitions
    import_module_a = global_scope.definitions["module_a"][0]
    assert isinstance(import_module_a, ast.Import)
    assert import_module_a.names[0].name == "module_a"
    assert import_module_a.names[0].asname is None

    # Test global scope imports - with alias
    assert "module_b" not in global_scope.definitions
    assert "alias_b" in global_scope.definitions
    import_module_b = global_scope.definitions["alias_b"][0]
    assert isinstance(import_module_b, ast.Import)
    assert import_module_b.names[0].name == "module_b"
    assert import_module_b.names[0].asname == "alias_b"

    # Test global scope imports - from import
    assert "name_c" in global_scope.definitions
    from_import_c = global_scope.definitions["name_c"][0]
    assert isinstance(from_import_c, ast.ImportFrom)
    assert from_import_c.module == "module_c"
    assert from_import_c.names[0].name == "name_c"
    assert from_import_c.names[0].asname is None

    # Test global scope imports - from import with alias
    assert "name_d" not in global_scope.definitions
    assert "alias_d" in global_scope.definitions
    from_import_d = global_scope.definitions["alias_d"][0]
    assert isinstance(from_import_d, ast.ImportFrom)
    assert from_import_d.module == "module_d"
    assert from_import_d.names[0].name == "name_d"
    assert from_import_d.names[0].asname == "alias_d"

    # Test function scope imports - module name only
    assert "module_e" in func_scope.definitions
    import_module_e = func_scope.definitions["module_e"][0]
    assert isinstance(import_module_e, ast.Import)
    assert import_module_e.names[0].name == "module_e"
    assert import_module_e.names[0].asname is None

    # Test function scope imports - with alias
    assert "module_f" not in func_scope.definitions
    assert "alias_f" in func_scope.definitions
    import_module_f = func_scope.definitions["alias_f"][0]
    assert isinstance(import_module_f, ast.Import)
    assert import_module_f.names[0].name == "module_f"
    assert import_module_f.names[0].asname == "alias_f"

    # Test function scope imports - from import
    assert "name_g" in func_scope.definitions
    from_import_g = func_scope.definitions["name_g"][0]
    assert isinstance(from_import_g, ast.ImportFrom)
    assert from_import_g.module == "module_g"
    assert from_import_g.names[0].name == "name_g"
    assert from_import_g.names[0].asname is None

    # Test function scope imports - from import with alias
    assert "name_h" not in func_scope.definitions
    assert "alias_h" in func_scope.definitions
    from_import_h = func_scope.definitions["alias_h"][0]
    assert isinstance(from_import_h, ast.ImportFrom)
    assert from_import_h.module == "module_h"
    assert from_import_h.names[0].name == "name_h"
    assert from_import_h.names[0].asname == "alias_h"
