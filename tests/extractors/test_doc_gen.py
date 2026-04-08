"""Comprehensive tests for the AST documentation extraction engine."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from refactor.extractors.doc_gen import extract_module, extract_package, to_json

FIXTURES = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Simple class extraction
# ---------------------------------------------------------------------------


class TestSimpleClassExtraction:
    def test_class_name_bases_docstring(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")
        animal = result["classes"][0]

        assert animal["name"] == "Animal"
        assert animal["bases"] == []
        assert animal["docstring"] == "Base animal class."

    def test_class_methods(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")
        animal = result["classes"][0]

        method_names = [m["name"] for m in animal["methods"]]
        assert "__init__" in method_names
        assert "speak" in method_names

    def test_dog_inherits_animal(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")
        dog = result["classes"][1]

        assert dog["name"] == "Dog"
        assert dog["bases"] == ["Animal"]
        assert dog["docstring"] == "A dog."


# ---------------------------------------------------------------------------
# 2. Pydantic model extraction
# ---------------------------------------------------------------------------


class TestPydanticModelExtraction:
    def test_is_pydantic_model(self):
        source = _read_fixture("pydantic_model.py")
        result = extract_module(source, "pydantic_model.py")
        config = result["classes"][0]

        assert config["is_pydantic_model"] is True
        assert config["name"] == "TradingConfig"

    def test_pydantic_fields(self):
        source = _read_fixture("pydantic_model.py")
        result = extract_module(source, "pydantic_model.py")
        config = result["classes"][0]
        fields = config["pydantic_fields"]

        field_names = [f["name"] for f in fields]
        assert "trading_pair" in field_names
        assert "amount" in field_names
        assert "enabled" in field_names

    def test_pydantic_field_info(self):
        source = _read_fixture("pydantic_model.py")
        result = extract_module(source, "pydantic_model.py")
        config = result["classes"][0]
        fields = {f["name"]: f for f in config["pydantic_fields"]}

        assert fields["trading_pair"]["field_info"]["description"] == "Trading pair"
        assert fields["amount"]["default"] == "1.0"

    def test_pydantic_validators(self):
        source = _read_fixture("pydantic_model.py")
        result = extract_module(source, "pydantic_model.py")
        config = result["classes"][0]
        fields = {f["name"]: f for f in config["pydantic_fields"]}

        assert "validate_pair" in fields["trading_pair"]["validators"]


# ---------------------------------------------------------------------------
# 3. Protocol detection
# ---------------------------------------------------------------------------


class TestProtocolDetection:
    def test_is_protocol(self):
        source = _read_fixture("protocol_example.py")
        result = extract_module(source, "protocol_example.py")
        connector = result["classes"][0]

        assert connector["is_protocol"] is True
        assert connector["name"] == "Connector"

    def test_runtime_checkable(self):
        source = _read_fixture("protocol_example.py")
        result = extract_module(source, "protocol_example.py")
        connector = result["classes"][0]

        assert connector["is_runtime_checkable"] is True

    def test_required_methods(self):
        source = _read_fixture("protocol_example.py")
        result = extract_module(source, "protocol_example.py")
        connector = result["classes"][0]

        assert "start" in connector["required_methods"]
        assert "stop" in connector["required_methods"]
        assert "get_balance" in connector["required_methods"]

    def test_required_attributes(self):
        source = _read_fixture("protocol_example.py")
        result = extract_module(source, "protocol_example.py")
        connector = result["classes"][0]

        assert "name" in connector["required_attributes"]


# ---------------------------------------------------------------------------
# 4. Enum extraction
# ---------------------------------------------------------------------------


class TestEnumExtraction:
    def test_is_enum(self):
        source = _read_fixture("enum_example.py")
        result = extract_module(source, "enum_example.py")

        order_type = result["classes"][0]
        assert order_type["is_enum"] is True
        assert order_type["name"] == "OrderType"

    def test_enum_members(self):
        source = _read_fixture("enum_example.py")
        result = extract_module(source, "enum_example.py")
        order_type = result["classes"][0]
        members = order_type["enum_members"]

        member_names = [m["name"] for m in members]
        assert "LIMIT" in member_names
        assert "MARKET" in member_names
        assert "LIMIT_MAKER" in member_names

    def test_int_enum(self):
        source = _read_fixture("enum_example.py")
        result = extract_module(source, "enum_example.py")
        trade_type = result["classes"][1]

        assert trade_type["is_enum"] is True
        assert trade_type["name"] == "TradeType"
        members = {m["name"]: m["value"] for m in trade_type["enum_members"]}
        assert members["BUY"] == "1"
        assert members["SELL"] == "2"


# ---------------------------------------------------------------------------
# 5. Property detection
# ---------------------------------------------------------------------------


class TestPropertyDetection:
    def test_property_extracted(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")
        animal = result["classes"][0]

        assert len(animal["properties"]) == 1
        prop = animal["properties"][0]
        assert prop["name"] == "is_adult"
        assert prop["type_annotation"] == "bool"
        assert prop["has_setter"] is False
        assert prop["has_deleter"] is False

    def test_property_with_setter(self):
        source = textwrap.dedent("""\
            class Foo:
                @property
                def value(self) -> int:
                    return self._value

                @value.setter
                def value(self, val: int) -> None:
                    self._value = val
        """)
        result = extract_module(source)
        foo = result["classes"][0]
        prop = foo["properties"][0]

        assert prop["name"] == "value"
        assert prop["has_setter"] is True


# ---------------------------------------------------------------------------
# 6. Async method detection
# ---------------------------------------------------------------------------


class TestAsyncMethodDetection:
    def test_async_flag(self):
        source = _read_fixture("protocol_example.py")
        result = extract_module(source, "protocol_example.py")
        connector = result["classes"][0]

        methods = {m["name"]: m for m in connector["methods"]}
        assert methods["get_balance"]["is_async"] is True
        assert methods["start"]["is_async"] is False


# ---------------------------------------------------------------------------
# 7. Decorator preservation
# ---------------------------------------------------------------------------


class TestDecoratorPreservation:
    def test_class_decorator(self):
        source = _read_fixture("protocol_example.py")
        result = extract_module(source, "protocol_example.py")
        connector = result["classes"][0]

        assert "runtime_checkable" in connector["decorators"]

    def test_method_decorator(self):
        source = _read_fixture("pydantic_model.py")
        result = extract_module(
            source, "pydantic_model.py", {"include_private": False, "include_dunder": True}
        )
        config = result["classes"][0]

        validator_methods = [
            m for m in config["methods"] if any("validator" in d for d in m["decorators"])
        ]
        assert len(validator_methods) == 1
        assert validator_methods[0]["name"] == "validate_pair"


# ---------------------------------------------------------------------------
# 8. Package dependency graph
# ---------------------------------------------------------------------------


class TestPackageDependencyGraph:
    def test_dependency_graph(self):
        pkg_path = str(FIXTURES / "package_example")
        result = extract_package(pkg_path)

        assert "module_a" in result["modules"]
        assert "module_b" in result["modules"]
        # module_a imports from module_b
        assert "module_a" in result["dependency_graph"]
        assert "module_b" in result["dependency_graph"]["module_a"]


# ---------------------------------------------------------------------------
# 9. Inheritance tree
# ---------------------------------------------------------------------------


class TestInheritanceTree:
    def test_inheritance_tree(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")

        # Dog inherits from Animal
        dog = next(c for c in result["classes"] if c["name"] == "Dog")
        assert dog["bases"] == ["Animal"]

    def test_package_inheritance_tree(self):
        pkg_path = str(FIXTURES / "package_example")
        result = extract_package(pkg_path)

        assert "Worker" in result["inheritance_tree"]
        assert "Helper" in result["inheritance_tree"]


# ---------------------------------------------------------------------------
# 10. Docstring param parsing
# ---------------------------------------------------------------------------


class TestDocstringParamParsing:
    def test_param_extraction(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")
        animal = result["classes"][0]

        init_params = animal["init_params"]
        param_map = {p["name"]: p for p in init_params}
        assert param_map["name"]["docstring"] == "The animal's name"
        assert param_map["age"]["docstring"] == "The animal's age in years"

    def test_param_types(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")
        animal = result["classes"][0]

        init_params = animal["init_params"]
        param_map = {p["name"]: p for p in init_params}
        assert param_map["name"]["type_annotation"] == "str"
        assert param_map["age"]["type_annotation"] == "int"
        assert param_map["age"]["default"] == "0"


# ---------------------------------------------------------------------------
# 11. Future annotations handling
# ---------------------------------------------------------------------------


class TestFutureAnnotations:
    def test_future_annotations(self):
        source = textwrap.dedent("""\
            from __future__ import annotations

            class Foo:
                def bar(self, x: int | str) -> list[int]:
                    pass
        """)
        result = extract_module(source)
        foo = result["classes"][0]
        bar = foo["methods"][0]

        assert bar["return_type"] == "list[int]"
        assert bar["params"][0]["type_annotation"] == "int | str"


# ---------------------------------------------------------------------------
# 12. Deterministic output
# ---------------------------------------------------------------------------


class TestDeterministicOutput:
    def test_same_input_same_output(self):
        source = _read_fixture("simple_class.py")
        result1 = to_json(extract_module(source, "simple_class.py"))
        result2 = to_json(extract_module(source, "simple_class.py"))

        assert result1 == result2

    def test_json_parseable(self):
        source = _read_fixture("simple_class.py")
        output = to_json(extract_module(source, "simple_class.py"))
        parsed = json.loads(output)

        assert isinstance(parsed, dict)
        assert "classes" in parsed


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


class TestModuleLevelAssignments:
    def test_constant_extraction(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")

        constants = result["constants"]
        names = [c["name"] for c in constants]
        assert "CONSTANT" in names

    def test_assignment_extraction(self):
        source = _read_fixture("simple_class.py")
        result = extract_module(source, "simple_class.py")

        assignments = result["module_level_assignments"]
        names = [a["name"] for a in assignments]
        assert "logger" in names


class TestIncludePrivateOption:
    def test_private_filtered_by_default(self):
        source = textwrap.dedent("""\
            class Foo:
                def public(self) -> None:
                    pass
                def _private(self) -> None:
                    pass
        """)
        result = extract_module(source)
        foo = result["classes"][0]
        method_names = [m["name"] for m in foo["methods"]]

        assert "public" in method_names
        assert "_private" not in method_names

    def test_private_included_with_option(self):
        source = textwrap.dedent("""\
            class Foo:
                def public(self) -> None:
                    pass
                def _private(self) -> None:
                    pass
        """)
        result = extract_module(source, options={"include_private": True})
        foo = result["classes"][0]
        method_names = [m["name"] for m in foo["methods"]]

        assert "public" in method_names
        assert "_private" in method_names


class TestExtractModuleFromString:
    def test_inline_source(self):
        source = textwrap.dedent("""\
            \"\"\"Test module.\"\"\"

            def greet(name: str) -> str:
                \"\"\"Say hello.\"\"\"
                return f"Hello, {name}"
        """)
        result = extract_module(source)

        assert result["docstring"] == "Test module."
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "greet"


class TestCLI:
    def test_cli_module_flag(self, tmp_path):
        from refactor.extractors.__main__ import main

        test_file = tmp_path / "sample.py"
        test_file.write_text('"""Sample."""\n\nX = 1\n')
        output_file = tmp_path / "out.json"

        rc = main(["--module", str(test_file), "--output", str(output_file)])

        assert rc == 0
        data = json.loads(output_file.read_text())
        assert data["docstring"] == "Sample."

    def test_cli_package_flag(self, tmp_path):
        from refactor.extractors.__main__ import main

        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text('"""A module."""\n\nclass Foo:\n    pass\n')
        output_file = tmp_path / "out.json"

        rc = main(["--package", str(pkg), "--output", str(output_file)])

        assert rc == 0
        data = json.loads(output_file.read_text())
        assert "mod" in data["modules"]

    def test_cli_missing_file(self):
        from refactor.extractors.__main__ import main

        rc = main(["--module", "/nonexistent/file.py"])
        assert rc == 1


class TestAbstractDetection:
    def test_abstract_via_decorator(self):
        source = textwrap.dedent("""\
            from abc import abstractmethod

            class Base:
                @abstractmethod
                def do_work(self) -> None:
                    pass
        """)
        result = extract_module(source, options={"include_dunder": False})
        assert result["classes"][0]["is_abstract"] is True

    def test_abstract_via_base(self):
        source = textwrap.dedent("""\
            from abc import ABC

            class Base(ABC):
                def do_work(self) -> None:
                    pass
        """)
        result = extract_module(source)
        assert result["classes"][0]["is_abstract"] is True


class TestDataclassDetection:
    def test_dataclass(self):
        source = textwrap.dedent("""\
            from dataclasses import dataclass

            @dataclass
            class Point:
                x: float
                y: float
        """)
        result = extract_module(source)
        assert result["classes"][0]["is_dataclass"] is True
        assert result["classes"][0]["name"] == "Point"
