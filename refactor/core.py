from __future__ import annotations

import ast
import importlib
import inspect
import sys
import tempfile
import tokenize
import warnings
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field, astuple
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import ClassVar, NoReturn, Tuple, Generator, Type, List, Dict, Optional, Set, Callable

# TODO: remove the deprecated aliases on 1.0.0
from refactor.actions import (  # unimport:skip
    Action,
    BaseAction,
    NewStatementAction,
    ReplacementAction,
    TargetedNewStatementAction,
)
from refactor.change import Change
from refactor.common import _FileInfo, has_positions, pascal_to_snake
from refactor.context import (
    Configuration,
    Context,
    Representative,
    _resolve_dependencies,
)
from refactor.internal.action_optimizer import optimize


def _unparsable_source_code(source: str, exc: SyntaxError) -> NoReturn:
    error_message = "Generated source is unparsable."

    if Session.c_current_config.debug_mode:
        fd, file_name = tempfile.mkstemp(prefix="refactor", text=True)
        with open(fd, "w") as stream:
            stream.write(source)
        error_message += f"\nSee {file_name} for the generated source."

    raise ValueError(error_message) from exc


def _match_from_rule_or_collection(
        r_or_c: Rule | RuleCollection,
        node: ast.AST
) -> Generator[Tuple[Rule, BaseAction | Iterator[BaseAction]]]:
    with suppress(AssertionError):
        if isinstance(r_or_c, RuleCollection):
            yield from r_or_c.match(node)
        else:
            yield r_or_c, r_or_c.match(node)


def _import_named_rules(rules: List[str | Type[Rule] | Type[RuleCollection]], module_set: Set[ModuleType] = None):
    """Import Rule/RuleCollection defined by name in the rules attribute."""
    if module_set is None:
        module_set: Set[ModuleType] = {module for n in rules if not isinstance(n, str) if
                                       (module := inspect.getmodule(n)) is not None}
    else:
        module_set = module_set.union(
            {module for n in rules if not isinstance(n, str) if (module := inspect.getmodule(n)) is not None})

    if not module_set:
        raise InvalidNameOfRuleOrRuleCollection("No module found for the rules. At least one master class must exists")

    is_rule_class: Callable = lambda n: issubclass(n, Rule) and n is not Rule
    is_rule_collection_class: Callable = lambda n: issubclass(n, RuleCollection) and n is not RuleCollection
    is_member: Callable = lambda n: inspect.isclass(n) and (is_rule_class(n) or is_rule_collection_class(n))

    for rule in rules:
        if isinstance(rule, str):
            script_class: Type[Rule | RuleCollection | None] = None
            for module in module_set:
                script_class = next((member for member_name, member in inspect.getmembers(module)
                                     if is_member(member) and member_name == rule), None)
                break

            if script_class is None:
                _module: str = pascal_to_snake(rule)
                rules.remove(rule)
                raise InvalidNameOfRuleOrRuleCollection(
                    f"Module for '{rule}' not found. Removed from the list of rules")

            if issubclass(script_class, RuleCollection):
                script_class.import_named_rules()
                _import_named_rules(script_class.rules, module_set)

            rules[rules.index(rule)] = script_class

        elif issubclass(rule, RuleCollection):
            rule.import_named_rules()
            _import_named_rules(rule.rules, module_set)


class MaybeOverlappingActions(Exception):
    pass


class InvalidNameOfRuleOrRuleCollection(Exception):
    pass


class FailedImportOfRuleORCollection(Exception):
    pass


@dataclass
class Rule:
    context_providers: ClassVar[Tuple[Type[Representative], ...]] = ()

    context: Context

    def check_file(self, path: Path | None) -> bool:
        """Check whether to process the given ``path``. If ``path`` is `None`,
        that means the user has submitted a string to be processed.

        By default it will always be `True` but can be overridden
        in subclasses.
        """
        return True

    def match(
            self,
            node: ast.AST,
    ) -> BaseAction | None | Iterator[BaseAction]:
        """Match the given ``node`` against current rule's scope.

        On success, it will return a source code transformation action
        (an instance of :class:`refactor.actions.BaseAction`). On failure
        it might either raise an `AssertionError` or return `None`.
        """
        raise NotImplementedError


class _IsIterable(type):
    """Makes the class iterable in the sense of dependencies."""
    context_providers: ClassVar[Tuple[Type[Representative], ...]] = ()

    def __iter__(self) -> Iterator[RuleCollection]:
        return iter(self.rules)


@dataclass
class RuleCollection(metaclass=_IsIterable):
    """Collects a set of Type[Rule] and Type[RuleCollection] to be used as a groupable Rules
    The idea is simply to allow cleaner complex Chained rules that may throw 'MaybeOverlap
    when too large, yet allowing the Session to have a short set of 'Rule' and 'Collection'"""
    rules: ClassVar[List[str | Type[Rule] | Type[RuleCollection]]]

    rule_instances: Dict[Type[Rule], Rule] = field(default_factory=dict)
    collection_instances: Dict[Type[RuleCollection], RuleCollection] = field(default_factory=dict)

    _validated: bool = field(default=False, repr=False)
    _initialized: bool = field(default=False, repr=False)

    @classmethod
    def import_named_rules(cls, indentation: str = ""):
        """Rudimentary search for classes defined by their name in the rules attribute."""

        def class_from_namespace() -> Type[Rule | RuleCollection] | None:
            return next((m for m_name, m in inspect.getmembers(module, inspect.isclass) if m_name == rule), None)

        def class_from_package(package: str = "") -> Type[Rule | RuleCollection] | None:
            module_name: str = pascal_to_snake(rule)
            if (path.parent / f"{package}" / f"{module_name}.py").exists():
                try:
                    module = SourceFileLoader(module_name,
                                              str(path.parent / f"{package}" / f"{module_name}.py")).load_module()
                    return getattr(module, rule)
                except ImportError:
                    raise FailedImportOfRuleORCollection(
                        f"Failed import of {module_name} from {package}. Check that this contains a valid Python code"
                        f"\n\tModule for '{rule}' in {path.parent / f'{package}' / f'{module_name}.py'} not found")
            else:
                return None

        def class_from_packages() -> Type[Rule | RuleCollection] | None:
            for sub_dir in [d for d in path.parent.iterdir() if d.is_dir()]:
                if (m := class_from_package(package=sub_dir.name)) is not None:
                    return m
            return None

        module = inspect.getmodule(cls)
        path = Path(module.__file__)

        for rule in cls.rules:
            if isinstance(rule, str):
                if (rule_or_collection_class := class_from_namespace()) is None:
                    if (rule_or_collection_class := class_from_package()) is None:
                        if (rule_or_collection_class := class_from_packages()) is None:
                            cls.rules.remove(rule)
                            continue

                if issubclass(rule_or_collection_class, RuleCollection):
                    rule_or_collection_class.import_named_rules(indentation=indentation + "   ")

                cls.rules[cls.rules.index(rule)] = rule_or_collection_class

    def _validate_collection(self) -> bool:
        """Check if the rules exists and are valid. Raise an error if not.
        Always returns True on valid RunCollection, otherwise raises an error

        :raises AttributeError: if the 'rules' attribute is not defined
        :raises TypeError: if the rules attribute is not a list
        :raises TypeError: if the rules attribute contains Rules or RuleCollections
        :return: True if the collection is valid
        """
        if not isinstance(self.rules, list):
            raise TypeError("RuleCollection.rules must be a list")
        if not all(issubclass(rule, (Rule, RuleCollection)) for rule in self.rules):
            for rule in self.rules:
                if not issubclass(rule, (Rule, RuleCollection)):
                    raise TypeError(f"RuleCollection.rules must contain only Rules or RuleCollections, not {rule}")

        # Remove duplicates
        rules: List[str | Type[Rule] | Type[RuleCollection]] = getattr(self, "rules", None)
        setattr(self, "rules", [x for i, x in enumerate(rules) if x not in rules[:i]])

        # Process collections within this collection. This is different, the collections need
        # to be initialized with the context, but the rules do not.
        for collection_type in self.rules:
            if issubclass(collection_type, RuleCollection):
                collection: RuleCollection = collection_type()
                if collection._validate_collection():
                    self.collection_instances[collection_type] = collection
                else:
                    del self.rules[self.rules.index(collection_type)]
        self._validated = True

        return self._validated

    def initialize_rules(self, context: Context, path: Path | None = None) -> None:
        """Initialize all rules in the collection. Intended to be called by the Session"""
        if not self._validated:
            # Validate the collection before initializing, returns True or Raises an error
            self._validate_collection()

        for rule in self.rules:
            if issubclass(rule, Rule) and (instance := rule(context)).check_file(path):
                self.rule_instances[rule] = instance

            # If the rule is a RuleCollection, call its initialization method
            if issubclass(rule, RuleCollection):
                self.collection_instances[rule].initialize_rules(context, path)

        self._initialized = True

    def check_file(self, path: Path | None) -> bool:
        """This should always be True, as it is only called on the top level RuleCollection"""
        return self._initialized

    def match(self, node: ast.AST) -> Generator[Tuple[Rule, BaseAction | None | Iterator[BaseAction]]]:
        """Match the given ``node`` against all the rules in the collection.

        It yields tuples of all the Rule, BaseAction that match.
        :raises RuntimeError: if the RuleCollection is not initialized
        :return: A generator of tuples of Rule, BaseAction
        """

        if not self._initialized:
            raise RuntimeError("RuleCollection must be initialized before matching")

        # We keep the order of 'rules'
        for rule_type in self.rules:
            # Only initialized rules are in the rule_instances dict
            if rule_type in self.rule_instances.keys():
                # For the rules in the Collection, we need to suppress the AssertionError
                # as it is not an error, it is just a failed match in the list of Rules
                with suppress(AssertionError):
                    matched_action = self.rule_instances[rule_type].match(node)
                    rule = self.rule_instances[rule_type]
                    yield rule, matched_action
            if rule_type in self.collection_instances.keys():
                # If the rule is a RuleCollection, call its 'match' method - recursion?
                yield from self.collection_instances[rule_type].match(node)
        return


@dataclass(frozen=True)
class _SourceFromIterator:
    """A match of a rule against a source file.

    :param rule: The rule that matched
    :param action: The action or action iterator that was returned by the rule
    :param source_code: source to which apply the actions
    :return: A updated source
    """

    rule: Rule
    action: BaseAction | Iterator[BaseAction]
    source_code: str
    enable_optimizations: bool = field(default=True)

    def __post_init__(self):
        if not isinstance(self.source_code, str) or not self.source_code:
            raise TypeError("source_code must be a non-empty string")
        if not self.action:
            raise TypeError("action cannot be None")

    def source(self) -> str:
        """Compute the path of the current node (against the starting tree).

        Adjust this path with the knowledge from the previously applied
        actions.

        Use the path to find the correct node in the new tree."""

        from refactor.internal.graph_access import AccessFailure, GraphPath

        shifts: List[Tuple[GraphPath, int]] = []
        updated_source = self.source_code
        previous_tree = self.rule.context.tree
        previous_action: BaseAction = None
        for action in self.action:
            input_node, stack_effect = action.stack_effect()

            # We compute each path against the initial revision of the tree
            # since the rule who is producing them doesn't have access to the
            # temporary trees we generate on the fly.
            path = GraphPath.backtrack_from(self.rule.context, input_node)

            # And due to this, some actions might have altered the tree in a
            # way that makes the path as is invalid. For ensuring that the path
            # now reflects the current state of the tree, we apply all the shifts
            # that the previous actions have caused.
            path = path.shift(shifts)

            # With the updated path, we can now find the same node in the new
            # tree. This allows us to know the exact position of the node.
            try:
                updated_input = path.execute(previous_tree)
            except AccessFailure:
                raise MaybeOverlappingActions(
                    "When using chained actions, individual actions should not"
                    " overlap with each other."
                    f"\n    Previous action: {previous_action}"
                    f"\n   Action attempted: {action}"
                    f"\n       Current Path: {GraphPath.backtrack_from(self.rule.context, input_node)}"
                    f"\n               Path: {path}"
                    f"\n             Shifts: {shifts}"
                    f"\n              Stack: {stack_effect}"
                    f"\n             P Tree: {previous_tree}"
                ) from None
            else:
                shifts.append((path, stack_effect))

            updated_action: BaseAction = action.replace_input(updated_input)
            updated_context: Context = self.rule.context.replace(source=updated_source, tree=previous_tree)

            # TODO: re-enable optimizations if it is viable to run them on the new tree/source code.
            updated_source: str = _SourceFromAction(self.rule,
                                                    updated_action,
                                                    updated_source,
                                                    context=updated_context,
                                                    enable_optimizations=False,
                                                    ).source()

            try:
                previous_tree = ast.parse(updated_source)
            except SyntaxError as exc:
                _unparsable_source_code(updated_source, exc)
            previous_action = action
        return updated_source


@dataclass(frozen=True)
class _SourceFromAction(_SourceFromIterator):
    """A match of a rule against a source file."""

    context: Context = field(default=None)

    def __post_init__(self):
        if not isinstance(self.context, Context):
            raise TypeError("context must be a Context instance")

    def source(self) -> str:
        """Apply a single action to the source"""
        if isinstance(action := self.action, Iterator):
            return super().source()

        if self.enable_optimizations:
            action = optimize(self.action, self.context)
        source: str = action.apply(self.context, self.source_code)
        return source


@dataclass(frozen=True)
class _SourceFromRuleOrCollection:
    rule_or_collection: Rule | RuleCollection

    _indent: str = field(default="")

    def source(self, node, source) -> str:
        """Mixes the method of creating the source update between BaseAction and Iterator[BaseAction]."""
        new_source: str = source
        for rule, action in _match_from_rule_or_collection(self.rule_or_collection, node):
            if action is None:
                continue
            builder: _SourceFromAction = _SourceFromAction(rule, action, new_source, context=rule.context)

            with suppress(AssertionError):
                new_source: str = builder.source()

            # Yield if source has changed, otherwise we continue to the next rule.
            if new_source is not None and new_source != "":
                yield new_source
            else:
                return source
        return source


@dataclass
class Session:
    """A refactoring session that consists of a set of rules and a configuration."""
    c_current_config: ClassVar[Configuration]

    rules: list[str | type[Rule] | type[RuleCollection]] = field(default_factory=list)
    config: Configuration = field(default_factory=Configuration)

    def __post_init__(self):
        # The frame for the session call will be up from __post_init__, __init__
        self.calling_module = sys.modules[inspect.currentframe().f_back.f_back.f_globals["__name__"]]

    def _import_named_rules(self):
        """Import all rules from the given module."""
        module = inspect.getmodule(self.calling_module)
        is_rule_class: Callable = lambda n: issubclass(n, Rule) and n is not Rule
        is_rule_collection_class: Callable = lambda n: issubclass(n, RuleCollection) and n is not RuleCollection
        is_member: Callable = lambda n: inspect.isclass(n) and (is_rule_class(n) or is_rule_collection_class(n))

        for i, rule in enumerate(self.rules):
            if isinstance(rule, str):
                rule_or_collection_class = next((m for m_name, m in inspect.getmembers(module)
                                                 if is_member(m) and m_name == rule), None)

                if rule_or_collection_class is None:
                    self.rules[i] = None
                    continue

                rule = rule_or_collection_class
                self.rules[i] = rule

            if issubclass(rule, RuleCollection):
                rule.import_named_rules()

        self.rules = [rule for rule in self.rules if rule is not None]

    def _initialize_rules(
            self,
            tree: ast.Module,
            source: str,
            file_info: _FileInfo,
    ) -> list[Rule]:
        """Initialize all the rules in the session. This is done by calling the ``initialize`` method on each rule. """
        Session.c_current_config = self.config

        # Rudimentary rule importer for string-defined rules
        self._import_named_rules()

        context = Context._from_dependencies(
            _resolve_dependencies(self.rules),  # type: ignore
            tree=tree,
            source=source,
            file_info=file_info,
            config=self.config,
        )
        instances: list[Rule | RuleCollection] = []
        for rule_or_collection in self.rules:
            if issubclass(rule_or_collection, RuleCollection):
                # We want to initialize the rules, but keep the rules grouped for intermediate tree update
                (collection := rule_or_collection()).initialize_rules(context, file_info.path)
                if collection.rule_instances and len(collection.rule_instances) > 0:
                    instances.append(collection)
            else:
                instances.append(rule_or_collection(context))
        return [i for i in instances if i.check_file(file_info.path)]

    def _run(
            self,
            source: str,
            file_info: _FileInfo,
            *,
            _changed: bool = False,
            _known_sources: frozenset[str] = frozenset(),
    ) -> Tuple[str, bool]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            if not _changed:
                return source, _changed
            return _unparsable_source_code(source, exc)

        _known_sources |= {source}
        rules = self._initialize_rules(tree, source, file_info)

        for node in ast.walk(tree):
            if not has_positions(type(node)):  # type: ignore
                continue

            for rule_or_collection in rules:
                for new_source in _SourceFromRuleOrCollection(rule_or_collection).source(node, source):
                    if new_source not in _known_sources:
                        return self._run(
                            new_source,
                            file_info,
                            _changed=True,
                            _known_sources=_known_sources,
                        )

        return source, _changed

    def run(self, source: str) -> str:
        """Apply all the rules from this session to the given ``source``
        and return the transformed version.

        In case of the given `source` is not parsable, it will return
        it unchanged.
        """

        source, _ = self._run(source, file_info=_FileInfo())
        return source

    def run_file(self, file: Path) -> Change | None:
        """Apply all the rules from this session to the given ``file``
        and return a :class:`refactor.Change` if any changes were made.

        In case of the given file is not parsable, it will return `None`.
        """

        try:
            with tokenize.open(file) as stream:
                source = stream.read()
                encoding = stream.encoding
        except (SyntaxError, UnicodeDecodeError):
            return None

        file_info = _FileInfo(file, encoding)
        new_source, is_changed = self._run(source, file_info)

        if is_changed:
            return Change(file_info, source, new_source)
