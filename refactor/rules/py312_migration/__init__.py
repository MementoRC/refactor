"""Python 3.12 modernization rules for the `refactor-py312` CLI.

This package is consumed by Hummingbot's `custom_git_setup/scripts/hummingbot-branch-tracking.sh`
to modernize upstream/development sources to py3.12 idioms on every cron rebuild.

Phase 1 rules (CORE_RULES) handle py3.12 breaking removals.
Phase 2 rules (IDIOMATIC_RULES) handle non-breaking modernization (added in PR #2).
Phase 3 rules (OPTIN_RULE_GROUPS) are high-risk/semantic-changing and opt-in only (added in PR #3).
"""

from __future__ import annotations

from refactor.rules.py312_migration.asyncio_modern import (
    AsyncioEnsureFutureRule,
    AsyncioGetEventLoopRule,
    AsyncioWaitForToTimeoutRule,
)
from refactor.rules.py312_migration.datetime_modern import (
    DatetimeUtcfromtimestampRule,
    DatetimeUtcnowRule,
)
from refactor.rules.py312_migration.distutils import (
    DistutilsCommandRule,
    DistutilsCoreSetupRule,
    DistutilsLogRule,
    DistutilsSpawnRule,
    DistutilsSysconfigRule,
    DistutilsUtilStrtoboolRule,
    DistutilsVersionRule,
)
from refactor.rules.py312_migration.enum_modern import (
    IntEnumRule,
    StrEnumRule,
)
from refactor.rules.py312_migration.functools_modern import LruCacheToCacheRule
from refactor.rules.py312_migration.inspect_modern import (
    InspectFormatargspecRule,
    InspectGetargspecRule,
)
from refactor.rules.py312_migration.open_encoding import OpenEncodingRule
from refactor.rules.py312_migration.override_decorator import OverrideDecoratorRule
from refactor.rules.py312_migration.stdlib_additions import BatchedRule, PairwiseRule
from refactor.rules.py312_migration.stdlib_removed import RemovedStdlibImportRule
from refactor.rules.py312_migration.typing_modern import (
    EnsureFutureAnnotationsImportRule,
    PEP695GenericClassRule,
    PEP695TypeAliasRule,
    TypingDeprecatedAliasRule,
    TypingOptionalRule,
    TypingTypeRule,
)

CORE_RULES = [
    # distutils removal (py3.12 removes distutils entirely)
    DistutilsCoreSetupRule,
    DistutilsCommandRule,
    DistutilsVersionRule,
    DistutilsSpawnRule,
    DistutilsUtilStrtoboolRule,
    DistutilsSysconfigRule,
    DistutilsLogRule,
    # PEP 594 stdlib removals
    RemovedStdlibImportRule,
    # datetime utcnow deprecation
    DatetimeUtcnowRule,
    DatetimeUtcfromtimestampRule,
    # asyncio get_event_loop deprecation in async contexts
    AsyncioGetEventLoopRule,
    # inspect.getargspec removal
    InspectGetargspecRule,
    InspectFormatargspecRule,
]

IDIOMATIC_RULES = [
    # functools cache modernization
    LruCacheToCacheRule,
    # asyncio create_task in async contexts
    AsyncioEnsureFutureRule,
    # enum mixins -> StrEnum / IntEnum
    StrEnumRule,
    IntEnumRule,
    # typing aliases -> builtin generics (pyupgrade edge cases)
    TypingDeprecatedAliasRule,
    TypingTypeRule,
    TypingOptionalRule,
    EnsureFutureAnnotationsImportRule,
]

OPTIN_RULE_GROUPS = {
    "asyncio-timeout": [AsyncioWaitForToTimeoutRule],
    "pep695-types": [PEP695TypeAliasRule],
    "pep695-generics": [PEP695GenericClassRule],
    "override-decorator": [OverrideDecoratorRule],
    "encoding-warning": [OpenEncodingRule],
    "itertools-modern": [PairwiseRule, BatchedRule],
}

__all__ = [
    "AsyncioEnsureFutureRule",
    "AsyncioGetEventLoopRule",
    "AsyncioWaitForToTimeoutRule",
    "BatchedRule",
    "DatetimeUtcfromtimestampRule",
    "DatetimeUtcnowRule",
    "DistutilsCommandRule",
    "DistutilsCoreSetupRule",
    "DistutilsLogRule",
    "DistutilsSpawnRule",
    "DistutilsSysconfigRule",
    "DistutilsUtilStrtoboolRule",
    "DistutilsVersionRule",
    "EnsureFutureAnnotationsImportRule",
    "InspectFormatargspecRule",
    "InspectGetargspecRule",
    "IntEnumRule",
    "LruCacheToCacheRule",
    "OpenEncodingRule",
    "OverrideDecoratorRule",
    "PEP695GenericClassRule",
    "PEP695TypeAliasRule",
    "PairwiseRule",
    "RemovedStdlibImportRule",
    "StrEnumRule",
    "TypingDeprecatedAliasRule",
    "TypingOptionalRule",
    "TypingTypeRule",
    "CORE_RULES",
    "IDIOMATIC_RULES",
    "OPTIN_RULE_GROUPS",
]
