from typing import Protocol, runtime_checkable


@runtime_checkable
class Connector(Protocol):
    """Protocol for exchange connectors."""

    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    async def get_balance(self) -> dict: ...
