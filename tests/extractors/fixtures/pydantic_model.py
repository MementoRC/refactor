from pydantic import BaseModel, Field, validator


class TradingConfig(BaseModel):
    """Configuration for trading."""

    trading_pair: str = Field(description="Trading pair")
    amount: float = Field(default=1.0, description="Trade amount")
    enabled: bool = True

    @validator("trading_pair")
    def validate_pair(cls, v):
        return v.upper()
