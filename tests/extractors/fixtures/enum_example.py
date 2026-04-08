from enum import Enum, IntEnum


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    LIMIT_MAKER = "limit_maker"


class TradeType(IntEnum):
    BUY = 1
    SELL = 2
