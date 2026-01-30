"""
Constants and enumerations for the Quote-Control simulator.

This module defines the core enums used throughout the platform to ensure
type safety and consistent semantics across different pricing mechanisms.
"""
from enum import Enum, auto

class Side(Enum):
    """Transaction side indicator.

    Attributes:
        BUY: Buyer-initiated transaction (customer purchases, market buy order)
        SELL: Seller-initiated transaction (market sell order, short sale)
    """
    BUY = auto()
    SELL = auto()

class MechanismType(Enum):
    """Pricing mechanism type defining how quotes translate to executions.

    Attributes:
        POSTED_PRICE: Single posted price per instrument (retail dynamic pricing)
        TWO_SIDED_QUOTE: Bid-ask spread quoting (market making, liquidity provision)
        AUCTION: Reserve price or bid shading (ad auctions, marketplaces)
    """
    POSTED_PRICE = auto()
    TWO_SIDED_QUOTE = auto()
    AUCTION = auto()

class InstrumentType(Enum):
    """Type of instrument being priced.

    Attributes:
        SKU: Retail product with inventory constraints
        ASSET: Financial instrument with position limits
        LOAN: Credit product with interest rate pricing
        SUBSCRIPTION: Recurring service with periodic fees
    """
    SKU = auto()
    ASSET = auto()
    LOAN = auto()
    SUBSCRIPTION = auto()

class OpportunityType(Enum):
    """Type of arrival opportunity.

    Attributes:
        SESSION: Retail browsing session with potential purchase intent
        MARKET_ORDER: Financial market order arrival (buy or sell)
        REQUEST: Service or credit request requiring quote response
    """
    SESSION = auto()
    MARKET_ORDER = auto()
    REQUEST = auto()

class EventType(Enum):
    """Type of logged event during simulation.

    Attributes:
        ARRIVAL: New opportunity arrived in the system
        EXPOSURE: Quote was shown to an arrival
        EXECUTION: Transaction was executed
        ABANDON: Opportunity abandoned without execution
        CANCEL: Pending order was cancelled
    """
    ARRIVAL = auto()
    EXPOSURE = auto()
    EXECUTION = auto()
    ABANDON = auto()
    CANCEL = auto()

class LogLevel(Enum):
    """Verbosity level for step logging.

    Attributes:
        NONE: No logging, fastest execution
        AGG_ONLY: Only aggregate statistics per step
        FULL: Full event-level logging with propensities for OPE
    """
    NONE = auto()
    AGG_ONLY = auto()
    FULL = auto()
