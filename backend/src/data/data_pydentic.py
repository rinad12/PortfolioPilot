from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


# Enums for categorical fields
class SourceType(str, Enum):
    """Data source type"""
    YAHOO_FINANCE = "yahoo_finance"
    FRED = "fred"
    NEWS = "news"


class CategoryType(str, Enum):
    """Data category type"""
    MARKET = "market"
    MACRO = "macro"
    NEWS = "news"


class EntityType(str, Enum):
    """Entity type"""
    ASSET = "asset"
    INDICATOR = "indicator"
    EVENT = "event"


class IntentType(str, Enum):
    """Intent type for data usage"""
    EXPLANATION = "explanation"
    RISK_CONTEXT = "risk_context"
    POLICY_CONTEXT = "policy_context"


class MacroType(str, Enum):
    """Macro indicator type"""
    INFLATION = "inflation"
    UNEMPLOYMENT = "unemployment"
    GDP = "gdp"
    INTEREST_RATE = "interest_rate"
    TRADE = "trade"


class SentimentLabel(str, Enum):
    """Sentiment classification for news"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EventType(str, Enum):
    """Event type for news data"""
    EARNINGS = "earnings"
    MERGER = "merger"
    REGULATORY = "regulatory"
    ECONOMIC = "economic"
    GEOPOLITICAL = "geopolitical"
    OTHER = "other"


class Frequency(str, Enum):
    """Data frequency for macro indicators"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class PolicyRelevance(str, Enum):
    """Policy relevance level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NarrativeRole(str, Enum):
    """Narrative role in analysis"""
    INFLATION_CONTEXT = "inflation_context"
    GROWTH_CONTEXT = "growth_context"
    EMPLOYMENT_CONTEXT = "employment_context"
    MONETARY_POLICY = "monetary_policy"
    RISK_INDICATOR = "risk_indicator"


class Relevance(str, Enum):
    """Relevance level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Base Record
class BaseRecord(BaseModel):
    """Base record with common fields for all data categories"""
    
    id: str = Field(..., description="Unique identifier")
    source: SourceType = Field(..., description="Data source")
    category: CategoryType = Field(..., description="Data category")
    created_at: datetime = Field(..., description="ISO-8601 datetime")
    entity_type: EntityType = Field(..., description="Entity type")
    intent: IntentType = Field(..., description="Intent for data usage")



# Market Data Payload
class MarketData(BaseRecord):
    """Market data from Yahoo Finance"""
    
    category: CategoryType = Field(default=CategoryType.MARKET, description="Category type")
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    name: Optional[str] = Field(None, description="Company/asset name")
    sector: Optional[str] = Field(None, description="Sector classification")
    price: float = Field(..., description="Current price")
    currency: str = Field(..., description="Currency (e.g., USD)")
    change_abs: Optional[float] = Field(None, description="Absolute price change")
    change_pct: Optional[float] = Field(None, description="Percentage price change")
    volume: int = Field(..., description="Trading volume")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    volatility_100d: Optional[float] = Field(None, description="100-day volatility")


# Macro Data Payload
class MacroData(BaseRecord):
    """Macroeconomic data from FRED"""
    
    category: CategoryType = Field(default=CategoryType.MACRO, description="Category type")
    indicator_id: str = Field(..., alias="id", description="Indicator code (e.g., CPIAUCSL)")
    indicator_name: str = Field(..., description="Indicator name")
    indicator_type: MacroType = Field(..., description="Type of macro indicator")
    value: float = Field(..., description="Indicator value")
    unit: str = Field(..., description="Measurement unit")
    frequency: Frequency = Field(..., description="Data frequency")
    country: str = Field(..., description="Country code")
    policy_relevance: PolicyRelevance = Field(..., description="Policy relevance level")
    narrative_role: NarrativeRole = Field(..., description="Role in economic narrative")


# News Data Payload
class NewsData(BaseRecord):
    """News data"""
    
    category: CategoryType = Field(default=CategoryType.NEWS, description="Category type")
    headline: str = Field(..., description="News headline")
    summary: str = Field(..., description="News summary/body")
    publisher: str = Field(..., description="News publisher")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: str = Field(..., description="URL to news article")
    related_assets: List[str] = Field(default_factory=list, description="Related asset symbols")
    sectors: List[str] = Field(default_factory=list, description="Related sectors")
    sentiment_label: SentimentLabel = Field(..., description="Sentiment classification")
    event_type: EventType = Field(..., description="Type of event")
    relevance: Relevance = Field(..., description="Relevance level")

