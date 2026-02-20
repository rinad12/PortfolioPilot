from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, AnyUrl, field_validator
import pycountry
import re


# Enums for categorical fields
class CategoryType(str, Enum):
    """Data category type"""
    MARKET = "market"
    MACRO = "macro"
    NEWS = "news"

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
    category: CategoryType = Field(..., description="Data category")
    created_at: datetime = Field(..., description="ISO-8601 datetime")



# Market Data Payload
class MarketData(BaseRecord):
    """Market data from Yahoo Finance"""
    
    category: CategoryType = Field(default=CategoryType.MARKET, description="Category type")
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    name: Optional[str] = Field(None, description="Company/asset name")
    sector: Optional[str] = Field(None, description="Sector classification")
    country: str = Field(..., description="Country code")
    price: float = Field(..., gt = 0, description="Current price")
    currency: str = Field(..., description="Currency (e.g., USD)")
    change_abs: Optional[float] = Field(None, description="Absolute price change")
    change_pct: Optional[float] = Field(None, description="Percentage price change")
    volume: int = Field(..., description="Trading volume")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    volatility_100d: Optional[float] = Field(None, gt = 0, description="100-day volatility")

    @field_validator('country')
    @classmethod
    def validate_country(cls, v: str) -> str:
        v = v.upper().strip()
        
        if not v.isascii() or not v.isalpha():
            raise ValueError("Country code must be English letters only")
        

        if len(v) != 2:
            raise ValueError("Country code must be exactly 2 letters (ISO 3166-1 alpha-2)")
        

        country = pycountry.countries.get(alpha_2=v)
        if not country:
            raise ValueError(f"Invalid country code: {v}. Must be valid ISO 3166-1 alpha-2 code")
        
        return v

    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v: str) -> str:
        v = v.upper().strip()
        
        if not v.isascii() or not v.isalpha():
            raise ValueError("Currency code must contain only English letters")

        if len(v) != 3:
            raise ValueError("Currency code must be exactly 3 letters (ISO 4217)")
        
 
        currency = pycountry.currencies.get(alpha_3=v)
        if not currency:
            raise ValueError(f"Invalid currency code: {v}. Must be valid ISO 4217 code")
        
        return v
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v = v.upper().strip()
        
        if not v.isascii():
            raise ValueError("Symbol must contain only English characters")
        
        if not re.match(r'^[A-Z0-9][A-Z0-9\.\-]{0,9}$', v):
            raise ValueError(
                "Invalid symbol format. Should be 1-10 characters, "
                "English letters, numbers, dots or hyphens (e.g., AAPL, BRK.B)"
            )
        
        if len(v) < 1 or len(v) > 10:
            raise ValueError("Symbol must be 1-10 characters long")
        
        return v
    
    


# Macro Data Payload
class MacroData(BaseRecord):
    """Macroeconomic data from FRED"""
    
    category: CategoryType = Field(default=CategoryType.MACRO, description="Category type")
    indicator_id: str = Field(..., description="Indicator code (e.g., CPIAUCSL)")
    indicator_name: str = Field(..., description="Indicator name")
    indicator_type: MacroType = Field(..., description="Type of macro indicator")
    value: float = Field(..., description="Indicator value")
    unit: str = Field(..., description="Measurement unit")
    frequency: Frequency = Field(..., description="Data frequency")
    country: str = Field(..., description="Country code")
    policy_relevance: PolicyRelevance = Field(..., description="Policy relevance level")
    narrative_role: NarrativeRole = Field(..., description="Role in economic narrative")

    @field_validator('country')
    @classmethod
    def validate_country(cls, v: str) -> str:
        v = v.upper().strip()
        
        if not v.isascii() or not v.isalpha():
            raise ValueError("Country code must be English letters only")
        

        if len(v) != 2:
            raise ValueError("Country code must be exactly 2 letters (ISO 3166-1 alpha-2)")
        

        country = pycountry.countries.get(alpha_2=v)
        if not country:
            raise ValueError(f"Invalid country code: {v}. Must be valid ISO 3166-1 alpha-2 code")
        
        return v


# News Data Payload
class NewsData(BaseRecord):
    """News data"""
    
    category: CategoryType = Field(default=CategoryType.NEWS, description="Category type")
    headline: str = Field(..., description="News headline")
    summary: str = Field(..., description="News summary/body")
    publisher: str = Field(..., description="News publisher")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: AnyUrl = Field(..., description="URL to news article")
    related_assets: List[str] = Field(default_factory=list, description="Related asset symbols")
    sectors: List[str] = Field(default_factory=list, description="Related sectors")
    sentiment_label: SentimentLabel = Field(..., description="Sentiment classification")
    event_type: EventType = Field(..., description="Type of event")
    relevance: Relevance = Field(..., description="Relevance level")

    @field_validator('related_assets')
    @classmethod
    def validate_related_assets(cls, v: List[str]) -> List[str]:
        validated = []
        
        for ticker in v:
            ticker = ticker.upper().strip()
            

            if not ticker.isascii():
                raise ValueError(f"Ticker {ticker} must be English characters only")
            
            if not re.match(r'^[A-Z0-9][A-Z0-9\.\-]{0,9}$', ticker):
                raise ValueError(
                    f"Invalid ticker format: {ticker}. "
                    f"Expected 1-10 characters: letters, numbers, dots, hyphens"
                )
            
            validated.append(ticker)
        
        return validated
