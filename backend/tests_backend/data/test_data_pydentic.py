import pytest
from datetime import datetime
from pydantic import ValidationError

from data.data_pydentic import (
    # Enums
    CategoryType,
    MacroType,
    SentimentLabel,
    EventType,
    Frequency,
    PolicyRelevance,
    NarrativeRole,
    Relevance,
    # Models
    BaseRecord,
    MarketData,
    MacroData,
    NewsData,
)


# -------------------------------------------------------------------------
# Test Enums
# -------------------------------------------------------------------------

class TestEnums:
    """Test all enum classes"""

    def test_category_type_enum(self):
        """Test CategoryType enum values"""
        assert CategoryType.MARKET == "market"
        assert CategoryType.MACRO == "macro"
        assert CategoryType.NEWS == "news"

    def test_macro_type_enum(self):
        """Test MacroType enum values"""
        assert MacroType.INFLATION == "inflation"
        assert MacroType.UNEMPLOYMENT == "unemployment"
        assert MacroType.GDP == "gdp"
        assert MacroType.INTEREST_RATE == "interest_rate"
        assert MacroType.TRADE == "trade"

    def test_sentiment_label_enum(self):
        """Test SentimentLabel enum values"""
        assert SentimentLabel.POSITIVE == "positive"
        assert SentimentLabel.NEGATIVE == "negative"
        assert SentimentLabel.NEUTRAL == "neutral"

    def test_event_type_enum(self):
        """Test EventType enum values"""
        assert EventType.EARNINGS == "earnings"
        assert EventType.MERGER == "merger"
        assert EventType.REGULATORY == "regulatory"
        assert EventType.ECONOMIC == "economic"
        assert EventType.GEOPOLITICAL == "geopolitical"
        assert EventType.OTHER == "other"

    def test_frequency_enum(self):
        """Test Frequency enum values"""
        assert Frequency.DAILY == "daily"
        assert Frequency.WEEKLY == "weekly"
        assert Frequency.MONTHLY == "monthly"
        assert Frequency.QUARTERLY == "quarterly"
        assert Frequency.ANNUAL == "annual"

    def test_policy_relevance_enum(self):
        """Test PolicyRelevance enum values"""
        assert PolicyRelevance.LOW == "low"
        assert PolicyRelevance.MEDIUM == "medium"
        assert PolicyRelevance.HIGH == "high"

    def test_narrative_role_enum(self):
        """Test NarrativeRole enum values"""
        assert NarrativeRole.INFLATION_CONTEXT == "inflation_context"
        assert NarrativeRole.GROWTH_CONTEXT == "growth_context"
        assert NarrativeRole.EMPLOYMENT_CONTEXT == "employment_context"
        assert NarrativeRole.MONETARY_POLICY == "monetary_policy"
        assert NarrativeRole.RISK_INDICATOR == "risk_indicator"

    def test_relevance_enum(self):
        """Test Relevance enum values"""
        assert Relevance.LOW == "low"
        assert Relevance.MEDIUM == "medium"
        assert Relevance.HIGH == "high"


# -------------------------------------------------------------------------
# Test BaseRecord
# -------------------------------------------------------------------------

class TestBaseRecord:
    """Test BaseRecord class"""

    def test_base_record_valid(self):
        """Test valid BaseRecord instantiation"""
        record = BaseRecord(
            category=CategoryType.MARKET,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
        )

        assert record.category == CategoryType.MARKET

    def test_base_record_missing_required_fields(self):
        """Test BaseRecord with missing required fields"""
        with pytest.raises(ValidationError) as exc_info:
            BaseRecord()

        errors = exc_info.value.errors()
        assert len(errors) == 2  # category and created_at


# -------------------------------------------------------------------------
# Test MarketData
# -------------------------------------------------------------------------

class TestMarketData:
    """Test MarketData class with validators"""

    def test_market_data_valid(self):
        """Test valid MarketData instantiation"""
        data = MarketData(
            category=CategoryType.MARKET,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            country="US",
            price=150.50,
            currency="USD",
            change_abs=2.5,
            change_pct=1.69,
            volume=1000000,
            market_cap=2500000000.0,
            volatility_100d=0.25,
        )

        assert data.symbol == "AAPL"
        assert data.price == 150.50
        assert data.currency == "USD"
        assert data.volume == 1000000
        assert data.country == "US"

    def test_market_data_symbol_validation_valid(self):
        """Test valid symbol formats"""
        valid_symbols = ["AAPL", "BRK.B", "GOOG", "TSM", "A", "SPY", "QQQ"]

        for symbol in valid_symbols:
            data = MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol=symbol,
                country="US",
                price=100.0,
                currency="USD",
                volume=1000,
            )
            assert data.symbol == symbol.upper()

    def test_market_data_symbol_validation_lowercase(self):
        """Test symbol validation converts to uppercase"""
        data = MarketData(
            category=CategoryType.MARKET,
            created_at=datetime.now(),
            symbol="aapl",
            country="US",
            price=100.0,
            currency="USD",
            volume=1000,
        )
        assert data.symbol == "AAPL"

    def test_market_data_symbol_validation_with_spaces(self):
        """Test symbol validation trims spaces"""
        data = MarketData(
            category=CategoryType.MARKET,
            created_at=datetime.now(),
            symbol="  AAPL  ",
            country="US",
            price=100.0,
            currency="USD",
            volume=1000,
        )
        assert data.symbol == "AAPL"

    def test_market_data_symbol_validation_too_long(self):
        """Test symbol validation rejects symbols longer than 10 characters"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="VERYLONGSYMBOL",  # 14 characters
                country="US",
                price=100.0,
                currency="USD",
                volume=1000,
            )

        assert "Invalid symbol format" in str(exc_info.value)

    def test_market_data_symbol_validation_invalid_chars(self):
        """Test symbol validation rejects invalid characters"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAP$",  # Invalid character $
                country="US",
                price=100.0,
                currency="USD",
                volume=1000,
            )

        assert "Invalid symbol format" in str(exc_info.value)

    def test_market_data_symbol_validation_non_ascii(self):
        """Test symbol validation rejects non-ASCII characters"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="АААА",  # Cyrillic characters
                country="US",
                price=100.0,
                currency="USD",
                volume=1000,
            )

        assert "Symbol must contain only English characters" in str(exc_info.value)

    def test_market_data_currency_validation_valid(self):
        """Test valid currency codes"""
        valid_currencies = ["USD", "EUR", "GBP", "JPY", "CNY"]

        for currency in valid_currencies:
            data = MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="US",
                price=100.0,
                currency=currency,
                volume=1000,
            )
            assert data.currency == currency.upper()

    def test_market_data_currency_validation_lowercase(self):
        """Test currency validation converts to uppercase"""
        data = MarketData(
            category=CategoryType.MARKET,
            created_at=datetime.now(),
            symbol="AAPL",
            country="US",
            price=100.0,
            currency="usd",
            volume=1000,
        )
        assert data.currency == "USD"

    def test_market_data_currency_validation_invalid(self):
        """Test currency validation rejects invalid codes"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="US",
                price=100.0,
                currency="ZZZ",  # Invalid ISO 4217 code
                volume=1000,
            )

        assert "Invalid currency code" in str(exc_info.value)

    def test_market_data_currency_validation_wrong_length(self):
        """Test currency validation rejects wrong length"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="US",
                price=100.0,
                currency="US",  # Only 2 characters
                volume=1000,
            )

        assert "Currency code must be exactly 3 letters" in str(exc_info.value)

    def test_market_data_currency_validation_non_alpha(self):
        """Test currency validation rejects non-alphabetic characters"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="US",
                price=100.0,
                currency="U$D",
                volume=1000,
            )

        assert "Currency code must contain only English letters" in str(exc_info.value)

    def test_market_data_price_validation_positive(self):
        """Test price must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="US",
                price=0.0,  # Not allowed (gt=0)
                currency="USD",
                volume=1000,
            )

        assert "greater than 0" in str(exc_info.value).lower()

    def test_market_data_price_validation_negative(self):
        """Test price cannot be negative"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="US",
                price=-10.0,
                currency="USD",
                volume=1000,
            )

        assert "greater than 0" in str(exc_info.value).lower()

    def test_market_data_change_abs_positive(self):
        """Test change_abs must be greater than 0 when provided"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="US",
                price=100.0,
                currency="USD",
                volume=1000,
                change_abs=-5.0,  # Cannot be negative
            )

        assert "greater than 0" in str(exc_info.value).lower()

    def test_market_data_volatility_positive(self):
        """Test volatility_100d must be greater than 0 when provided"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="US",
                price=100.0,
                currency="USD",
                volume=1000,
                volatility_100d=-0.1,  # Cannot be negative
            )

        assert "greater than 0" in str(exc_info.value).lower()

    def test_market_data_country_validation_valid(self):
        """Test valid ISO 3166-1 alpha-2 country codes"""
        valid_countries = ["US", "GB", "DE", "JP", "CN"]

        for country in valid_countries:
            data = MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country=country,
                price=100.0,
                currency="USD",
                volume=1000,
            )
            assert data.country == country.upper()

    def test_market_data_country_validation_lowercase(self):
        """Test country code is normalized to uppercase"""
        data = MarketData(
            category=CategoryType.MARKET,
            created_at=datetime.now(),
            symbol="AAPL",
            country="us",
            price=100.0,
            currency="USD",
            volume=1000,
        )
        assert data.country == "US"

    def test_market_data_country_validation_invalid_code(self):
        """Test country validation rejects invalid ISO codes"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="XX",  # Not a valid ISO 3166-1 alpha-2 code
                price=100.0,
                currency="USD",
                volume=1000,
            )

        assert "Invalid country code" in str(exc_info.value)

    def test_market_data_country_validation_wrong_length(self):
        """Test country validation rejects codes that are not 2 letters"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="USA",  # 3 letters instead of 2
                price=100.0,
                currency="USD",
                volume=1000,
            )

        assert "Country code must be exactly 2 letters" in str(exc_info.value)

    def test_market_data_country_validation_non_alpha(self):
        """Test country validation rejects non-alphabetic characters"""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                category=CategoryType.MARKET,
                created_at=datetime.now(),
                symbol="AAPL",
                country="U1",
                price=100.0,
                currency="USD",
                volume=1000,
            )

        assert "Country code must be English letters only" in str(exc_info.value)


# -------------------------------------------------------------------------
# Test MacroData
# -------------------------------------------------------------------------

class TestMacroData:
    """Test MacroData class"""

    def test_macro_data_valid(self):
        """Test valid MacroData instantiation"""
        data = MacroData(
            id="CPIAUCSL",
            category=CategoryType.MACRO,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            indicator_name="Consumer Price Index",
            indicator_type=MacroType.INFLATION,
            value=305.5,
            unit="Index 1982-1984=100",
            frequency=Frequency.MONTHLY,
            country="US",
            policy_relevance=PolicyRelevance.HIGH,
            narrative_role=NarrativeRole.INFLATION_CONTEXT,
        )

        assert data.indicator_id == "CPIAUCSL"
        assert data.indicator_name == "Consumer Price Index"
        assert data.indicator_type == MacroType.INFLATION
        assert data.value == 305.5
        assert data.frequency == Frequency.MONTHLY

    def test_macro_data_missing_required_fields(self):
        """Test MacroData with missing required fields"""
        with pytest.raises(ValidationError) as exc_info:
            MacroData(
                id="CPIAUCSL",
                category=CategoryType.MACRO,
                created_at=datetime.now(),
                # Missing: indicator_name, indicator_type, value, unit, frequency, country, policy_relevance, narrative_role
            )

        errors = exc_info.value.errors()
        assert len(errors) > 0

    def test_macro_data_country_validation_valid(self):
        """Test valid country codes for MacroData"""
        data = MacroData(
            id="CPIAUCSL",
            category=CategoryType.MACRO,
            created_at=datetime.now(),
            indicator_name="Consumer Price Index",
            indicator_type=MacroType.INFLATION,
            value=305.5,
            unit="Index 1982-1984=100",
            frequency=Frequency.MONTHLY,
            country="de",  # lowercase should be normalized
            policy_relevance=PolicyRelevance.HIGH,
            narrative_role=NarrativeRole.INFLATION_CONTEXT,
        )

        assert data.country == "DE"

    def test_macro_data_country_validation_invalid(self):
        """Test MacroData rejects invalid country codes"""
        with pytest.raises(ValidationError) as exc_info:
            MacroData(
                id="CPIAUCSL",
                category=CategoryType.MACRO,
                created_at=datetime.now(),
                indicator_name="Consumer Price Index",
                indicator_type=MacroType.INFLATION,
                value=305.5,
                unit="Index 1982-1984=100",
                frequency=Frequency.MONTHLY,
                country="XX",  # Invalid code
                policy_relevance=PolicyRelevance.HIGH,
                narrative_role=NarrativeRole.INFLATION_CONTEXT,
            )

        assert "Invalid country code" in str(exc_info.value)


# -------------------------------------------------------------------------
# Test NewsData
# -------------------------------------------------------------------------

class TestNewsData:
    """Test NewsData class with validators"""

    def test_news_data_valid(self):
        """Test valid NewsData instantiation"""
        data = NewsData(
            category=CategoryType.NEWS,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            headline="Apple announces record earnings",
            summary="Apple Inc. reported record quarterly earnings...",
            publisher="Reuters",
            published_at=datetime(2024, 1, 1, 10, 0, 0),
            url="https://example.com/news/apple-earnings",
            related_assets=["AAPL", "MSFT"],
            sectors=["Technology"],
            sentiment_label=SentimentLabel.POSITIVE,
            event_type=EventType.EARNINGS,
            relevance=Relevance.HIGH,
        )

        assert data.headline == "Apple announces record earnings"
        assert data.publisher == "Reuters"
        assert data.sentiment_label == SentimentLabel.POSITIVE
        assert data.related_assets == ["AAPL", "MSFT"]

    def test_news_data_url_validation_valid(self):
        """Test valid URL formats"""
        data = NewsData(
            category=CategoryType.NEWS,
            created_at=datetime.now(),
            headline="Test headline",
            summary="Test summary",
            publisher="Test Publisher",
            published_at=datetime.now(),
            url="https://example.com/article",
            sentiment_label=SentimentLabel.NEUTRAL,
            event_type=EventType.OTHER,
            relevance=Relevance.MEDIUM,
        )

        assert str(data.url) == "https://example.com/article"

    def test_news_data_url_validation_invalid(self):
        """Test invalid URL format"""
        with pytest.raises(ValidationError) as exc_info:
            NewsData(
                category=CategoryType.NEWS,
                created_at=datetime.now(),
                headline="Test headline",
                summary="Test summary",
                publisher="Test Publisher",
                published_at=datetime.now(),
                url="not-a-valid-url",
                sentiment_label=SentimentLabel.NEUTRAL,
                event_type=EventType.OTHER,
                relevance=Relevance.MEDIUM,
            )

        assert "url" in str(exc_info.value).lower()

    def test_news_data_related_assets_validation_valid(self):
        """Test valid related_assets"""
        data = NewsData(
            category=CategoryType.NEWS,
            created_at=datetime.now(),
            headline="Test headline",
            summary="Test summary",
            publisher="Test Publisher",
            published_at=datetime.now(),
            url="https://example.com/article",
            related_assets=["AAPL", "MSFT", "GOOG", "BRK.B"],
            sentiment_label=SentimentLabel.NEUTRAL,
            event_type=EventType.OTHER,
            relevance=Relevance.MEDIUM,
        )

        assert data.related_assets == ["AAPL", "MSFT", "GOOG", "BRK.B"]

    def test_news_data_related_assets_validation_lowercase(self):
        """Test related_assets converts to uppercase"""
        data = NewsData(
            category=CategoryType.NEWS,
            created_at=datetime.now(),
            headline="Test headline",
            summary="Test summary",
            publisher="Test Publisher",
            published_at=datetime.now(),
            url="https://example.com/article",
            related_assets=["aapl", "msft"],
            sentiment_label=SentimentLabel.NEUTRAL,
            event_type=EventType.OTHER,
            relevance=Relevance.MEDIUM,
        )

        assert data.related_assets == ["AAPL", "MSFT"]

    def test_news_data_related_assets_validation_with_spaces(self):
        """Test related_assets trims spaces"""
        data = NewsData(
            category=CategoryType.NEWS,
            created_at=datetime.now(),
            headline="Test headline",
            summary="Test summary",
            publisher="Test Publisher",
            published_at=datetime.now(),
            url="https://example.com/article",
            related_assets=["  AAPL  ", "  MSFT  "],
            sentiment_label=SentimentLabel.NEUTRAL,
            event_type=EventType.OTHER,
            relevance=Relevance.MEDIUM,
        )

        assert data.related_assets == ["AAPL", "MSFT"]

    def test_news_data_related_assets_validation_invalid_format(self):
        """Test related_assets rejects invalid ticker format"""
        with pytest.raises(ValidationError) as exc_info:
            NewsData(
                category=CategoryType.NEWS,
                created_at=datetime.now(),
                headline="Test headline",
                summary="Test summary",
                publisher="Test Publisher",
                published_at=datetime.now(),
                url="https://example.com/article",
                related_assets=["AAPL", "INVALID$TICKER"],
                sentiment_label=SentimentLabel.NEUTRAL,
                event_type=EventType.OTHER,
                relevance=Relevance.MEDIUM,
            )

        assert "Invalid ticker format" in str(exc_info.value)

    def test_news_data_related_assets_validation_non_ascii(self):
        """Test related_assets rejects non-ASCII characters"""
        with pytest.raises(ValidationError) as exc_info:
            NewsData(
                category=CategoryType.NEWS,
                created_at=datetime.now(),
                headline="Test headline",
                summary="Test summary",
                publisher="Test Publisher",
                published_at=datetime.now(),
                url="https://example.com/article",
                related_assets=["AAPL", "АААА"],  # Cyrillic
                sentiment_label=SentimentLabel.NEUTRAL,
                event_type=EventType.OTHER,
                relevance=Relevance.MEDIUM,
            )

        assert "must be English characters only" in str(exc_info.value)

    def test_news_data_empty_related_assets(self):
        """Test NewsData with empty related_assets list"""
        data = NewsData(
            category=CategoryType.NEWS,
            created_at=datetime.now(),
            headline="Test headline",
            summary="Test summary",
            publisher="Test Publisher",
            published_at=datetime.now(),
            url="https://example.com/article",
            related_assets=[],  # Empty list is valid
            sentiment_label=SentimentLabel.NEUTRAL,
            event_type=EventType.OTHER,
            relevance=Relevance.MEDIUM,
        )

        assert data.related_assets == []

    def test_news_data_default_related_assets(self):
        """Test NewsData with default related_assets"""
        data = NewsData(
            category=CategoryType.NEWS,
            created_at=datetime.now(),
            headline="Test headline",
            summary="Test summary",
            publisher="Test Publisher",
            published_at=datetime.now(),
            url="https://example.com/article",
            # related_assets not provided
            sentiment_label=SentimentLabel.NEUTRAL,
            event_type=EventType.OTHER,
            relevance=Relevance.MEDIUM,
        )

        assert data.related_assets == []
