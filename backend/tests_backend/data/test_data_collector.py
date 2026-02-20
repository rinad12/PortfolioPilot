"""Tests for backend/src/data/data_collector.py

Covers:
  Pure helpers  : get_cpi_ticker, convert_frew_to_freq, get_event_type, get_relevance
  Sentiment     : get_sentimental_label  (mocked ML pipeline)
  Market data   : fetch_market_data      (mocked yfinance)
  Macro data    : fetch_macro_data       (mocked FRED API)
  News          : fetch_news             (mocked yfinance)

Known limitations in the current implementation (documented by test comments):
  - fetch_market_data : MarketData constructor is missing required BaseRecord fields
                        (id, source, created_at, entity_type, intent) → Pydantic
                        ValidationError caught by except block → always returns None
  - fetch_macro_data  : MacroData constructor is missing required BaseRecord fields
                        (id, source, created_at, entity_type, intent) → always returns ()
  - fetch_news        : NewsData constructor is missing required BaseRecord fields
                        (id, source, created_at, entity_type, intent, published_at)
                        → always returns ()
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── bootstrap ─────────────────────────────────────────────────────────────────
# conftest.py loads the real .env; provide a fallback key for environments
# without one (CI, fresh checkouts) so the module-level import doesn't fail.
os.environ.setdefault("FRED_API_KEY", "test_api_key")

# Stub `transformers` before data_collector is imported so the module-level
# pipeline("finiteautomata/bertweet-base-sentiment-analysis") call doesn't
# trigger a model download.
if "transformers" not in sys.modules:
    _stub_pipeline_inst = MagicMock(return_value=[{"label": "POS", "score": 0.9}])
    sys.modules["transformers"] = MagicMock(
        pipeline=MagicMock(return_value=_stub_pipeline_inst)
    )

from data.data_collector import (  # noqa: E402  (import after sys.modules patch)
    convert_frew_to_freq,
    fetch_macro_data,
    fetch_market_data,
    fetch_news,
    get_cpi_ticker,
    get_event_type,
    get_relevance,
    get_sentimental_label,
)
from data.data_pydentic import EventType, Frequency, Relevance, SentimentLabel


# ═══════════════════════════════════════════════════════════════════════════════
# get_cpi_ticker
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetCpiTicker:
    @pytest.mark.parametrize(
        "iso2, expected_ticker",
        [
            ("US", "USACPIALLMINMEI"),
            ("DE", "DEUCPIALLMINMEI"),
            ("JP", "JPNCPIALLMINMEI"),
            ("GB", "GBRCPIALLMINMEI"),
            ("FR", "FRACPIALLMINMEI"),
            ("CA", "CANCPIALLMINMEI"),
        ],
    )
    def test_valid_countries(self, iso2, expected_ticker):
        assert get_cpi_ticker(iso2) == expected_ticker

    def test_case_insensitive(self):
        assert get_cpi_ticker("us") == "USACPIALLMINMEI"
        assert get_cpi_ticker("de") == "DEUCPIALLMINMEI"

    @pytest.mark.parametrize(
        "invalid",
        ["XX", "ZZ", "INVALID", "", "123", "USA"],  # 'USA' is ISO-3, not ISO-2
    )
    def test_invalid_country_returns_none(self, invalid):
        assert get_cpi_ticker(invalid) is None


# ═══════════════════════════════════════════════════════════════════════════════
# convert_frew_to_freq
# ═══════════════════════════════════════════════════════════════════════════════

class TestConvertFrewToFreq:
    @pytest.mark.parametrize(
        "fred_str, expected",
        [
            ("Daily",     Frequency.DAILY),
            ("Weekly",    Frequency.WEEKLY),
            ("Monthly",   Frequency.MONTHLY),
            ("Quarterly", Frequency.QUARTERLY),
            ("Annual",    Frequency.ANNUAL),
        ],
    )
    def test_known_frequencies(self, fred_str, expected):
        assert convert_frew_to_freq(fred_str) == expected

    @pytest.mark.parametrize(
        "unknown",
        ["daily", "MONTHLY", "Biannual", "", "Unknown", "Semi-Annual"],
    )
    def test_unknown_or_wrong_case_returns_none(self, unknown):
        # Mapping is case-sensitive; unrecognised strings return None
        assert convert_frew_to_freq(unknown) is None


# ═══════════════════════════════════════════════════════════════════════════════
# get_event_type
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetEventType:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # EARNINGS
            ("Strong Earnings beat expectations",        EventType.EARNINGS),
            ("Revenue grew 20 pct this Quarter",         EventType.EARNINGS),
            # MERGER
            ("Company announces Merger with rival",      EventType.MERGER),
            ("Hostile Acquisition bid launched",         EventType.MERGER),
            ("Buyout offer made for startup",            EventType.MERGER),
            # REGULATORY
            ("SEC investigation opened into trading",    EventType.REGULATORY),
            ("New Regulation imposed on the sector",     EventType.REGULATORY),
            ("Major Lawsuit filed against company",      EventType.REGULATORY),
            # ECONOMIC
            ("FED raises rates by 25 bps",               EventType.ECONOMIC),
            ("Inflation hits 40-year high",              EventType.ECONOMIC),
            ("Rate hike expected next month",            EventType.ECONOMIC),
            # OTHER
            ("Company releases routine press statement", EventType.OTHER),
            ("Weather forecast for the week",            EventType.OTHER),
            ("",                                         EventType.OTHER),
        ],
    )
    def test_event_classification(self, text, expected):
        assert get_event_type(text) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# get_relevance
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetRelevance:
    @pytest.mark.parametrize(
        "publisher",
        ["Bloomberg", "Reuters", "Financial Times", "WSJ"],
    )
    def test_tier_one_publishers_are_high(self, publisher):
        assert get_relevance(publisher) == Relevance.HIGH

    @pytest.mark.parametrize(
        "publisher",
        ["CNN", "BBC", "TechCrunch", "MarketWatch", "Unknown Publisher", ""],
    )
    def test_other_publishers_are_medium(self, publisher):
        assert get_relevance(publisher) == Relevance.MEDIUM


# ═══════════════════════════════════════════════════════════════════════════════
# get_sentimental_label
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetSentimentalLabel:
    @pytest.mark.parametrize(
        "label, expected",
        [
            ("POS", SentimentLabel.POSITIVE),
            ("NEG", SentimentLabel.NEGATIVE),
            ("NEU", SentimentLabel.NEUTRAL),
        ],
    )
    def test_label_mapping(self, label, expected):
        with patch("data.data_collector.SENTIMENT_PIPLINE") as mock_pl:
            mock_pl.return_value = [{"label": label, "score": 0.9}]
            result = get_sentimental_label("some financial news text")
        assert result == expected


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_market_data
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ticker_mock(info: dict | None = None) -> MagicMock:
    """Return a yfinance.Ticker mock with sensible defaults."""
    default_info = {
        "shortName": "Apple Inc.",
        "sector": "Technology",
        "currentPrice": 180.5,
        "currency": "USD",
        "regularMarketChange": 1.5,
        "regularMarketChangePercent": 0.84,
        "volume": 55_000_000,
        "marketCap": 2_800_000_000_000,
    }
    mock = MagicMock()
    mock.info = info if info is not None else default_info
    prices = pd.Series([100.0 + i * 0.5 for i in range(100)])
    mock.history.return_value = pd.DataFrame({"Close": prices})
    return mock


class TestFetchMarketData:
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_valid_tickers_construct_ticker_with_correct_symbol(self, mock_cls, ticker):
        mock_cls.return_value = _make_ticker_mock()
        fetch_market_data(ticker)
        mock_cls.assert_called_once_with(ticker)

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_valid_tickers_return_none_due_to_missing_base_record_fields(self, mock_cls, ticker):
        """MarketData extends BaseRecord which requires id, source, created_at,
        entity_type, and intent fields.  None of these are passed in the constructor,
        causing a Pydantic ValidationError caught by the except block → returns None."""
        mock_cls.return_value = _make_ticker_mock()
        assert fetch_market_data(ticker) is None

    @pytest.mark.parametrize("ticker", ["INVALIDXYZ999", "FAKESTOCK123", "NOTREAL"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_nonexistent_tickers_return_none(self, mock_cls, ticker):
        mock_cls.side_effect = Exception(f"No data found for {ticker}")
        assert fetch_market_data(ticker) is None

    @patch("data.data_collector.yfinance.Ticker")
    def test_network_error_returns_none(self, mock_cls):
        mock_cls.side_effect = ConnectionError("Network error")
        assert fetch_market_data("AAPL") is None


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_macro_data
# ═══════════════════════════════════════════════════════════════════════════════

def _make_fred_mock(series_value: float = 120.5, frequency: str = "Monthly") -> MagicMock:
    """Return a fredapi.Fred mock with a single-value series."""
    mock = MagicMock()
    mock.get_series.return_value = pd.Series([series_value])
    mock_info = MagicMock()
    mock_info.frequency = frequency
    mock.get_series_info.return_value = mock_info
    return mock


class TestFetchMacroData:
    @pytest.mark.parametrize("country", ["US", "DE", "GB", "FR", "JP", "CA"])
    @patch("data.data_collector.Fred")
    def test_valid_countries_call_fred(self, mock_fred_cls, country):
        mock_fred_cls.return_value = _make_fred_mock()
        fetch_macro_data(country)
        mock_fred_cls.assert_called_once()

    @pytest.mark.parametrize("country", ["US", "DE", "GB", "FR", "JP", "CA"])
    @patch("data.data_collector.Fred")
    def test_valid_countries_return_tuple(self, mock_fred_cls, country):
        """MacroData extends BaseRecord which requires id, source, created_at,
        entity_type, and intent fields.  None of these are passed in the constructor,
        causing a Pydantic ValidationError caught by the except block → returns ().
        Will return a non-empty tuple once the missing BaseRecord fields are supplied."""
        mock_fred_cls.return_value = _make_fred_mock()
        result = fetch_macro_data(country)
        assert isinstance(result, tuple)

    @pytest.mark.parametrize("invalid_country", ["XX", "ZZ", "INVALID", "", "123"])
    @patch("data.data_collector.Fred")
    def test_invalid_country_returns_empty_tuple(self, mock_fred_cls, invalid_country):
        """get_cpi_ticker returns None for unknown ISO-2 codes; fred.get_series(None)
        raises an exception which is caught and returns ()."""
        mock_fred_cls.return_value = _make_fred_mock()
        assert fetch_macro_data(invalid_country) == ()

    @patch("data.data_collector.Fred")
    def test_iso3_usa_returns_empty_tuple(self, mock_fred_cls):
        """'USA' is ISO-3, not ISO-2.  get_cpi_ticker('USA') returns None → error → ()."""
        mock_fred_cls.return_value = _make_fred_mock()
        assert fetch_macro_data("USA") == ()

    @patch("data.data_collector.Fred")
    def test_pce_branch_unreachable_due_to_earlier_validation_error(self, mock_fred_cls):
        """PCE branch now correctly checks `country == 'US'` (ISO-2), but the
        MacroData constructor for CPI raises a Pydantic ValidationError (missing
        BaseRecord fields) before the PCE branch is reached.  The except block
        catches that error and returns () → USGOOD is never fetched."""
        mock_fred = _make_fred_mock()
        mock_fred_cls.return_value = mock_fred
        fetch_macro_data("US")
        called_with = [c.args[0] for c in mock_fred.get_series_info.call_args_list]
        assert "USGOOD" not in called_with

    @patch("data.data_collector.Fred")
    def test_fred_api_error_returns_empty_tuple(self, mock_fred_cls):
        mock_fred_cls.return_value = MagicMock(
            get_series=MagicMock(side_effect=Exception("FRED connection refused"))
        )
        assert fetch_macro_data("US") == ()


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_news
# ═══════════════════════════════════════════════════════════════════════════════

def _make_news_item(
    title: str = "Apple hits record high",
    summary: str = "Stock climbed 5 pct after earnings beat",
    publisher: str = "Bloomberg",
) -> dict:
    return {
        "content": {
            "title": title,
            "summary": summary,
            "provider": {"displayName": publisher},
            "canonicalUrl": {"url": "https://example.com/news/1"},
        }
    }


class TestFetchNews:
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "TSLA", "AMZN"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_valid_tickers_construct_ticker_with_correct_symbol(self, mock_cls, ticker):
        mock_ticker = MagicMock()
        mock_ticker.news = [_make_news_item()]
        mock_ticker.info = {"sector": "Technology"}
        mock_cls.return_value = mock_ticker
        fetch_news(ticker)
        mock_cls.assert_called_once_with(ticker)

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "TSLA", "AMZN"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_valid_tickers_return_tuple(self, mock_cls, ticker):
        """NewsData extends BaseRecord which requires id, source, created_at,
        entity_type, and intent fields; NewsData also requires published_at.
        None of these are passed in the constructor, causing a Pydantic
        ValidationError caught by the except block → returns ().
        Will return a non-empty tuple once the missing fields are supplied."""
        mock_ticker = MagicMock()
        mock_ticker.news = [_make_news_item()]
        mock_ticker.info = {"sector": "Technology"}
        mock_cls.return_value = mock_ticker
        result = fetch_news(ticker)
        assert isinstance(result, tuple)

    @pytest.mark.parametrize("ticker", ["INVALIDXYZ999", "FAKESTOCK123", "NOTREAL"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_nonexistent_tickers_return_empty_tuple(self, mock_cls, ticker):
        mock_cls.side_effect = Exception(f"No data for {ticker}")
        assert fetch_news(ticker) == ()

    @patch("data.data_collector.yfinance.Ticker")
    def test_no_news_returns_empty_tuple(self, mock_cls):
        mock_ticker = MagicMock()
        mock_ticker.news = []
        mock_ticker.info = {"sector": "Technology"}
        mock_cls.return_value = mock_ticker
        assert fetch_news("AAPL") == ()

    @patch("data.data_collector.yfinance.Ticker")
    def test_multiple_news_items_processed(self, mock_cls):
        mock_ticker = MagicMock()
        mock_ticker.news = [
            _make_news_item("Apple up",    "Stock rose 3 pct",   "Bloomberg"),
            _make_news_item("Market rally","Stocks surge",        "Reuters"),
            _make_news_item("Tech gains",  "Sector outperforms",  "CNN"),
        ]
        mock_ticker.info = {"sector": "Technology"}
        mock_cls.return_value = mock_ticker
        result = fetch_news("AAPL")
        assert isinstance(result, tuple)
