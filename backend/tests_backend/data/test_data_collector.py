"""Tests for backend/src/data/data_collector.py

Covers:
  Pure helpers  : get_cpi_ticker, convert_frew_to_freq, get_event_type, get_relevance
  Sentiment     : get_sentimental_label  (mocked ML pipeline)
  Market data   : fetch_market_data      (mocked yfinance)
  Macro data    : fetch_macro_data       (mocked FRED API)
  News          : fetch_news             (mocked yfinance)

All fetch functions are tested with mocks that satisfy every Pydantic model
requirement.  Each test that receives a model object asserts that every
required (non-Optional) field is not None.
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
from data.data_pydentic import (
    EventType,
    Frequency,
    MacroData,
    MarketData,
    NewsData,
    Relevance,
    SentimentLabel,
)


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
    """Return a yfinance.Ticker mock with all fields required by MarketData.

    'country' is the full country name as yfinance returns it; the collector
    converts it to ISO-2 via pycountry.countries.search_fuzzy().
    """
    default_info = {
        "shortName": "Apple Inc.",
        "sector":    "Technology",
        "country":   "United States",   # yfinance full name → collector → "US"
        "currentPrice": 180.5,
        "currency":  "USD",
        "regularMarketChange": 1.5,
        "regularMarketChangePercent": 0.84,
        "volume":    55_000_000,
        "marketCap": 2_800_000_000_000,
    }
    mock = MagicMock()
    mock.info = info if info is not None else default_info
    prices = pd.Series([100.0 + i * 0.5 for i in range(100)])
    mock.history.return_value = pd.DataFrame({"Close": prices})
    return mock


def _assert_market_data_no_none(data: MarketData) -> None:
    """Assert every required MarketData field is not None."""
    assert data.category   is not None
    assert data.created_at is not None
    assert data.symbol     is not None
    assert data.country    is not None


class TestFetchMarketData:
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_valid_tickers_construct_ticker_with_correct_symbol(self, mock_cls, ticker):
        mock_cls.return_value = _make_ticker_mock()
        fetch_market_data(ticker)
        mock_cls.assert_called_once_with(ticker)

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_valid_tickers_return_market_data_instance(self, mock_cls, ticker):
        mock_cls.return_value = _make_ticker_mock()
        result = fetch_market_data(ticker)
        assert result is not None
        assert isinstance(result, MarketData)

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_valid_tickers_no_none_in_required_fields(self, mock_cls, ticker):
        mock_cls.return_value = _make_ticker_mock()
        result = fetch_market_data(ticker)
        assert result is not None
        _assert_market_data_no_none(result)

    @patch("data.data_collector.yfinance.Ticker")
    def test_country_resolved_to_iso2(self, mock_cls):
        """search_fuzzy('United States') must resolve to ISO-2 'US' to pass the
        MarketData.country validator which requires exactly 2 letters."""
        mock_cls.return_value = _make_ticker_mock()
        result = fetch_market_data("AAPL")
        assert result is not None
        assert result.country == "US"

    @patch("data.data_collector.yfinance.Ticker")
    def test_missing_country_in_info_returns_none(self, mock_cls):
        """country is a required field; when yfinance info has no 'country' key,
        country=None is passed to MarketData → ValidationError → returns None."""
        info_without_country = {
            "shortName":    "Apple Inc.",
            "sector":       "Technology",
            "currentPrice": 180.5,
            "currency":     "USD",
            "volume":       55_000_000,
        }
        mock_cls.return_value = _make_ticker_mock(info=info_without_country)
        assert fetch_market_data("AAPL") is None

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


def _assert_macro_data_no_none(data: MacroData) -> None:
    """Assert every required MacroData field is not None."""
    assert data.category         is not None
    assert data.created_at       is not None
    assert data.indicator_id     is not None
    assert data.indicator_name   is not None
    assert data.indicator_type   is not None
    assert data.value            is not None
    assert data.unit             is not None
    assert data.frequency        is not None
    assert data.country          is not None
    assert data.policy_relevance is not None
    assert data.narrative_role   is not None


class TestFetchMacroData:
    @pytest.mark.parametrize("country", ["US", "DE", "GB", "FR", "JP", "CA"])
    @patch("data.data_collector.Fred")
    def test_valid_countries_call_fred(self, mock_fred_cls, country):
        mock_fred_cls.return_value = _make_fred_mock()
        fetch_macro_data(country)
        mock_fred_cls.assert_called_once()

    @pytest.mark.parametrize("country", ["DE", "GB", "FR", "JP", "CA"])
    @patch("data.data_collector.Fred")
    def test_non_us_countries_return_single_element_tuple(self, mock_fred_cls, country):
        mock_fred_cls.return_value = _make_fred_mock()
        result = fetch_macro_data(country)
        assert isinstance(result, tuple)
        assert len(result) == 1
        _assert_macro_data_no_none(result[0])

    @patch("data.data_collector.Fred")
    def test_us_returns_two_element_tuple(self, mock_fred_cls):
        """US fetches both CPI and PCE (USGOOD) data → 2-element tuple."""
        mock_fred_cls.return_value = _make_fred_mock()
        result = fetch_macro_data("US")
        assert isinstance(result, tuple)
        assert len(result) == 2
        for macro in result:
            _assert_macro_data_no_none(macro)

    @patch("data.data_collector.Fred")
    def test_us_fetches_pce_series(self, mock_fred_cls):
        """For country='US', USGOOD (PCE) must be requested from FRED."""
        mock_fred = _make_fred_mock()
        mock_fred_cls.return_value = mock_fred
        fetch_macro_data("US")
        called_series = [c.args[0] for c in mock_fred.get_series.call_args_list]
        assert "USGOOD" in called_series

    @patch("data.data_collector.Fred")
    def test_non_us_does_not_fetch_pce(self, mock_fred_cls):
        """For non-US countries, USGOOD must NOT be requested."""
        mock_fred = _make_fred_mock()
        mock_fred_cls.return_value = mock_fred
        fetch_macro_data("DE")
        called_series = [c.args[0] for c in mock_fred.get_series.call_args_list]
        assert "USGOOD" not in called_series

    @pytest.mark.parametrize("invalid_country", ["XX", "ZZ", "INVALID", "", "123"])
    @patch("data.data_collector.Fred")
    def test_invalid_country_returns_empty_tuple(self, mock_fred_cls, invalid_country):
        """get_cpi_ticker returns None for unknown ISO-2 codes;
        fred.get_series(None) raises an exception → returns ()."""
        mock_fred_cls.return_value = _make_fred_mock()
        assert fetch_macro_data(invalid_country) == ()

    @patch("data.data_collector.Fred")
    def test_iso3_usa_returns_empty_tuple(self, mock_fred_cls):
        """'USA' is ISO-3, not ISO-2.  get_cpi_ticker('USA') returns None → ()."""
        mock_fred_cls.return_value = _make_fred_mock()
        assert fetch_macro_data("USA") == ()

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
    title:     str = "Apple hits record high",
    summary:   str = "Stock climbed 5 pct after Earnings beat",
    publisher: str = "Bloomberg",
    pub_date:  str = "2024-01-15T10:30:00Z",
) -> dict:
    """Return a yfinance-style news item dict with all fields fetch_news needs."""
    return {
        "content": {
            "title":        title,
            "summary":      summary,
            "provider":     {"displayName": publisher},
            "canonicalUrl": {"url": "https://example.com/news/1"},
            "pubDate":      pub_date,
        }
    }


def _assert_news_data_no_none(data: NewsData) -> None:
    """Assert every required NewsData field is not None."""
    assert data.category        is not None
    assert data.created_at      is not None
    assert data.headline        is not None
    assert data.summary         is not None
    assert data.publisher       is not None
    assert data.published_at    is not None
    assert data.url             is not None
    assert data.sentiment_label is not None
    assert data.event_type      is not None
    assert data.relevance       is not None


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
    def test_valid_tickers_return_news_data_tuple(self, mock_cls, ticker):
        mock_ticker = MagicMock()
        mock_ticker.news = [_make_news_item()]
        mock_ticker.info = {"sector": "Technology"}
        mock_cls.return_value = mock_ticker
        result = fetch_news(ticker)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], NewsData)

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "TSLA", "AMZN"])
    @patch("data.data_collector.yfinance.Ticker")
    def test_valid_tickers_no_none_in_required_fields(self, mock_cls, ticker):
        mock_ticker = MagicMock()
        mock_ticker.news = [_make_news_item()]
        mock_ticker.info = {"sector": "Technology"}
        mock_cls.return_value = mock_ticker
        result = fetch_news(ticker)
        assert len(result) == 1
        _assert_news_data_no_none(result[0])

    @patch("data.data_collector.yfinance.Ticker")
    def test_missing_sector_is_none(self, mock_cls):
        """When info has no 'sector' key, sector is None (Optional[str] field)
        → NewsData is still created successfully."""
        mock_ticker = MagicMock()
        mock_ticker.news = [_make_news_item()]
        mock_ticker.info = {}   # no 'sector' key
        mock_cls.return_value = mock_ticker
        result = fetch_news("AAPL")
        assert len(result) == 1
        assert result[0].sector is None

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
    def test_multiple_news_items_all_returned(self, mock_cls):
        mock_ticker = MagicMock()
        mock_ticker.news = [
            _make_news_item("Apple up",     "Stock rose 3 pct",   "Bloomberg"),
            _make_news_item("Market rally", "Stocks surge",        "Reuters"),
            _make_news_item("Tech gains",   "Sector outperforms",  "CNN"),
        ]
        mock_ticker.info = {"sector": "Technology"}
        mock_cls.return_value = mock_ticker
        result = fetch_news("AAPL")
        assert isinstance(result, tuple)
        assert len(result) == 3
        for item in result:
            _assert_news_data_no_none(item)
