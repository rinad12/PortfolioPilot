import yfinance
import pycountry
from typing import Tuple
from fredapi import Fred
import os

from .data_pydentic import MarketData, MacroData, NewsData, MacroType, Relevance, NarrativeRole, Frequency

FRED_API_KEY = os.environ["FRED_API_KEY"]

def fetch_market_data(ticker: str) -> MarketData:
    """Fetch market data for a given ticker symbol using yfinance"""
    try:
        stock = yfinance.Ticker(ticker)
        info = stock.info

        # Calculate volatility over the past 100 days
        data = stock.history(period="100d")
        prices = data['Close']
        log_returns = np.log(prices / prices.shift(1))
        log_returns = log_returns.dropna()
        volatility = log_returns.std()
        
        return MarketData(
            category = MarketData.CategoryType.MARKET,
            symbol=ticker,
            name=info.get('shortName'),
            sector=info.get('sector'),
            price=info.get('currentPrice'),
            currency=info.get('currency'),
            change_abs=info.get('regularMarketChange'),
            change_pct=info.get('regularMarketChangePercent'),
            volume=info.get('volume'),
            market_cap=info.get('marketCap'),
            volatility_100d=volatility
        )
    except Exception as e:
        print(f"Error fetching market data for {ticker}: {e}")
        return None

def get_cpi_ticker(country_iso2: str) -> str:
    '''Convert a country ISO2 code to the corresponding Fred CPI ticker symbol'''
    country = pycountry.countries.get(alpha_2=country_iso2.upper())
    if not country:
        return None
    
    iso3 = country.alpha_3
    return f"{iso3}CPIALLMINMEI"

def convert_frew_to_freq(fred_freq: str) -> Frequency:
    '''Convert FRED frequency to our internal Frequency enum'''
    mapping = {
        'Daily': Frequency.DAILY,
        'Weekly': Frequency.WEEKLY,
        'Monthly': Frequency.MONTHLY,
        'Quarterly': Frequency.QUARTERLY,
        'Annual': Frequency.ANNUAL
    }
    return mapping.get(fred_freq, None)

def fetch_macro_data(country: str) -> Tuple[MacroData,...]:
    '''Return macroeconomic data for a given country'''
    fred = Fred(FRED_API_KEY)
    # Get cpi of country
    cpi_ticker = get_cpi_ticker(country)
    cpi = fred.get_series(cpi_ticker)
    cpi = cpi.iloc[-1]
    freq_cpi = fred.get_series_info('CPIAUCSL').frequency

    cpi_data = MacroData(
        idicator_id = cpi_ticker,
        indicator_name = f"CPI for {country}",
        indicator_type = MacroType.INFLATION,
        value = cpi,
        unit = "Index",
        frequency = convert_frew_to_freq(freq_cpi),
        country = country,
        policy_relevance = Relevance.HIGH,
        NarrativeRole = NarrativeRole.INFLATION_CONTEXT
    )
    # If country is USA, also get PCE data
    if country.upper() == 'USA':
        pce = fred.get_series('USGOOD')
        pce = pce.iloc[-1]
        freq_pce = fred.get_series_info('USGOOD').frequency
        pce_data = MacroData(
            idicator_id = 'USGOOD',
            indicator_name = "PCE for USA",
            indicator_type = MacroType.INFLATION,
            value = pce,
            unit = "Index",
            frequency = convert_frew_to_freq(freq_pce),
            country = 'USA',
            policy_relevance = Relevance.HIGH,
            NarrativeRole = NarrativeRole.MONETARY_POLICY
        )
        return cpi_data, pce_data
    else:
        return cpi_data, 