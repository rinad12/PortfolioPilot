import yfinance
import pycountry
import fredapi

from .data_pydentic import MarketData, MacroData, NewsData



def fetch_market_data(ticker):
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

def get_cpi_ticker(country_iso2):
    '''Convert a country ISO2 code to the corresponding Fred CPI ticker symbol'''
    country = pycountry.countries.get(alpha_2=country_iso2.upper())
    if not country:
        return None
    
    iso3 = country.alpha_3
    return f"{iso3}CPIALLMINMEI"

def fetch_macro_data(country):
    '''Return macroeconomic data for a given country'''
    #If USA
    if country.upper() == 'USA':
        cpi = 
        return MacroData(
            category = MacroData.CategoryType.MACRO,
            name = 'United States',
            cpi = 300.84,  # Example CPI value
            gdp = 21433226.0,  # Example GDP value in millions USD
            unemployment_rate = 3.5,  # Example unemployment rate
            interest_rate = 0.25  # Example interest rate
        )