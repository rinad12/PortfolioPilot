import yfinance
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