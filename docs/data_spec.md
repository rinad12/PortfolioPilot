# Base Record (Common Fields)

The base record defines both structural and semantic attributes common to all data categories.

```
{
  "id": "string",
  "source": "yahoo_finance | fred | news",
  "category": "market | macro | news",
  "created_at": "ISO-8601 datetime",
  "entity_type": "asset | indicator | event",
  "intent": "explanation | risk_context | policy_context"
}
```

# Market Data Payload (Yahoo Finance)

```
{
  "category": "market",
  "symbol": "AAPL",
  "name": null,
  "asset_type": "equity",
  "sector": null,
  "price": 192.34,
  "currency": "USD",
  "change_abs": null,
  "change_pct": null
  "volume": 53422100,
  "market_cap": null,
  "volatility_100d": 0.021
}
```

# Macro Data Payload (FRED)

```
{
  "category": "macro",
  "id": "CPIAUCSL",
  "name": "Consumer Price Index",
  "type": "inflation"
  "value": 312.23,
  "unit": "index",
  "frequency": "monthly",
  "country": "US",
  "policy_relevance": "high",
  "narrative_role": "inflation_context"
  }
```

# News Data Payload

```
{
  "category": "news",
  "headline": "Apple shares fall after earnings report",
  "summary": "Apple reported lower-than-expected revenue amid slowing iPhone sales.",
  "publisher": "Reuters",
  "published_at": "2025-03-10T14:30:00Z",
  "url": "https://example.com",
  "related_assets": ["AAPL"],
  "sectors": ["Technology"],
  "sentiment_label": "negative",
  "event_type": "earnings",
  "relevance": "high"
}
```
