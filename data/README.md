`data/` holds downloaded prices and generated outputs; it is git-ignored except `reference/`.

- `reference/members_at_rev.csv`, `reference/sp500_changes.csv`: S&P 500 constituents and the historical changes table,
  taken from Wikipedia's "List of S&P 500 companies" (CC BY-SA 4.0). Used to rebuild point-in-time membership.
- `prices_pit.parquet` (+ `.json` metadata): daily OHLCV for every ticker that was ever a member since 2011, built and
  extended with `src/extend_prices.py` (Yahoo Finance via yfinance). Not included: Yahoo data may not be redistributed.
