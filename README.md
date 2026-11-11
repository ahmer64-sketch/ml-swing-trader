# ML Swing Trader (S&P 500)

A weekly swing-trading system for S&P 500 stocks and index ETFs. Three machine-learning models score every stock each week. A GARCH volatility forecast filters the risk. A ledger tracks every trade with real fees and taxes.

The hard part was not the models. It was making sure no number used information that did not exist on the day it was computed.

## What it does each week

1. **Rebuild the universe for that date.** Only stocks that were in the S&P 500 on the signal date are eligible. Membership is rolled back from a 2026 snapshot through the historical changes table, so delisted and removed names are included and later additions are not.
2. **Score every member with three models.** Each predicts the stock's 5-day return relative to the average member.
   - Gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`). Output is a cross-sectional z-score, σ.
   - MLP neural network trained with Adam.
   - MLP neural network trained with L-BFGS.
   - All three use the same 12 rank-normalised features built from price and volume: momentum, distance from the 52-week high, total and idiosyncratic volatility, volume trend, Amihud illiquidity and a size proxy. ETFs are scored by ranking their features against that day's members.
3. **Filter.** Keep names where all three models predict a gain. The strict rule buys only σ ≥ 3.
4. **Measure risk.** A GARCH(1,1) model, fitted by quasi-maximum likelihood on the last 756 daily returns, forecasts next week's volatility. Names above 10% weekly volatility, or running well above their long-run level, are flagged.
5. **Rate and size.** Each name gets a 1 to 10 rating from signal strength, expected return, return per unit of risk and volatility stability. Positions are weighted by σ, by ensemble return, or by return per risk.
6. **Trade and track.** Buy at the next open. Sell any holding the day it touches +2%. The ledger charges 3 bp plus a liquidity charge of 30/√(daily dollar volume in $m) bp per stock trade, 2 bp for SPY, 24% tax on net realised gains with losses carried forward, and 4.5% interest on idle cash. A replay script rebuilds the whole account from scratch.

## Leakage controls

- Point-in-time S&P membership for training and for scoring.
- Features ranked and labels de-meaned across that day's members only.
- Purged and embargoed training window that ends at a fixed cut-off. Hyperparameters chosen on the last 12 months before the cut-off.
- Signals use the prior close. Trades use the next open.

## What the research found

The original backtest (my [ml-alpha-sp500](https://github.com/ahmer64-sketch/ml-alpha-sp500) repo) looked strong. It was not. The edge came from survivorship bias: it trained on today's S&P list, which quietly selects stocks that went on to do well. Once membership was made point-in-time, the clean hold-out showed no reliable edge after costs.

This repo is the rebuilt, point-in-time version. Out-of-sample information coefficients are small. The GBM's validation IC for the 2026 model is 0.018. That is typical for weekly equity signals and far too thin to trust without a long live record. I ran a live paper account from August 2026 to test it, but I made discretionary changes along the way, including some with hindsight, so I do not report its return as evidence.

The useful output of this project is the pipeline and the discipline around it. Not a strategy claim.

## Run it

Requires the `alpha` package from [ml-alpha-sp500](https://github.com/ahmer64-sketch/ml-alpha-sp500) (data loading and feature code). Clone both repos side by side, or set `ALPHA_SRC` to its `src/` folder.

```bash
git clone https://github.com/ahmer64-sketch/ml-alpha-sp500
git clone https://github.com/ahmer64-sketch/ml-swing-trader
cd ml-swing-trader
```

```bash
pip install -r requirements.txt
cd src
python build_prices.py 2011-01-01 2026-11-11          # download prices for every ever-member + ETFs (~600 tickers)
cd ..
zsh scripts/score_week.sh 2026-11-10 2026-07-31      # signal date, training cut-off -> data/scores_<date>.csv
python scripts/shortlist.py 2026-11-10              # all-three-up + GARCH filter + 1-10 rating
```

Ledger tools in `src/`: `pit_ledger.py` runs a weekly rebalance, `tp2.py` applies the +2% take-profit, `buyat.py` and `sellat.py` record mid-week trades. Set `SWING_DATA` to keep data outside the repo.

## Layout

```
src/
  members.py        point-in-time S&P 500 membership
  pit_week.py       GBM training and scoring (members only)
  pit_nn.py         MLP Adam / L-BFGS models
  pit_etfscore.py   scores ETFs against members
  pit_three.py      three-model comparison tables
  garch_week.py     GARCH(1,1) weekly volatility
  pit_lambda.py     position weights
  pit_ledger.py     weekly ledger: fees, tax, carry-forward, cash interest
  tp2.py            +2% take-profit
  buyat.py sellat.py  mid-week trades
  build_prices.py extend_prices.py   data download
scripts/
  score_week.sh     one week, all models, stocks + ETFs
  shortlist.py      filter, GARCH, rating
data/reference/     S&P 500 constituents and changes (Wikipedia, CC BY-SA 4.0)
```

## Known issues

- Yahoo Finance data is not clean. SOXX prices sometimes come back divided by 3, intraday bars contain bad prints, and same-day daily closes can be missing. Check new data before scoring.
- The ensemble's GBM leg is converted from σ to a percentage with a fixed scale (0.49% per week for the average ≥3σ pick). It is a heuristic, not a calibrated forecast.
- Stocks pinned by a pending takeover produce meaningless GARCH fits and inflated model scores. The shortlist drops degenerate fits.

Research code. Not investment advice.
