"""Build data/prices_pit.parquet from scratch: every ticker that was an S&P 500 member at any time since START,
plus the index ETFs, daily OHLCV from Yahoo Finance. Usage: python build_prices.py START END [LIMIT]"""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',next((p for p in (_os.path.join(HERE,'..','..',n,'src') for n in ('ml-alpha-sp500','project4_ml_alpha')) if _os.path.isdir(p)),''))
import sys, json, pandas as pd; sys.path.insert(0,ALPHA_SRC); sys.path.insert(0,HERE)
from alpha import data
from members import ever
START,END=sys.argv[1:3]; LIMIT=int(sys.argv[3]) if len(sys.argv)>3 else None
ETF=['SPY','QQQ','TQQQ','SOXX','SOXL','DIA','IWM']
tick=sorted(ever(START,END)|set(ETF))
if LIMIT: tick=tick[:LIMIT]+[e for e in ETF if e not in tick[:LIMIT]]
p=data.fetch_prices(tick,START,END,P); d,meta=data.load_prices(p)
d.to_parquet(f'{P}/prices_pit.parquet',index=False); json.dump(meta,open(f'{P}/prices_pit.json','w'),default=str)
_os.remove(p); _os.remove(str(p).replace('.parquet','.json'))
print('prices',d.date.min().date(),'to',d.date.max().date(),'tickers',d.ticker.nunique())
