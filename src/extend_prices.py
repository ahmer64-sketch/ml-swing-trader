"""Append daily prices for the same point-in-time ticker list up to END (exclusive). Usage: python extend_prices.py END"""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',_os.path.join(HERE,'..','..','project4_ml_alpha','src'))
import sys, json, pandas as pd; sys.path.insert(0,ALPHA_SRC)
from alpha import data
END=sys.argv[1]
base=pd.read_parquet(f'{P}/prices_pit.parquet'); tick=sorted(base.ticker.unique()); start=(base.date.max()-pd.Timedelta(days=10)).strftime('%Y-%m-%d')
p=data.fetch_prices(tick,start,END,P); d,_=data.load_prices(p)
allp=pd.concat([base,d]).drop_duplicates(['date','ticker'],keep='last').sort_values(['ticker','date']); allp.to_parquet(f'{P}/prices_pit.parquet',index=False)
import os; os.remove(p); os.remove(str(p).replace('.parquet','.json'))
print('prices now to',allp.date.max().date(),'tickers',allp.ticker.nunique())
