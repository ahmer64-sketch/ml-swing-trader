"""Manual mid-week buy of one ticker with all spare cash at a given intraday price (from yfinance 5-min bars).
The +2% take-profit reference for it becomes this buy price, checked from the next trading day (st['tpref']).
Usage: python buyat.py LEDGER TICKER DATE PRICE [AMOUNT | SALE (proceeds of this ticker's sellat sale), default all cash]"""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',_os.path.join(HERE,'..','..','project4_ml_alpha','src'))
import sys, json, pandas as pd
LG,T,D,PX=sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4])
SF=f'{P}/ledger_state_{LG}.json'; TF=f'{P}/ledger_trades_{LG}.csv'
st=json.load(open(SF)); WK=st['history'][-1]['week']
amt=min(st['sales'].pop(T) if sys.argv[5]=='SALE' else float(sys.argv[5]),st['cash']) if len(sys.argv)>5 else st['cash']; fe=amt*3.0/1e4; sh=(amt-fe)/PX
if T=='SPY':   # back into the ETF slot, stored in dollars of the last weekly trade-day open (adjusted)
    p=pd.read_parquet(f'{P}/prices_pit.parquet'); p['date']=pd.to_datetime(p['date']); s=p[p.ticker=='SPY'].sort_values('date')
    q=s[s.date==st['history'][-1]['trade_date']].iloc[0]; a0=q.open*q.adj_close/q.close; k=s.adj_close.iloc[-1]/s.close.iloc[-1]
    fe=amt*2.0/1e4; sh=(amt-fe)/PX; st['etf']=[st['etf'][0]+(amt-fe)*a0/(PX*k),st['etf'][1]+amt]
else:
    o=st['pos'].get(T,[0.0,0.0]); st['pos'][T]=[o[0]+sh,o[1]+amt]
st['cash']-=amt
st.setdefault('tpref',{})[T]=[PX,D]
st['history'][-1].setdefault('manual',[]).append(dict(stock=T,action=f'BUY {D} @ {PX}',value=round(amt,2)))
json.dump(st,open(SF,'w'),indent=1,default=float)
pd.DataFrame([dict(week=WK,stock=T,action=f'BUY {D} intraday',sigma=None,shares=sh,open=PX,value=amt-fe,fee=fe,gain=None)]).to_csv(TF,mode='a',header=False,index=False)
print(T,round(sh,4),PX,round(amt,2),round(fe,2))
