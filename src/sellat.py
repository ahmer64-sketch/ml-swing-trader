"""Manual mid-week sale of one holding at a given day's close (4 pm). Proceeds to cash until the next weekly trade;
the realised gain goes to 'pend' (taxed at the next weekly step), as in tp2.py.
Usage: python sellat.py LEDGER TICKER DATE [PRICE, default that day's close; a limit fill such as a +2% sale]"""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',next((p for p in (_os.path.join(HERE,'..','..',n,'src') for n in ('ml-alpha-sp500','project4_ml_alpha')) if _os.path.isdir(p)),''))
import sys, json, pandas as pd
LG,T,D=sys.argv[1],sys.argv[2],pd.Timestamp(sys.argv[3])
SF=f'{P}/ledger_state_{LG}.json'; TF=f'{P}/ledger_trades_{LG}.csv'
st=json.load(open(SF)); p=pd.read_parquet(f'{P}/prices_pit.parquet'); p['date']=pd.to_datetime(p['date'])
x=p[(p.ticker==T)&(p.date<=D)].sort_values('date'); r=x.iloc[-1].copy()   # a day past the price file: intraday PX, latest adj factor
PX=float(sys.argv[4]) if len(sys.argv)>4 else r.close; r['adj_close']=PX*r.adj_close/r.close; r['close']=PX; WK=st['history'][-1]['week']
if T=='SPY':   # the ETF slot: value carried as dollars since the last weekly trade-day open (adjusted)
    q=p[(p.ticker=='SPY')&(p.date==st['history'][-1]['trade_date'])].iloc[0]; a0=q.open*q.adj_close/q.close
    val=st['etf'][0]*r.adj_close/a0; b=st['etf'][1]; sh=val/r.adj_close; st['etf']=[0.0,0.0]; fe=val*2.0/1e4
else:
    sh,b=st['pos'].pop(T); val=sh*r.adj_close; fe=val*3.0/1e4
g=val-fe-b
st['cash']+=val-fe; st.setdefault('sales',{})[T]=val-fe; st['pend']+=g
st['history'][-1].setdefault('manual',[]).append(dict(stock=T,action=(f'SELL {D.date()} @ {PX:.2f}' if len(sys.argv)>4 else f'SELL close {D.date()}'),value=round(val,2),gain=round(g,2)))
json.dump(st,open(SF,'w'),indent=1,default=float)
pd.DataFrame([dict(week=WK,stock=T,action=(f'SELL {D.date()} @ {PX:.2f}' if len(sys.argv)>4 else f'SELL close {D.date()}'),sigma=None,shares=sh,open=r.close,value=val,fee=fe,gain=g)]).to_csv(TF,mode='a',header=False,index=False)
print(T,round(sh,4),round(r.close,2),round(val,2),round(fe,2),round(g,2),'cash',round(st['cash'],2),'pend',round(st['pend'],2))
