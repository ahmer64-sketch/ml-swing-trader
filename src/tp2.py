"""+2% take-profit between weekly trades. For every holding (stocks, EXTRA ETFs and SPY) the reference is that
week's trade-day open; if any day's high (trade day .. day before the next trade) reaches ref*(1+TP), it is sold
at exactly ref*(1+TP) that day (limit order). Proceeds go to cash until the next weekly trade; the realised gain
is added to 'pend' (taxed at the next weekly step, like ETF sales). Run right after the week's pit_ledger call.
Usage: python tp2.py LEDGER TRADE_DATE NEXT_TRADE_DATE [TP=0.02]"""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',_os.path.join(HERE,'..','..','project4_ml_alpha','src'))
import sys, json, pandas as pd
LG,D0,D1=sys.argv[1],pd.Timestamp(sys.argv[2]),pd.Timestamp(sys.argv[3]); TP=float(sys.argv[4]) if len(sys.argv)>4 else 0.02
SF=f'{P}/ledger_state_{LG}.json'; TF=f'{P}/ledger_trades_{LG}.csv'
st=json.load(open(SF)); p=pd.read_parquet(f'{P}/prices_pit.parquet'); p['date']=pd.to_datetime(p['date'])
WK=st['history'][-1]['week']; rows=[]; fees=0.0
def path(t):
    r=p[(p.ticker==t)&(p.date>=D0)&(p.date<D1)].set_index('date').sort_index(); return r
TR=st.get('tpref',{})
def sell(t,val_fn,basis):
    r=path(t); ref=r.open.iloc[0]
    if t in TR: ref=TR[t][0]; r=r[r.index>pd.Timestamp(TR[t][1])]   # mid-week buy: its price, from the next day
    hit=r[r.high>=ref*(1+TP)]
    if not len(hit): return None
    d=hit.index[0]; px=ref*(1+TP); val=val_fn(px*r.at[d,'adj_close']/r.at[d,'close'],r)
    fe=val*(2.0 if t=='SPY' else 3.0)/1e4; return d,px,val,fe,val-fe-basis
for s,(sh,b) in list(st['pos'].items()):
    o=sell(s,lambda apx,r: sh*apx,b)
    if o:
        d,px,val,fe,g=o; st['pos'].pop(s); st['cash']+=val-fe; st['pend']+=g; fees+=fe
        rows.append(dict(week=WK,stock=s,action=f'TP +{TP:.0%} {d.date()}',sigma=None,shares=None,open=px,value=val,fee=fe,gain=g))
if st['etf'][0]>0:
    r=path('SPY'); a0=r.open.iloc[0]*r.adj_close.iloc[0]/r.close.iloc[0]
    o=sell('SPY',lambda apx,r: st['etf'][0]*apx/a0,st['etf'][1])
    if o:
        d,px,val,fe,g=o; st['cash']+=val-fe; st['pend']+=g; fees+=fe; st['etf']=[0.0,0.0]
        rows.append(dict(week=WK,stock='ETF (SPY)',action=f'TP +{TP:.0%} {d.date()}',sigma=None,shares=None,open=px,value=val,fee=fe,gain=g))
st['tpref']={}
st['history'][-1]['tp']=[dict(stock=x['stock'],action=x['action'],value=round(x['value'],2),gain=round(x['gain'],2)) for x in rows]
json.dump(st,open(SF,'w'),indent=1,default=float)
if rows: pd.DataFrame(rows).to_csv(TF,mode='a',header=False,index=False)
for x in rows: print(x['stock'],x['action'],round(x['open'],2),round(x['value'],2),round(x['gain'],2))
print('TP sales',len(rows),'cash now',round(st['cash'],2),'pend',round(st['pend'],2))
