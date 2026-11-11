"""Point-in-time F account, one week per call. Rules = current F: z>=3 picks score-weighted, 50% cap per stock,
keep-and-swap, trade at the open after the signal close, fees 3+30/sqrt(ADV$m) bps (ETF 2bp), spare cash in
the ETF (SPY), 24% weekly tax on net realised gains with loss carry-forward.
Usage: python pit_ledger.py WEEK SIGNAL_DATE TRADE_DATE [LAMBDA] [TOPN]   (state in ledger_state.json)
With LAMBDA: target weights come from pit_lambda.py (lambda_<signal>.csv, row 'λ=<LAMBDA>'); new picks are bought at
their target, held picks are kept (no fee/tax) unless above target, then trimmed to it; the rest goes to the ETF."""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',_os.path.join(HERE,'..','..','project4_ml_alpha','src'))
import sys, json, os, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
sys.path.insert(0,ALPHA_SRC)
from alpha import data, features as F, evaluate as E
WK=int(sys.argv[1]); T=pd.Timestamp(sys.argv[2]); D=pd.Timestamp(sys.argv[3]); LAM=sys.argv[4] if len(sys.argv)>4 else None; CASH=(LAM=='cash'); LAM=None if CASH else LAM; TOPN=int(sys.argv[5]) if len(sys.argv)>5 else None; CAP=0.5; TAX=0.24; EF=2.0; ETF='SPY'
df,_=data.load_prices(f'{P}/prices_pit.parquet')
O=data.wide(df,'open'); C=data.wide(df,'close'); AC=data.wide(df,'adj_close'); AO=O*(AC/C)
pnl=F.panel(data.clean(df[df.date<=T])[0]); cost=E.cost_bps(pnl['dvol'].rolling(63,min_periods=20).median()/1e6).loc[T]
LG=os.environ.get('LEDGER',''); SF=f'{P}/ledger_state{("_"+LG) if LG else ""}.json'; TF=f'{P}/ledger_trades{("_"+LG) if LG else ""}.csv'
st=json.load(open(SF)) if os.path.exists(SF) and (WK>1 or os.environ.get('CARRY')=='1') else dict(cash=float(os.environ.get('START','100000')),pos={},etf=[0.0,0.0],carry=0.0,pend=0.0,history=[])
assert len(st['history'])==WK-1, f"ledger is at week {len(st['history'])}"
z=pd.read_csv(f'{P}/pit_scores_{T.date()}.csv',index_col=0).sigma
ax=lambda s: AO.at[D,s]; rw=lambda s: O.at[D,s]
pos={s:list(v) for s,v in st['pos'].items()}; etf=st['etf']; cash=st['cash']; carry=st['carry']
# revalue ETF from last trade date
if st['history'] and cash>0: cash*=1+0.045*(D-pd.Timestamp(st['history'][-1]['trade_date'])).days/365
if st['history'] and etf[0]>0: etf[0]*=ax(ETF)/AO.at[pd.Timestamp(st['history'][-1]['trade_date']),ETF]
vb=cash+etf[0]+sum(sh*ax(s) for s,(sh,b) in pos.items())
tg=z.iloc[0:0] if CASH else (z.sort_values(ascending=False).head(TOPN) if TOPN else z[z>=3.0]); tg=pd.concat([tg,pd.Series(np.nan,index=[x for x in os.environ.get('EXTRA','').split(',') if x and x not in tg.index])]) if os.environ.get('EXTRA') else tg   # EXTRA: non-member ETFs (e.g. QQQ) held as positions
tg=tg[[s in AO.columns and ax(s)==ax(s) for s in tg.index]]
tr=[]; fees=0.0; real=st['pend']; pend=0.0
for s in list(pos):
    if s not in tg.index:
        sh,b=pos.pop(s); pr=sh*ax(s); fe=pr*cost.get(s,50)/1e4; fees+=fe; cash+=pr-fe; real+=pr-fe-b
        tr.append(dict(stock=s,action='SELL',sigma=round(z.get(s,np.nan),2),shares=sh*ax(s)/rw(s),open=rw(s),value=pr,fee=fe,gain=pr-fe-b))
total=cash+etf[0]+sum(sh*ax(s) for s,(sh,b) in pos.items())
if LAM:
    lt=pd.read_csv(f'{P}/lambda_{T.date()}'+(f'_top{TOPN}' if TOPN else '')+(('_'+os.environ['MODEL']) if os.environ.get('MODEL') else ('_shrink' if os.environ.get('SHRINK')=='1' else ''))+'.csv').set_index('weights').loc[f'λ={LAM}']/100
    TW={k:float(lt.get(k,0.0)) for k in tg.index}; tg=tg[[TW[k]>0 for k in tg.index]]   # zero-weight names are not held
else: TW={k:CAP for k in tg.index}
for s,(sh,b) in list(pos.items()):
    v=sh*ax(s)
    if v>TW[s]*total+1:
        cut=v-TW[s]*total; fr=cut/v; fe=cut*cost.get(s,50)/1e4; fees+=fe; cash+=cut-fe; real+=cut-fe-b*fr; pos[s]=[sh*(1-fr),b*(1-fr)]
        tr.append(dict(stock=s,action='TRIM to target',sigma=round(z.get(s,np.nan),2),shares=sh*fr*ax(s)/rw(s),open=rw(s),value=cut,fee=fe,gain=cut-fe-b*fr))
    else: tr.append(dict(stock=s,action='KEEP',sigma=round(z.get(s,np.nan),2),shares=sh*ax(s)/rw(s),open=rw(s),value=v,fee=0.0,gain=np.nan))
pos={s:v for s,v in pos.items() if v[0]>1e-9}             # fully trimmed (target 0) = sold
new=tg[[s for s in tg.index if s not in pos]]
need_top=bool(LAM) and any(pos[s][0]*ax(s)<TW[s]*total-1 for s in pos if s in TW)
if (len(new) or need_top or CASH or os.environ.get('SELLETF')=='1') and etf[0]>0 and os.environ.get('KEEPETF')!='1':   # KEEPETF=1: leave SPY untouched, buy from freed cash only
    fe=etf[0]*EF/1e4; fees+=fe; gg=etf[0]-fe-etf[1]
    if gg>0: pend+=gg
    else: carry+=gg
    tr.append(dict(stock=f'ETF ({ETF})',action='SELL',sigma=np.nan,shares=np.nan,open=rw(ETF),value=etf[0],fee=fe,gain=gg)); cash+=etf[0]-fe; etf=[0.0,0.0]
net=real+carry; tax=TAX*net if net>0 else 0.0; carry=0.0 if net>0 else net
KE=os.environ.get('KEEPETF')=='1'
budget=max(0.0,cash-tax+(etf[0] if KE else 0.0))   # KEEPETF: buys may draw on SPY, selling only what is needed
if LAM and budget>0:                                   # λ mode: top up held picks to target, then buy new ones
    left=budget
    for s in tg.index:
        cur=pos[s][0]*ax(s) if s in pos else 0.0; amt=min(max(0.0,TW[s]*total-cur),left)
        if amt<=1: continue
        fe=amt*cost.get(s,50)/1e4; fees+=fe; left-=amt; cash-=amt
        if s in pos: pos[s]=[pos[s][0]+(amt-fe)/ax(s),pos[s][1]+amt]; act='TOP-UP'
        else: pos[s]=[(amt-fe)/ax(s),amt]; act='BUY'
        tr.append(dict(stock=s,action=act,sigma=round(z.get(s,np.nan),2),shares=(amt-fe)/rw(s),open=rw(s),value=amt,fee=fe,gain=np.nan))
    new=new.iloc[0:0]
if len(new) and budget>0:
    wn=new/new.sum(); left=budget
    for s in new.index:
        amt=min(TW[s]*total,left) if LAM else min(budget*wn[s],CAP*total)
        if amt<=0: continue
        left-=amt
        fe=amt*cost.get(s,50)/1e4; fees+=fe; sh=(amt-fe)/ax(s); pos[s]=[sh,amt]; cash-=amt
        tr.append(dict(stock=s,action='BUY',sigma=round(z.get(s,np.nan),2),shares=(amt-fe)/rw(s),open=rw(s),value=amt,fee=fe,gain=np.nan))
cash-=tax
if cash<0 and etf[0]>0:
    a=min(-cash/(1-EF/1e4),etf[0]); fr=a/etf[0]; fe=a*EF/1e4; fees+=fe; gg=a-fe-etf[1]*fr
    if KE:
        if gg>0: pend+=gg
        else: carry+=gg
        tr.append(dict(stock=f'ETF ({ETF})',action='SELL part (fund buys)',sigma=np.nan,shares=a/ax(ETF)*ax(ETF)/rw(ETF),open=rw(ETF),value=a,fee=fe,gain=gg))
    etf=[etf[0]-a,etf[1]*(1-fr)]; cash+=a-fe
if cash>1 and not CASH and os.environ.get('KEEPCASH')!='1':
    fe=cash*EF/1e4; fees+=fe; tr.append(dict(stock=f'ETF ({ETF})',action='BUY (spare cash)',sigma=np.nan,shares=(cash-fe)/rw(ETF),open=rw(ETF),value=cash,fee=fe,gain=np.nan))
    etf=[etf[0]+cash-fe,etf[1]+cash]; cash=0.0
va=cash+etf[0]+sum(sh*ax(s) for s,(sh,b) in pos.items())
H=dict(weights=('ALL CASH' if CASH else ('λ='+LAM) if LAM else '∝σ, 50% cap')+(f', top {TOPN}' if TOPN else ', σ≥3')+((', μ model '+os.environ['MODEL']+(' + rest cash' if os.environ.get('KEEPCASH')=='1' else '')) if os.environ.get('MODEL') else (', shrunk μ̂' if os.environ.get('SHRINK')=='1' else '')),week=WK,signal=str(T.date()),trade_date=str(D.date()),value_before=vb,realised=real,tax=tax,carry_forward=carry,fees=fees,value_after=va,
       holdings={s:round(sh*ax(s)/va*100,1) for s,(sh,b) in pos.items()}|({ETF:round(etf[0]/va*100,1)} if etf[0]>0 else {})|({'CASH':round(cash/va*100,1)} if cash>1 else {}))
st.update(cash=cash,pos=pos,etf=etf,carry=carry,pend=pend); st['history'].append(H); json.dump(st,open(SF,'w'),indent=1,default=float)
TR=pd.DataFrame(tr); TR.insert(0,'week',WK); TR.to_csv(TF,mode='w' if WK==1 else 'a',header=(WK==1 or not os.path.exists(TF)),index=False)
pd.set_option('display.width',200); print(f"WEEK {WK}: signal {T.date()} close -> trade {D.date()} open | value before trades ${vb:,.2f}")
print(TR.round(2).to_string(index=False)); print({k:(round(v,2) if isinstance(v,float) else v) for k,v in H.items()})
