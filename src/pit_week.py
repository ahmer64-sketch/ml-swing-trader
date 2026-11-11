"""Fully point-in-time F model scoring for one signal date (S&P 500 members only).
Training: GBT on stock-days where the stock WAS an S&P member that day (no survivorship), features ranked
and labels de-meaned across that day's members only, 2013 -> signal date with purge + embargo,
settings chosen on the last 12 months (inner validation). Scoring: that day's members only."""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',next((p for p in (_os.path.join(HERE,'..','..',n,'src') for n in ('ml-alpha-sp500','project4_ml_alpha')) if _os.path.isdir(p)),''))
import sys, glob, os, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0,HERE); sys.path.insert(0,ALPHA_SRC)
from members import members, REN, CH, REVSET, REV
from alpha import data, features as F, models as M
from alpha.splits import purged_before
T=pd.Timestamp(sys.argv[1]); TE=pd.Timestamp(sys.argv[2]) if len(sys.argv)>2 else T   # TE = training cut-off (first signal of the year)
PRICES=f'{P}/prices_pit.parquet'
df,meta=data.load_prices(PRICES); df=df[df.date<=T]
p=F.panel(data.clean(df)[0]); cal=p['adj'].index; f,el=F.raw_features(p)
# membership matrix (dates x tickers), stepping changes backwards once
mem=pd.DataFrame(False,index=cal,columns=p['adj'].columns); cur=set(REVSET)
chs=CH[CH.date<=REV].sort_values('date',ascending=False); j=0; rn=lambda x: REN.get(x,x)
for d in cal[::-1]:
    while j<len(chs) and chs.date.iloc[j]>d:
        r=chs.iloc[j]
        if isinstance(r.added,str): cur.discard(rn(r.added))
        if isinstance(r.removed,str): cur.add(rn(r.removed))
        j+=1
    mem.loc[d,mem.columns.intersection(list(cur))]=True
elig=el & mem
mt=members(T); have=[s for s in mt if s in p['adj'].columns and p['adj'][s].notna().any()]
print(f"signal {T.date()}: members {len(mt)} | with price data {len(have)} | eligible today {int(elig.loc[T].sum())}")
# point-in-time panel: features ranked across members, label de-meaned across members
z=F.normalise(f,elig); raw,rk=F.label(p,elig,5)
keep=elig.stack(); keep=keep[keep].index
pan=pd.DataFrame({k:z[k].stack().reindex(keep) for k in F.FEATURES}); pan['y_raw']=raw.stack().reindex(keep); pan['y_rank']=rk.stack().reindex(keep)
pan.index.names=['date','ticker']
train=purged_before(cal,cal[(cal>=pd.Timestamp('2013-01-01'))&(cal<TE)],TE,5,5)
val=train[train>=train[-1]-pd.DateOffset(months=12)+pd.Timedelta(days=1)]; inner=purged_before(cal,train[train<val[0]],val[0],5,5)
rows=lambda ds: pan[pan.index.get_level_values('date').isin(ds)&pan.y_rank.notna()]
I,V,TR=rows(inner),rows(val),rows(train)
print(f"training rows {len(TR):,} ({train[0].date()} -> {train[-1].date()}), inner {len(I):,}, validation {len(V):,} ({val[0].date()} -> {val[-1].date()})")
best,bic=None,-9
for prm in M.GRIDS['gbt']:
    m=M.make('gbt',prm).fit(I[F.FEATURES].to_numpy(),I.y_rank.to_numpy())
    ic=M.daily_ic(pd.Series(m.predict(V[F.FEATURES].to_numpy()),V.index),V.y_rank).mean(); print('  ',prm,'validation IC',round(ic,4))
    if ic>bic: best,bic=prm,ic
model=M.make('gbt',best).fit(TR[F.FEATURES].to_numpy(),TR.y_rank.to_numpy()); print('chosen',best)
X=pan.xs(T,level='date')[F.FEATURES]; s=pd.Series(model.predict(X.to_numpy()),X.index); zz=((s-s.mean())/s.std()).sort_values(ascending=False)
out=pd.DataFrame({'sigma':zz,'close':p['close'].loc[T,zz.index],'rank':range(1,len(zz)+1)})
out.to_csv(f'{P}/pit_scores_{T.date()}.csv'); json.dump(dict(train_cutoff=str(TE.date()),signal=str(T.date()),params=best,val_ic=bic,members=len(mt),with_data=len(have),scored=len(zz)),open(f'{P}/pit_model_{T.date()}.json','w'))
print(out.head(12).round(3).to_string()); print('picks >=3σ:',list(out.index[out.sigma>=3]))
