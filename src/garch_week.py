"""GARCH(1,1) weekly vol for a list of tickers. QMLE (Nelder-Mead) on the last 756 daily log returns up to END.
Usage: python garch_week.py END in.csv out.csv   (in.csv index = tickers). Adds vol5 (5-day %), lr5 (long-run 5-day %),
alpha, beta, pers, vratio=vol5/lr5, ens_garch=ens/vol5."""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',_os.path.join(HERE,'..','..','project4_ml_alpha','src'))
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
END,IN,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
d=pd.read_csv(IN,index_col=0); p=pd.read_parquet(f'{P}/prices_pit.parquet'); p['date']=pd.to_datetime(p['date'])
p=p[p.date<=END]; px=p.pivot(index='date',columns='ticker',values='adj_close')
def fit(r):
    r=r-r.mean(); v0=r.var()
    def nll(x):
        w,a,b=x
        if w<=0 or a<0 or b<0 or a+b>=0.9999: return 1e10
        s=np.empty_like(r); s[0]=v0
        for i in range(1,len(r)): s[i]=w+a*r[i-1]**2+b*s[i-1]
        return 0.5*np.sum(np.log(s)+r**2/s)
    x=minimize(nll,[v0*0.05,0.08,0.9],method='Nelder-Mead',options=dict(maxiter=2000,xatol=1e-10,fatol=1e-8)).x
    w,a,b=x; s=v0
    for i in range(1,len(r)): s=w+a*r[i-1]**2+b*s
    s1=w+a*r[-1]**2+b*s; lr=w/(1-a-b); ph=a+b
    f=sum(lr+ph**h*(s1-lr) for h in range(5))
    return np.sqrt(f)*100,np.sqrt(5*lr)*100,a,b,ph
rows={}
for t in d.index:
    if t not in px: continue
    r=np.log(px[t].dropna()).diff().dropna().values[-756:]
    if len(r)<300: continue
    rows[t]=fit(r)
g=pd.DataFrame(rows,index=['vol5','lr5','alpha','beta','pers']).T
d=d.join(g,how='inner'); d['vratio']=d.vol5/d.lr5; d['ens_garch']=d.ens/d.vol5
d.round(4).to_csv(OUT); print(len(d),'saved',OUT)
