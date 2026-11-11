"""Three-model λ tables on the top-N shortlist (point-in-time).
μ models: GBM (shrunk past-pick mean × σ/mean σ), MLP-Adam, MLP-L-BFGS (5-day excess return prediction), ENSEMBLE (average).
Risk (same for all): GARCH(1,1) weekly vol (L-BFGS-B) × 63-day correlations; SPY: 52-week trend, 63-day vol.
Weights: max w·μ − λ/2 wᵀΣw, 0–50% per stock, ETF 0–100%, Σw = 1 (L-BFGS-B + budget price, SLSQP cross-check).
Writes lambda_<T>_top<N>_<model>.csv for the ledger (MODEL=<model>). Usage: python pit_three.py T TRAIN_CUTOFF [N=5]"""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',next((p for p in (_os.path.join(HERE,'..','..',n,'src') for n in ('ml-alpha-sp500','project4_ml_alpha')) if _os.path.isdir(p)),''))
import sys, os, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
T=pd.Timestamp(sys.argv[1]); TE=pd.Timestamp(sys.argv[2]); ARG=sys.argv[3] if len(sys.argv)>3 else "5"; S3=(ARG=="s3"); LST=ARG.split(":")[1].split(",") if ARG.startswith("list:") else None; N=None if (S3 or LST) else int(ARG)
ZS=pd.read_csv(f"{P}/pit_scores_{T.date()}.csv",index_col=0).sigma.sort_values(ascending=False); ZS=ZS[LST] if LST else (ZS[ZS>=3.0] if S3 else ZS.head(N)); TK=list(ZS.index)
if not TK:                                            # nothing qualifies: every model = 100% ETF
    rows=[[f'λ={l}',100.0,np.nan,np.nan,np.nan,np.nan] for l in [0.5,1,1.5,2,3,4,6,8,12,20]]
    for mdl in ['gbm','adam','lbfgs','ens']: pd.DataFrame(rows,columns=['weights','ETF(SPY)','exp_wk%','wk_vol%','exp/vol','growth_wk%']).to_csv(f'{P}/lambda_{T.date()}_{mdl}.csv' if S3 else f'{P}/lambda_{T.date()}_top{N}_{mdl}.csv',index=False)
    print(f"signal {T.date()}: no stock ≥3σ -> 100% ETF for every model"); sys.exit()
src=open(f'{HERE}/pit_nn.py').read(); src=src[:src.index("O=data.wide(df_all")]
sys.argv=['x',str(T.date()),str(TE.date()),str(T.date()),str(T.date())]+TK; exec(src)
dfa=pd.read_parquet(f'{P}/prices_pit.parquet'); dfa=dfa[dfa.date<=T]; Ap=dfa.pivot(index='date',columns='ticker',values='adj_close')
Op=dfa.pivot(index='date',columns='ticker',values='open'); Cp=dfa.pivot(index='date',columns='ticker',values='close'); AOp=Op*Ap/Cp
# GBM μ: shrunk mean of past PIT picks' weekly excess returns (same rule as pit_lambda.py SHRINK=1)
hist=[]; HS=[json.load(open(f'{P}/ledger_state.json'))['history'][:50]]+([json.load(open(f'{P}/ledger_state_2024.json'))['history']] if os.path.exists(f'{P}/ledger_state_2024.json') else [])+([json.load(open(f'{P}/ledger_state_2025.json'))['history']] if os.path.exists(f'{P}/ledger_state_2025.json') else [])
for h0,h1 in [pr for hh in HS for pr in zip(hh[:-1],hh[1:])]:
    if pd.Timestamp(h1['trade_date'])>T: break
    for s in h0['holdings']:
        if s not in ('SPY','CASH','QQQ','TQQQ','SOXX','SOXL','DIA','IWM'): hist.append(AOp[s].asof(pd.Timestamp(h1['trade_date']))/AOp[s].asof(pd.Timestamp(h0['trade_date']))-1-0.045/52)
mu_hat=(len(hist)*np.mean(hist)+52*0.005)/(len(hist)+52)
EX=[x for x in os.environ.get('ETFX','').split(',') if x]   # optional extra ETFs (μ = 52-wk trend, 63-day vol, 0-100%)
names=TK+EX+['SPY']; n=len(TK)+len(EX)
r=np.log(Ap[names]/Ap[names].shift()).tail(63); corr=r.corr().values
gv=np.array([garch_week_sigma(s)[0] for s in TK]+[r[e].std()*np.sqrt(5) for e in EX]+[r['SPY'].std()*np.sqrt(5)]); S=corr*np.outer(gv,gv)
trend=lambda e:(Ap[e].iloc[-1]/Ap[e].iloc[-253])**(1/52)-1; spy_mu=trend('SPY'); exm=[trend(e) for e in EX]
GD=float(os.environ['GBMDEN']) if os.environ.get('GBMDEN') else ZS.mean()   # GBMDEN: fixed σ denominator (e.g. the ≥3σ list mean) so long lists are not inflated
MU={'gbm':np.r_[(mu_hat*ZS/GD).values,exm,spy_mu],
    'adam':np.r_[[res['adam']['pred'][s] for s in TK],exm,spy_mu],
    'lbfgs':np.r_[[res['lbfgs']['pred'][s] for s in TK],exm,spy_mu]}
MU['ens']=(MU['gbm']+MU['adam']+MU['lbfgs'])/3
bnd=[(0,.5)]*len(TK)+[(0,1)]*len(EX)+[(0,1)]
def solve(mu,lam):
    def box(nu):
        q=minimize(lambda w:-(w@(mu-nu)-0.5*lam*w@S@w),np.full(n+1,1/(n+1)),jac=lambda w:-((mu-nu)-lam*S@w),method='L-BFGS-B',bounds=bnd,options={'ftol':1e-15,'gtol':1e-13})
        return q.x
    lo,hi=-1.0,1.0
    for _ in range(80):
        mid=(lo+hi)/2
        if box(mid).sum()>1: lo=mid
        else: hi=mid
    w=box((lo+hi)/2); w=w/w.sum()
    s_=min((minimize(lambda w:-(w@mu-0.5*lam*w@S@w),x0,method='SLSQP',bounds=bnd,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'ftol':1e-14}) for x0 in [np.full(n+1,1/(n+1)),np.r_[np.zeros(n),1.0]]),key=lambda q:q.fun)
    return w,np.abs(w-s_.x).max()
print(f"\nsignal {T.date()} | {('selected '+','.join(LST)) if LST else ('σ≥3' if S3 else 'top '+str(N))}: {dict(ZS.round(2))} | GBM μ̂ {mu_hat*100:.2f}%/wk ({len(hist)} past picks)")
print(pd.DataFrame({'σ':list(ZS.round(2))+[np.nan]*(len(EX)+1),'GARCH vol %':gv*100,'μ GBM %':MU['gbm']*100,'μ Adam %':MU['adam']*100,'μ L-BFGS %':MU['lbfgs']*100,'μ ensemble %':MU['ens']*100},index=names).round(3).to_string())
for mdl in ['gbm','adam','lbfgs','ens']:
    mu=MU[mdl]; rows=[]; dmax=0
    for lam in ([float(x) for x in os.environ['LAMS'].split(',')] if os.environ.get('LAMS') else [0.5,1,1.5,2,3,4,6,8,12,20]):
        w,dd=solve(mu,lam); dmax=max(dmax,dd); e=w@mu; v=np.sqrt(w@S@w)
        rows.append([f'λ={lam:g}',*np.round(w*100,1),round(e*100,2),round(v*100,2),round(e/v,3),round((e-0.5*v*v)*100,2)])
    Tb=pd.DataFrame(rows,columns=['weights',*TK,*EX,'ETF(SPY)','exp_wk%','wk_vol%','exp/vol','growth_wk%']); Tb.to_csv((f'{P}/lambda_{T.date()}_sel_{mdl}' if LST else f'{P}/lambda_{T.date()}_{mdl}' if S3 else f'{P}/lambda_{T.date()}_top{N}_{mdl}')+('_etf' if EX else '')+'.csv',index=False)
    print(f"\n=== μ model: {mdl}  (L-BFGS-B vs SLSQP max diff {dmax*100:.4f}%)"); print(Tb[Tb.weights.isin(['λ=1','λ=2','λ=4','λ=8','λ=20'])].to_string(index=False))
