"""λ (risk-aversion) weights for one week's shortlist, point-in-time.
max  w·μ − (λ/2) wᵀΣw   s.t. 0≤w_stock≤50%, w_ETF≥0, Σw=1   (solved with L-BFGS-B + budget multiplier bisection; SLSQP cross-check)
μ_stock = μ̂ × σ_i / mean σ of the shortlist; μ̂ = mean weekly excess return of past PIT picks (0.5% until 12 weeks of history)
μ_ETF = SPY past-52-week average weekly return; Σ = 63-day daily covariance × 5. Usage: python pit_lambda.py SIGNAL_DATE"""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',next((p for p in (_os.path.join(HERE,'..','..',n,'src') for n in ('ml-alpha-sp500','project4_ml_alpha')) if _os.path.isdir(p)),''))
import sys, json, os, pandas as pd, numpy as np
from scipy.optimize import minimize
T=pd.Timestamp(sys.argv[1]); TOPN=int(sys.argv[2]) if len(sys.argv)>2 else None   # TOPN: shortlist = top N by σ instead of σ≥3
df=pd.read_parquet(f'{P}/prices_pit.parquet'); df=df[df.date<=T]; A=df.pivot(index='date',columns='ticker',values='adj_close')
z=pd.read_csv(f'{P}/pit_scores_{T.date()}.csv',index_col=0).sigma; z=z.sort_values(ascending=False).head(TOPN) if TOPN else z[z>=3.0]
hist=[]
if os.path.exists(f'{P}/ledger_trades.csv'):
    tr=pd.read_csv(f'{P}/ledger_trades.csv'); st=json.load(open(f'{P}/ledger_state.json'))
    # weekly excess return of each past pick (open->next open), only weeks fully in the past
    O=df.pivot(index='date',columns='ticker',values='open'); AO=O*(A/df.pivot(index='date',columns='ticker',values='close'))
    for h0,h1 in zip(st['history'][:-1],st['history'][1:]):
        if pd.Timestamp(h1['trade_date'])>T: break
        for s in h0['holdings']:
            if s not in ('SPY','CASH'): hist.append(AO[s].asof(pd.Timestamp(h1['trade_date']))/AO[s].asof(pd.Timestamp(h0['trade_date']))-1-0.045/52)
SHRINK=os.environ.get('SHRINK')=='1'
mu_hat=(len(hist)*np.mean(hist)+52*0.005)/(len(hist)+52) if (SHRINK and hist) else (np.mean(hist) if len(hist)>=12 else 0.005)
names=list(z.index)+['SPY']; n=len(z)
r=np.log(A[names]/A[names].shift()).tail(63); S=r.cov().values*5
spy_mu=(A['SPY'].iloc[-1]/A['SPY'].iloc[-253])**(1/52)-1
mu=np.r_[(mu_hat*z/z.mean()).values,spy_mu]
print(f"{'SHRUNK μ̂' if SHRINK else 'raw μ̂'} | signal {T.date()} | shortlist {dict(z.round(2))} | past picks {len(hist)} -> μ̂ {mu_hat*100:.2f}%/wk")
print('expected wk %',dict(zip(names,np.round(mu*100,3))),'| weekly vol %',dict(zip(names,np.round(np.sqrt(np.diag(S))*100,2))))
print('correlation\n',r.corr().round(2).to_string())
def solve(lam):
    """L-BFGS-B on all assets (stocks 0-50%, ETF 0-100%) with the budget handled by a price ν (Lagrange multiplier):
    max w·(μ-ν) - λ/2 wᵀΣw under box bounds, then bisect ν until Σw = 1. Cross-checked with SLSQP (exact constraint)."""
    bnd=[(0,.5)]*n+[(0,1)]; its=[0]
    def box(nu):
        f=lambda w: -(w@(mu-nu)-0.5*lam*w@S@w); g=lambda w: -((mu-nu)-lam*S@w)
        q=minimize(f,np.full(n+1,1/(n+1)),jac=g,method='L-BFGS-B',bounds=bnd,options={'ftol':1e-15,'gtol':1e-13,'maxiter':5000}); its[0]+=q.nit; return q.x
    lo,hi=-1.0,1.0                                   # Σw(ν) falls as ν rises
    for _ in range(80):
        mid=(lo+hi)/2
        if box(mid).sum()>1: lo=mid
        else: hi=mid
    w=box((lo+hi)/2); w=w/w.sum()
    obj=lambda w: -(w@mu-0.5*lam*w@S@w); jac=lambda w: -(mu-lam*S@w)
    s_=min((minimize(obj,x0,jac=jac,method='SLSQP',bounds=bnd,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n+1)}],options={'ftol':1e-14,'maxiter':1000})
            for x0 in [np.full(n+1,1/(n+1)),np.r_[np.zeros(n),1.0]]),key=lambda r:r.fun)
    SOLVER[lam]=f"L-BFGS-B + budget price ν={(lo+hi)/2*100:.3f}%/wk ({its[0]} it total) | diff vs SLSQP {np.abs(w-s_.x).max()*100:.4f}%"
    return w
SOLVER={}
def project(v,ub):
    """Projection onto {0<=w<=ub, sum w = 1}: w = clip(v - tau, 0, ub), tau found by bisection."""
    lo,hi=v.min()-1,v.max()
    for _ in range(100):
        t=(lo+hi)/2
        if np.clip(v-t,0,ub).sum()>1: lo=t
        else: hi=t
    return np.clip(v-(lo+hi)/2,0,ub)
def solve_adam(lam,lr=0.02,b1=0.9,b2=0.999,eps=1e-12,iters=30000,tol=1e-9):
    """Projected Adam. Two fixes vs textbook Adam: (1) the gradient's common part (same for every asset) is removed,
    since the 100% budget makes it irrelevant - otherwise Adam's per-coordinate scaling turns every step into the
    same +lr for all assets and the projection undoes it (stuck at equal weights); (2) learning rate decays as 1/sqrt(k)."""
    ub=np.r_[np.full(n,.5),1.0]; w=project(np.full(n+1,1/(n+1)),ub); m=np.zeros(n+1); v=np.zeros(n+1)
    for k in range(1,iters+1):
        g=-(mu-lam*S@w); free=(w>1e-12)&(w<ub-1e-12); c=g[free].mean() if free.any() else g.mean(); g=g-c
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
        mh=m/(1-b1**k); vh=v/(1-b2**k); wn=project(w-lr/np.sqrt(k)*mh/(np.sqrt(vh)+eps),ub)
        if np.abs(wn-w).max()<tol and k>500: w=wn; break
        w=wn
    return w,k
if os.environ.get('SOLVER')=='adam':
    _lb=solve
    def solve(lam):
        w,k=solve_adam(lam); wl=_lb(lam)
        SOLVER[lam]=f"Adam ({k} steps) | diff vs L-BFGS-B {np.abs(w-wl).max()*100:.4f}% | "+SOLVER[lam]
        return w
cur=np.minimum(0.5,(z/z.sum()).values); cur=np.r_[cur,1-cur.sum()]
rows=[]
for nm,w in [('current ∝σ (50% cap)',cur)]+[(f'λ={l}',solve(l)) for l in [0.5,1,1.5,2,3,4,6,8,12,20]]:
    e=w@mu; v=np.sqrt(w@S@w); rows.append([nm,*np.round(w*100,1),round(e*100,2),round(v*100,2),round(e/v,3),round((e-0.5*v*v)*100,2)])
Tb=pd.DataFrame(rows,columns=['weights',*z.index,'ETF(SPY)','exp_wk%','wk_vol%','exp/vol','growth_wk%']); print(Tb.to_string(index=False))
Tb.to_csv(f'{P}/lambda_{T.date()}'+(f'_top{TOPN}' if TOPN else '')+('_shrink' if SHRINK else '')+'.csv',index=False)
for k,v in SOLVER.items(): print(f'  λ={k}: {v}')
