"""Point-in-time MLP return predictor trained with Adam and with L-BFGS (same 12 rank features, same purged training
window and member-only universe as the PIT GBT), plus GARCH(1,1) volatility fitted by L-BFGS-B.
Usage: python pit_nn.py SIGNAL_DATE TRAIN_CUTOFF TRADE_DATE END_DATE TICKERS..."""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',_os.path.join(HERE,'..','..','project4_ml_alpha','src'))
import sys, os, json, warnings, time; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
T,TE,D0,D1=[pd.Timestamp(x) for x in sys.argv[1:5]]; TK=sys.argv[5:]
# reuse pit_week.py's panel construction (members-only features & labels), without its model part
src=open(f'{HERE}/pit_week.py').read(); src=src[:src.index("best,bic=None,-9")]
sys.argv=['x',str(T.date()),str(TE.date())]; exec(src)
from alpha.models import daily_ic
tr=TR.sample(n=min(200000,len(TR)),random_state=0); Xtr=tr[F.FEATURES].to_numpy(); ytr=tr.y_raw.to_numpy()
Vv=V; res={}
for solver in ['adam','lbfgs']:
    t0=time.time()
    m=MLPRegressor(hidden_layer_sizes=(32,16),activation='relu',solver=solver,alpha=1e-3,max_iter=200 if solver=='adam' else 400,
                   learning_rate_init=1e-3,batch_size=1024,early_stopping=(solver=='adam'),random_state=0).fit(Xtr,ytr)
    ic=daily_ic(pd.Series(m.predict(V[F.FEATURES].to_numpy()),V.index),V.y_raw).mean()
    Xt=pan.xs(T,level='date').reindex(TK)[F.FEATURES].to_numpy(); pr=m.predict(Xt)
    res[solver]=dict(model=m,ic=ic,pred=dict(zip(TK,pr)),secs=time.time()-t0,iters=m.n_iter_)
    print(f"MLP trained with {solver}: {m.n_iter_} iterations, {time.time()-t0:.0f}s, validation IC {ic:.4f}")
# GARCH(1,1) by L-BFGS-B, reparametrised so alpha+beta<1: a=p*s, b=p*(1-s), p in (0,0.999), s in (0,1)
AC=p['adj']
def garch_week_sigma(s):
    r=np.log(AC[s]/AC[s].shift()).loc[:T].dropna().values[-750:]*100; r=r-r.mean()
    def nll(x):
        w,pp,ss=x; a,b=pp*ss,pp*(1-ss); v=np.empty_like(r); v[0]=r.var()
        for i in range(1,len(r)): v[i]=w+a*r[i-1]**2+b*v[i-1]
        return 0.5*np.sum(np.log(2*np.pi*v)+r**2/v)
    q=minimize(nll,[0.05*r.var(),0.95,0.1],method='L-BFGS-B',bounds=[(1e-6,None),(0.01,0.999),(0.001,0.999)])
    w,pp,ss=q.x; a,b=pp*ss,pp*(1-ss); v=r.var()
    for i in range(1,len(r)): v=w+a*r[i-1]**2+b*v
    vn=w+a*r[-1]**2+b*v; lr=w/(1-a-b); f5=sum(lr+(a+b)**k*(vn-lr) for k in range(5))
    return np.sqrt(f5)/100, dict(alpha=a,beta=b,persist=a+b,iters=q.nit)
O=data.wide(df_all,'open') if 'df_all' in dir() else None
dfa=pd.read_parquet(f'{P}/prices_pit.parquet'); Op=dfa.pivot(index='date',columns='ticker',values='open'); Cp=dfa.pivot(index='date',columns='ticker',values='close'); Ap=dfa.pivot(index='date',columns='ticker',values='adj_close'); AOp=Op*Ap/Cp
rows=[]
for s in TK:
    sg,gp=garch_week_sigma(s); S0=Op.at[D0,s]; act=Op.at[D1,s]; ret=AOp.at[D1,s]/AOp.at[D0,s]-1
    s63=np.log(AC[s]/AC[s].shift()).loc[:T].tail(63).std()*np.sqrt(5)
    for nm in ['adam','lbfgs']:
        mu=res[nm]['pred'][s]+0.0                          # predicted 5-day excess log return
        rows.append(dict(stock=s,model=f'MLP-{nm}',pred_wk_pct=mu*100,buy=S0,pred_price=S0*np.exp(mu),
             lo68=S0*np.exp(mu-sg),hi68=S0*np.exp(mu+sg),lo95=S0*np.exp(mu-2*sg),hi95=S0*np.exp(mu+2*sg),actual=act,actual_pct=ret*100,z=(np.log(act/S0)-mu)/sg))
    rows.append(dict(stock=s,model='GARCH vol (L-BFGS-B)',pred_wk_pct=np.nan,buy=S0,pred_price=np.nan,lo68=sg*100,hi68=s63*100,lo95=gp['alpha'],hi95=gp['beta'],actual=gp['persist'],actual_pct=np.nan,z=gp['iters']))
R=pd.DataFrame(rows); R.to_csv(f'{P}/nn_pred_{T.date()}.csv',index=False); pd.set_option('display.width',250); print(R.round(3).to_string(index=False))
