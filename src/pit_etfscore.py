"""Score ETFs (or any ticker) with the PIT models: each feature is ranked against that day's S&P members,
then fed to the same GBT (as in pit_week.py) and the same two MLPs (as in pit_nn.py).
GBM -> σ vs members; MLPs -> predicted 5-day log return in excess of the average member stock.
Usage: python pit_etfscore.py SIGNAL_DATE TRAIN_CUTOFF TICKERS..."""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',next((p for p in (_os.path.join(HERE,'..','..',n,'src') for n in ('ml-alpha-sp500','project4_ml_alpha')) if _os.path.isdir(p)),''))
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
T,TE=[pd.Timestamp(x) for x in sys.argv[1:3]]; TK=sys.argv[3:]
src=open(f'{HERE}/pit_week.py').read(); src=src[:src.index("out=pd.DataFrame")]
sys.argv=['x',str(T.date()),str(TE.date())]; exec(src)
def xrow(e):
    m=elig.loc[T]; v=[]
    for k in F.FEATURES:
        col=f[k].loc[T]; mv=col[m].dropna(); x=col.get(e,np.nan)
        v.append(0.0 if pd.isna(x) else ((mv<x).sum()+0.5*(mv==x).sum()+0.5)/(len(mv)+1)-0.5)
    return np.array(v)
X=np.vstack([pan.xs(T,level='date').loc[e,F.FEATURES].to_numpy() if e in pan.xs(T,level='date').index else xrow(e) for e in TK])
g=model.predict(X); gz=(g-s.mean())/s.std()
tr=TR.sample(n=min(200000,len(TR)),random_state=0); out=pd.DataFrame({'GBM σ':gz,'GBM rank of 482':[int((zz>v).sum())+1 for v in gz]},index=TK)
for solver in ['adam','lbfgs']:
    mm=MLPRegressor(hidden_layer_sizes=(32,16),activation='relu',solver=solver,alpha=1e-3,max_iter=200 if solver=='adam' else 400,
                    learning_rate_init=1e-3,batch_size=1024,early_stopping=(solver=='adam'),random_state=0).fit(tr[F.FEATURES].to_numpy(),tr.y_raw.to_numpy())
    out[f'MLP-{solver} %']=mm.predict(X)*100
feat=pd.DataFrame(X,index=TK,columns=F.FEATURES)
print(out.round(3).to_string()); print('\nfeature ranks vs members (-0.5 lowest .. +0.5 highest)'); print(feat.round(2).to_string())
