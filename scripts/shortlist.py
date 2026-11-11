"""All-three-up shortlist with GARCH stability filter and a 1-10 rating.
usage: python scripts/shortlist.py SIGNAL_DATE   (needs data/scores_<SIGNAL_DATE>.csv from score_week.sh)"""
import sys, os, subprocess, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.environ.get('SWING_DATA',os.path.join(HERE,'..','data'))
T=sys.argv[1]; d=pd.read_csv(f'{DATA}/scores_{T}.csv',index_col=0)
a=d[(d.g>0)&(d.adam>0)&(d.lb>0)&~d.index.isin(['TQQQ','SOXL'])]           # all three models predict a gain; no leveraged ETFs
a.to_csv(f'{DATA}/all3_{T}.csv')
subprocess.run([sys.executable,os.path.join(HERE,'..','src','garch_week.py'),T,f'{DATA}/all3_{T}.csv',f'{DATA}/garch_{T}.csv'],check=True)
g=pd.read_csv(f'{DATA}/garch_{T}.csv',index_col=0)
g=g[(g.pers<0.999)&(g.vol5>=1.5)].copy()                                   # drop degenerate fits (pinned / takeover prices)
pr=lambda x: x.rank(pct=True)
stab=np.clip(1-(g.vratio-1).clip(lower=0)/0.8,0,1)*np.where(g.vol5<=10,1,0.3)
g['score']=0.35*pr(g.g)+0.25*pr(g.ens)+0.25*pr(g.ens_garch)+0.15*stab
g['rate']=(1+9*(g.score-g.score.min())/(g.score.max()-g.score.min())).round(1)
g['stable']=(g.vol5<=10)&(g.vratio<=1.15)
g=g.sort_values('rate',ascending=False); g.to_csv(f'{DATA}/rated_{T}.csv')
print(g[['rate','g','gbm','adam','lb','ens','vol5','vratio','ens_garch','stable']].round(2).head(30).to_string())
