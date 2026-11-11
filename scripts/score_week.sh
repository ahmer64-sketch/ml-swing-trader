#!/usr/bin/env zsh
# Score one signal date: every S&P 500 member that day plus index ETFs, with all three models.
# usage: zsh scripts/score_week.sh SIGNAL_DATE TRAIN_CUTOFF   ->  data/scores_<SIGNAL_DATE>.csv
#        columns: g (GBM z-score σ), rank, adam, lb, gbm, ens  (model columns = expected 5-day excess return, %)
set -e
cd "${0:A:h}/../src"; T=$1; TE=$2; DATA=${SWING_DATA:-../data}
ETF=(SPY QQQ TQQQ SOXX SOXL DIA IWM)
python pit_week.py $T $TE 2>&1 | grep -vi warn | grep -v "^  " | tail -14
sc=$(python -c "import pandas as pd; print(' '.join(pd.read_csv('$DATA/pit_scores_$T.csv',index_col=0).index))")
python pit_etfscore.py $T $TE ${=sc} $ETF 2>&1 | grep -vi warn > $DATA/etfscore_$T.txt
python - $T $DATA <<'PY'
import sys, pandas as pd
T,DATA=sys.argv[1:3]; L=open(f'{DATA}/etfscore_{T}.txt').read().splitlines()
i=[k for k,l in enumerate(L) if 'GBM σ' in l and 'MLP' in l][0]; rows=[]
for l in L[i+1:]:
    if l.startswith('feature'): break
    x=l.split()
    if len(x)==5: rows.append([x[0]]+[float(v) for v in x[1:]])
D=pd.DataFrame(rows,columns=['t','g','rank','adam','lb']).set_index('t')
ETF=['SPY','QQQ','TQQQ','SOXX','SOXL','DIA','IWM']
s3=D.drop(ETF,errors='ignore'); s3=s3[s3.g>=3]; den=s3.g.mean() if len(s3) else 3.6
D['gbm']=0.0049*D.g/den*100          # GBM σ mapped to % on the scale of the average ≥3σ pick (0.49%/week)
D['ens']=(D.gbm+D.adam+D.lb)/3
D.to_csv(f'{DATA}/scores_{T}.csv'); print('saved',f'{DATA}/scores_{T}.csv',len(D))
PY
