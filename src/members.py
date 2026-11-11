"""Point-in-time S&P 500 membership: Wikipedia revision (2026-06-30) rolled back through the changes table."""
import os as _os
HERE=_os.path.dirname(_os.path.abspath(__file__))
P=_os.environ.get('SWING_DATA',_os.path.join(HERE,'..','data'))
ALPHA_SRC=_os.environ.get('ALPHA_SRC',_os.path.join(HERE,'..','..','project4_ml_alpha','src'))
import pandas as pd
REN={'PSKY':'SKYD','SATS':'ECHO','EQR':'VMRK'}; rn=lambda x: REN.get(x,x)
rev=pd.read_csv(f'{P}/reference/members_at_rev.csv'); REV=pd.Timestamp(rev.rev_time.iloc[0][:10]); REVSET={rn(t) for t in rev.ticker}
CH=pd.read_csv(f'{P}/reference/sp500_changes.csv',parse_dates=['date']).dropna(subset=['date'])
def members(t):
    t=pd.Timestamp(t); m=set(REVSET)
    for _,r in CH[(CH.date>t)&(CH.date<=REV)].iterrows():
        if isinstance(r.added,str): m.discard(rn(r.added))
        if isinstance(r.removed,str): m.add(rn(r.removed))
    return m
def ever(t0,t1):
    """Every ticker that was a member at some point between t0 and t1."""
    s=members(t1)|members(t0)
    for _,r in CH[(CH.date>pd.Timestamp(t0))&(CH.date<=REV)].iterrows():
        for c in ('added','removed'):
            if isinstance(r[c],str): s.add(rn(r[c]))
    return s
