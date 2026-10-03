from common import *
import pandas as pd
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(992)
gen=readj('results/transfer_generations.jsonl')+[r for r in readj('results/generations.jsonl') if r['condition']=='baseline'];judges={r['key']:r['ratings'] for r in readj('results/judge_raw.jsonl') if 'ratings' in r};det={r['key']:r['detector'] for r in readj('results/detector_scores.jsonl')};rows=[]
for r in gen:
 k=r['condition']+'__'+r['id'];rows.append({**r,**judges[k],'detector':det[k]})
df=pd.DataFrame(rows);df.to_csv('results/transfer_scored.csv',index=False);b=df[df.condition=='baseline'].set_index('id');res=[]
for cond,g in df.groupby('condition',sort=False):
 g=g.set_index('id').loc[b.index];good=(g.coherence>=4)&(g.adequacy>=4)&(b.coherence>=4)&(b.adequacy>=4)
 row={'condition':cond,'n':len(g),'quality_pass':float(((g.coherence>=4)&(g.adequacy>=4)).mean()),'matched_n':int(good.sum())}
 for met in ['detector','ai_style','coherence','adequacy','formality','tokens']:
  dd=(g[met]-b[met]).to_numpy();lo,hi=np.quantile(dd[rng.integers(len(dd),size=(2000,len(dd)))].mean(1),[.025,.975]);row.update({met:float(g[met].mean()),met+'_delta':float(dd.mean()),met+'_lo':float(lo),met+'_hi':float(hi)})
  if met in ['detector','ai_style']:row[met+'_matched_delta']=float(dd[good].mean()) if good.sum() else None
 res.append(row)
pd.DataFrame(res).to_csv('results/transfer_summary.csv',index=False);print(pd.DataFrame(res).to_string(index=False))
