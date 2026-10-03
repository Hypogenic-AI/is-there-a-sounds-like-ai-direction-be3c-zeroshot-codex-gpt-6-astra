"""Additional training-set separation diagnostics for base/instruct models."""
from common import *
import pandas as pd
rows=readj('data/corpus.jsonl');y=np.array([r['label'] for r in rows]);tr=np.array([r['split']=='train' for r in rows]);res=[]
for l in LAYERS:
 for name in ['base','instruct']:
  f=np.load('results/features_'+name+'.npz');X=f[str(l)][:len(rows)];d=X[tr&(y==1)].mean(0)-X[tr&(y==0)].mean(0);u=d/np.linalg.norm(d);s=X@u;scale=np.sqrt((s[tr&(y==0)].var()+s[tr&(y==1)].var())/2)
  res.append({'model':name,'layer':l,'mean_difference_norm':float(np.linalg.norm(d)),'separation_standardized':float(np.linalg.norm(d)/scale)})
pd.DataFrame(res).to_csv('results/base_instruct_separation.csv',index=False)
