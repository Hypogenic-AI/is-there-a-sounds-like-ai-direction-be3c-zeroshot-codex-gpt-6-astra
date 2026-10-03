from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
corpus=readj('data/corpus.jsonl');ood=readj('data/ood.jsonl');y=np.array([r['label'] for r in corpus]);tr=np.array([r['split']=='train' for r in corpus]);rows=[];scores=[]
for kind in ['base','instruct']:
 f=np.load('results/features_'+kind+'.npz')
 for l in LAYERS:
  X=f[str(l)][:len(corpus)];O=f[str(l)][len(corpus):];d=X[tr&(y==1)].mean(0)-X[tr&(y==0)].mean(0)
  probe=LogisticRegression(C=1,max_iter=3000).fit(X[tr],y[tr])
  for method in ['mean','logistic']:
   sc=X@d if method=='mean' else probe.decision_function(X);os=O@d if method=='mean' else probe.decision_function(O)
   for split in ['val','test']:
    ix=np.array([r['split']==split for r in corpus]);rows.append(dict(model=kind,layer=l,method=method,dataset='HC3_'+split,auc=roc_auc_score(y[ix],sc[ix]),n=int(ix.sum())))
    for r,s in zip(np.array(corpus,dtype=object)[ix],sc[ix]):scores.append(dict(model=kind,layer=l,method=method,dataset='HC3_'+split,id=r['id'],label=r['label'],score=float(s)))
   for gen in ['gpt4mini','llama8i']:
    ix=np.array([r['generator'] in ['human',gen] for r in ood]);oy=np.array([r['label'] for r in ood]);rows.append(dict(model=kind,layer=l,method=method,dataset='HAPE_'+gen,auc=roc_auc_score(oy[ix],os[ix]),n=int(ix.sum())))
    for r,s in zip(np.array(ood,dtype=object)[ix],os[ix]):scores.append(dict(model=kind,layer=l,method=method,dataset='HAPE_'+gen,id=r['id'],label=r['label'],score=float(s)))
pd.DataFrame(rows).to_csv('results/readout.csv',index=False);writej('results/readout_scores.jsonl',scores);print(pd.DataFrame(rows).to_string(index=False))
