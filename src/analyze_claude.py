"""Late external measurement audit; no changes to interventions or previous outputs."""
from common import *
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rng=np.random.default_rng(6401)
def ci(d):
 d=np.asarray(d);return np.quantile(d[rng.integers(len(d),size=(2000,len(d)))].mean(1),[.025,.975])
j={r['key']:r['ratings'] for r in readj('results/claude_judge_raw.jsonl') if 'ratings' in r};rr=readj('results/generations.jsonl')+readj('results/transfer_generations.jsonl');assert all(r['condition']+'__'+r['id'] in j for r in rr)
old={r['key']:r['detector'] for r in readj('results/detector_scores.jsonl')};modern={r['key']:r for r in readj('results/modern_detector_scores.jsonl') if r['mode']=='full'};rows=[]
for r in rr:
 k=r['condition']+'__'+r['id'];rows.append({**r,**j[k],'roberta':old[k],'desklib':modern[k]['score'],'logit':modern[k]['logit']})
df=pd.DataFrame(rows);df.to_csv('results/claude_scored.csv',index=False);b=df[df.condition=='baseline'].set_index('id');res=[]
for cond,g in df.groupby('condition',sort=False):
 g=g.set_index('id').loc[b.index];good=(g.coherence>=4)&(g.adequacy>=4)&(b.coherence>=4)&(b.adequacy>=4);row={'condition':cond,'quality_pass':float(((g.coherence>=4)&(g.adequacy>=4)).mean()),'matched_n':int(good.sum())}
 for met in ['ai_style','coherence','adequacy','formality']:
  delta=(g[met]-b[met]).to_numpy();lo,hi=ci(delta);row.update({met:g[met].mean(),met+'_delta':delta.mean(),met+'_lo':lo,met+'_hi':hi})
 for met in ['ai_style','roberta','desklib','logit']:
  delta=(g[met]-b[met]).to_numpy()[good];lo,hi=ci(delta);row.update({met+'_matched_delta':delta.mean(),met+'_matched_lo':lo,met+'_matched_hi':hi})
 res.append(row)
pd.DataFrame(res).to_csv('results/claude_summary.csv',index=False)
cal=readj('data/calibration.jsonl');cr=[]
for ds in ['HC3','HAPE']:
 rr=[r for r in cal if r['dataset']==ds];ids=list(dict.fromkeys(r['id'] for r in rr));ix=np.array([ids.index(r['id']) for r in rr]);y=np.array([r['label'] for r in rr]);score=np.array([j[r['key']]['ai_style'] for r in rr]);au=[]
 for _ in range(2000):
  w=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))[ix];au.append(roc_auc_score(y,score,sample_weight=w))
 lo,hi=np.quantile(au,[.025,.975]);cr.append({'dataset':ds,'measure':'Claude','n':len(rr),'auc':roc_auc_score(y,score),'lo':lo,'hi':hi,'human_mean':score[y==0].mean(),'machine_mean':score[y==1].mean()})
pd.DataFrame(cr).to_csv('results/claude_calibration.csv',index=False)
contr=[]
for v in ['auth','formal','persona','orth','random101','random202','hape']:
 for a in ([.5,1,2] if v=='auth' else [1]):
  pos=df[df.condition==v+'_'+str(a)].set_index('id').loc[b.index];neg=df[df.condition==v+'_'+str(-a)].set_index('id').loc[b.index]
  for met in ['ai_style','adequacy','formality']:
   d=(pos[met]-neg[met]).to_numpy();lo,hi=ci(d);contr.append({'vector':v,'alpha':a,'metric':met,'delta':d.mean(),'lo':lo,'hi':hi})
pd.DataFrame(contr).to_csv('results/claude_sign_contrasts.csv',index=False)
spec=[]
a=df[df.condition=='auth_1'].set_index('id').loc[b.index].ai_style-df[df.condition=='auth_-1'].set_index('id').loc[b.index].ai_style
for v in ['formal','persona','orth','random101','random202','hape']:
 delta=(a-df[df.condition==v+'_1'].set_index('id').loc[b.index].ai_style+df[df.condition==v+'_-1'].set_index('id').loc[b.index].ai_style).to_numpy();lo,hi=ci(delta);spec.append({'control':v,'delta':delta.mean(),'lo':lo,'hi':hi})
pd.DataFrame(spec).to_csv('results/claude_specificity.csv',index=False)
audit=readj('results/quality_audit_inputs.jsonl');rows=[{**r,**j[r['key']]} for r in audit];ids={r['id'] for r in audit};rows +=[{**r,**j['baseline__'+r['id']],'condition':'original'} for r in readj('results/generations.jsonl') if r['condition']=='baseline' and r['id'] in ids];pd.DataFrame(rows).groupby('condition')[['coherence','adequacy','ai_style']].mean().to_csv('results/claude_quality_audit.csv')
s=pd.DataFrame(res);ss=s[s.condition.str.startswith('auth_')].copy();ss['a']=ss.condition.str.replace('auth_','').astype(float);ss=pd.concat([ss,pd.DataFrame([{**s[s.condition=='baseline'].iloc[0].to_dict(),'a':0}])]).sort_values('a');fig,axs=plt.subplots(1,3,figsize=(11,3.2))
for ax,met,title in zip(axs,['ai_style','adequacy','formality'],['Claude AI-style rating','Claude answer adequacy','Claude formality']):
 ax.errorbar(ss.a,ss[met+'_delta'],yerr=[ss[met+'_delta']-ss[met+'_lo'],ss[met+'_hi']-ss[met+'_delta']],marker='o',capsize=3);ax.axhline(0,color='gray',ls='--');ax.set(xlabel='Authorship steering strength',ylabel='Change from baseline',title=title)
fig.tight_layout();fig.savefig('paper_draft/figures/claude_dose.pdf');plt.close(fig)
print(s[['condition','ai_style','ai_style_delta','ai_style_lo','ai_style_hi','adequacy_delta','adequacy_lo','adequacy_hi','quality_pass','matched_n']].to_string(index=False));print(pd.DataFrame(cr).to_string(index=False))
