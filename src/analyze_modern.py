from common import *
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rng=np.random.default_rng(281)
def ci(d):
 d=np.asarray(d);return np.quantile(d[rng.integers(len(d),size=(2000,len(d)))].mean(1),[.025,.975])
raw=readj('results/modern_detector_scores.jsonl');scores={(r['key'],r['mode']):r['score'] for r in raw};logits={(r['key'],r['mode']):r['logit'] for r in raw};jud={r['key']:r['ratings'] for r in readj('results/judge_raw.jsonl') if 'ratings' in r};rows=[]
for r in readj('results/generations.jsonl')+readj('results/transfer_generations.jsonl'):
 key=r['condition']+'__'+r['id'];rows.append({**r,**jud[key],'modern':scores[(key,'full')],'modern_logit':logits[(key,'full')],'modern_fixed_logit':logits.get((key,'fixed64'),np.nan),'modern_fixed':scores.get((key,'fixed64'),np.nan)})
df=pd.DataFrame(rows);df.to_csv('results/modern_scored.csv',index=False);b=df[df.condition=='baseline'].set_index('id');summary=[]
for cond,g in df.groupby('condition',sort=False):
 g=g.set_index('id').loc[b.index];good=(g.coherence>=4)&(g.adequacy>=4)&(b.coherence>=4)&(b.adequacy>=4);delta=(g.modern-b.modern).to_numpy();lo,hi=ci(delta);fixed=(g.modern_fixed-b.modern_fixed).dropna();flo,fhi=ci(fixed.to_numpy());qualified=delta[good];qlo,qhi=ci(qualified)
 summary.append({'condition':cond,'mean':g.modern.mean(),'delta':delta.mean(),'lo':lo,'hi':hi,'quality_n':int(good.sum()),'quality_delta':qualified.mean(),'quality_lo':qlo,'quality_hi':qhi,'fixed_n':len(fixed),'fixed_delta':fixed.mean(),'fixed_lo':flo,'fixed_hi':fhi})
for row in summary:
 g=df[df.condition==row['condition']].set_index('id').loc[b.index]
 delta=(g.modern_logit-b.modern_logit).to_numpy();lo,hi=ci(delta);fixed=(g.modern_fixed_logit-b.modern_fixed_logit).dropna();flo,fhi=ci(fixed.to_numpy())
 good=(g.coherence>=4)&(g.adequacy>=4)&(b.coherence>=4)&(b.adequacy>=4);qlo,qhi=ci(delta[good])
 row.update({'logit_mean':g.modern_logit.mean(),'logit_delta':delta.mean(),'logit_lo':lo,'logit_hi':hi,'fixed_logit_delta':fixed.mean(),'fixed_logit_lo':flo,'fixed_logit_hi':fhi,'quality_logit_delta':delta[good].mean(),'quality_logit_lo':qlo,'quality_logit_hi':qhi,'detection_rate':float((g.modern>=.5).mean())})
s=pd.DataFrame(summary);s.to_csv('results/modern_summary.csv',index=False)
cal=readj('data/calibration.jsonl');cr=[]
for ds in ['HC3','HAPE']:
 rr=[r for r in cal if r['dataset']==ds];ids=list(dict.fromkeys(r['id'] for r in rr));ix=np.array([ids.index(r['id']) for r in rr]);y=np.array([r['label'] for r in rr]);sc=np.array([scores[(r['key'],'full')] for r in rr]);au=[]
 for _ in range(2000):
  weight=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))[ix];au.append(roc_auc_score(y,sc,sample_weight=weight))
 lo,hi=np.quantile(au,[.025,.975]);cr.append({'dataset':ds,'measure':'Desklib','n':len(rr),'auc':roc_auc_score(y,sc),'lo':lo,'hi':hi,'human_mean':sc[y==0].mean(),'machine_mean':sc[y==1].mean()})
pd.DataFrame(cr).to_csv('results/modern_calibration.csv',index=False)
contr=[]
for v in ['auth','formal','persona','orth','random101','random202','hape']:
 for a in ([.5,1,2] if v=='auth' else [1]):
  pos=df[df.condition==v+'_'+str(a)].set_index('id').loc[b.index].modern;neg=df[df.condition==v+'_'+str(-a)].set_index('id').loc[b.index].modern;d=(pos-neg).to_numpy();lo,hi=ci(d);contr.append({'vector':v,'alpha':a,'delta':d.mean(),'lo':lo,'hi':hi})
for row in contr:
 v=row['vector'];a=row['alpha'];pos=df[df.condition==v+'_'+str(a)].set_index('id').loc[b.index].modern_logit;neg=df[df.condition==v+'_'+str(-a)].set_index('id').loc[b.index].modern_logit;d=(pos-neg).to_numpy();lo,hi=ci(d);row.update({'logit_delta':d.mean(),'logit_lo':lo,'logit_hi':hi})
pd.DataFrame(contr).to_csv('results/modern_sign_contrasts.csv',index=False)
specificity=[]
auth=(df[df.condition=='auth_1'].set_index('id').loc[b.index].modern-df[df.condition=='auth_-1'].set_index('id').loc[b.index].modern)
for v in ['formal','persona','orth','random101','random202','hape']:
 dd=(auth-df[df.condition==v+'_1'].set_index('id').loc[b.index].modern+df[df.condition==v+'_-1'].set_index('id').loc[b.index].modern).to_numpy();lo,hi=ci(dd);specificity.append({'control':v,'delta':dd.mean(),'lo':lo,'hi':hi})
authlog=(df[df.condition=='auth_1'].set_index('id').loc[b.index].modern_logit-df[df.condition=='auth_-1'].set_index('id').loc[b.index].modern_logit)
for row in specificity:
 v=row['control'];dd=(authlog-df[df.condition==v+'_1'].set_index('id').loc[b.index].modern_logit+df[df.condition==v+'_-1'].set_index('id').loc[b.index].modern_logit).to_numpy();lo,hi=ci(dd);row.update({'logit_delta':dd.mean(),'logit_lo':lo,'logit_hi':hi})
pd.DataFrame(specificity).to_csv('results/modern_specificity.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(11,3.2));primary=pd.read_csv('results/summary.csv');ss=s[s.condition.str.startswith('auth_')].copy();ss['a']=ss.condition.str.replace('auth_','').astype(float);ss=pd.concat([ss,pd.DataFrame([{**s[s.condition=='baseline'].iloc[0].to_dict(),'a':0}])]).sort_values('a')
for ax,mean,lo,hi,title in zip(axs,['delta','logit_delta','fixed_logit_delta'],['lo','logit_lo','fixed_logit_lo'],['hi','logit_hi','fixed_logit_hi'],['Full-answer sigmoid score','Full-answer logit (exploratory)','Fixed 64-token logit']):
 ax.errorbar(ss.a,ss[mean],yerr=[ss[mean]-ss[lo],ss[hi]-ss[mean]],marker='o',capsize=3);ax.axhline(0,color='gray',ls='--');ax.set(xlabel='Authorship steering strength',ylabel='Change from baseline',title=title)
fig.tight_layout();fig.savefig('paper_draft/figures/modern_dose.pdf');plt.close(fig)
print(s.to_string(index=False));print(pd.DataFrame(cr).to_string(index=False))
