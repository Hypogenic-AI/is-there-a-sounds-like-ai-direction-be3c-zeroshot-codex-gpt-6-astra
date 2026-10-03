from common import *
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
Path('paper_draft/figures').mkdir(exist_ok=True)
RNG=np.random.default_rng(1234)
def ci(a):
 a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
 if not len(a):return [None,None]
 return np.quantile(a[RNG.integers(len(a),size=(2000,len(a)))].mean(1),[.025,.975]).tolist()
def auc_ci(df):
 ids=df.id.unique();groups={id:df[df.id==id] for id in ids};vals=[]
 # Paired cluster resampling, using equivalent sample weights.
 idx=np.array([list(ids).index(id) for id in df.id]);y=df.label.to_numpy();s=df.score.to_numpy()
 for _ in range(2000):
  w=np.bincount(RNG.integers(len(ids),size=len(ids)),minlength=len(ids))[idx]
  vals.append(roc_auc_score(y,s,sample_weight=w))
 return np.quantile(vals,[.025,.975])
gens=readj('results/generations.jsonl');judges={r['key']:r['ratings'] for r in readj('results/judge_raw.jsonl') if 'ratings' in r};det={r['key']:r['detector'] for r in readj('results/detector_scores.jsonl')};nll={r['key']:r['nll'] for r in readj('results/generation_nll.jsonl')}
rows=[]
for r in gens:
 key=r['condition']+'__'+r['id'];words=r['text'].lower().split();tri=list(zip(words,words[1:],words[2:]));rep=1-len(set(tri))/len(tri) if tri else 0
 rows.append({**r,**judges.get(key,{}),'detector':det.get(key,np.nan),'nll':nll.get(key,np.nan),'words':len(words),'repetition':rep})
df=pd.DataFrame(rows);df.to_csv('results/scored_generations.csv',index=False)
assert len(df)==1140 and df.ai_style.notna().all(),(len(df),df.ai_style.isna().sum())
b=df[df.condition=='baseline'].set_index('id');summary=[];paired=[]
metrics=['detector','ai_style','coherence','adequacy','formality','tokens','words','nll','repetition']
for cond,g in df.groupby('condition',sort=False):
 g=g.set_index('id').loc[b.index];good=(g.coherence>=4)&(g.adequacy>=4)&(b.coherence>=4)&(b.adequacy>=4)
 row={'condition':cond,'n':len(g),'quality_pass':float(((g.coherence>=4)&(g.adequacy>=4)).mean()),'matched_n':int(good.sum())}
 for m in metrics:
  a=g[m].to_numpy();delta=a-b[m].to_numpy();lo,hi=ci(delta)
  row.update({m:float(a.mean()),m+'_delta':float(delta.mean()),m+'_lo':lo,m+'_hi':hi})
  if m in ['detector','ai_style']:
   dd=delta[good];lo,hi=ci(dd);row.update({m+'_matched_delta':float(dd.mean()) if len(dd) else None,m+'_matched_lo':lo,m+'_matched_hi':hi})
 summary.append(row)
s=pd.DataFrame(summary);s.to_csv('results/summary.csv',index=False)
# Direct sign contrasts, each paired by question.
for vec in ['auth','random101','random202','formal','persona','orth']:
 for alpha in ([.5,1,2] if vec=='auth' else [1]):
  aa=str(alpha);neg=df[df.condition==vec+'_'+str(-alpha)].set_index('id').loc[b.index];pos=df[df.condition==vec+'_'+aa].set_index('id').loc[b.index]
  good=(neg.coherence>=4)&(neg.adequacy>=4)&(pos.coherence>=4)&(pos.adequacy>=4)
  for met in ['detector','ai_style','coherence','adequacy']:
   d=(pos[met]-neg[met]).to_numpy();lo,hi=ci(d);paired.append({'vector':vec,'alpha':alpha,'metric':met,'delta_plus_minus':d.mean(),'lo':lo,'hi':hi,'n':len(d),'quality_matched_n':int(good.sum()),'quality_matched_delta':float(d[good].mean()) if good.sum() else None})
pd.DataFrame(paired).to_csv('results/sign_contrasts.csv',index=False)
# Difference-in-differences specificity contrasts, retaining question pairing.
specificity=[]
for met in ['detector','ai_style']:
 ap=df[df.condition=='auth_1'].set_index('id').loc[b.index][met];an=df[df.condition=='auth_-1'].set_index('id').loc[b.index][met]
 for control in ['formal','persona','random101','random202','orth']:
  cp=df[df.condition==control+'_1'].set_index('id').loc[b.index][met];cn=df[df.condition==control+'_-1'].set_index('id').loc[b.index][met]
  delta=(ap-an-cp+cn).to_numpy();lo,hi=ci(delta);specificity.append({'control':control,'metric':met,'auth_sign_effect_minus_control':delta.mean(),'lo':lo,'hi':hi})
pd.DataFrame(specificity).to_csv('results/specificity.csv',index=False)

# Detector validation; not treated as calibrated authorship probabilities.
cal=[]
for r in readj('data/calibration.jsonl'):
 if r['key'] in judges:cal.append({**r,**judges[r['key']],'detector':det[r['key']]})
cd=pd.DataFrame(cal);cd.drop(columns=['text','question'],errors='ignore').to_csv('results/calibration_scores.csv',index=False)
cr=[]
for ds,sub in cd.groupby('dataset'):
 for met in ['detector','ai_style']:
  c=sub.rename(columns={met:'score'});lo,hi=auc_ci(c);cr.append({'dataset':ds,'measure':met,'n':len(sub),'auc':roc_auc_score(sub.label,sub[met]),'lo':lo,'hi':hi,'human_mean':sub[sub.label==0][met].mean(),'machine_mean':sub[sub.label==1][met].mean()})
pd.DataFrame(cr).to_csv('results/calibration_summary.csv',index=False)
# Readout uncertainties for validation-selected layer, identical layer for base comparison.
info=json.load(open('results/direction_info.json'));rs=pd.DataFrame(readj('results/readout_scores.jsonl'));rr=[]
for (model,method,ds),sub in rs[rs.layer==info['layer']].groupby(['model','method','dataset']):
 lo,hi=auc_ci(sub);rr.append({'model':model,'method':method,'dataset':ds,'layer':info['layer'],'auc':roc_auc_score(sub.label,sub.score),'lo':lo,'hi':hi})
pd.DataFrame(rr).to_csv('results/readout_ci.csv',index=False)
# Orthogonalized readout without refitting; train-only direction.
f=np.load('results/features_instruct.npz');corp=readj('data/corpus.jsonl');ood=readj('data/ood.jsonl');vec=np.load('results/directions.npz');orth=[]
for name in ['auth','orth','formal','persona']:
 sc=f[str(info['layer'])]@vec[name]
 for ds,ix,yy in [('HC3_test',[i for i,r in enumerate(corp) if r['split']=='test'],None),('HAPE_gpt4mini',[len(corp)+i for i,r in enumerate(ood) if r['generator'] in ['human','gpt4mini']],None),('HAPE_llama8i',[len(corp)+i for i,r in enumerate(ood) if r['generator'] in ['human','llama8i']],None)]:
  allr=corp+ood;orth.append({'direction':name,'dataset':ds,'auc':roc_auc_score([allr[i]['label'] for i in ix],sc[ix])})
pd.DataFrame(orth).to_csv('results/orth_readout.csv',index=False)
# Figures.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(11,3.2))
for ax,met,title in zip(axs,['detector','ai_style','adequacy'],['Frozen detector score','Blinded AI-style rating','Answer adequacy']):
 ss=s[s.condition.str.startswith('auth_')].copy();ss['a']=ss.condition.str.replace('auth_','').astype(float);ss=pd.concat([ss,pd.DataFrame([{**s[s.condition=='baseline'].iloc[0].to_dict(),'a':0}])]).sort_values('a')
 ax.errorbar(ss.a,ss[met+'_delta'],yerr=[ss[met+'_delta']-ss[met+'_lo'],ss[met+'_hi']-ss[met+'_delta']],marker='o',capsize=3)
 ax.axhline(0,color='gray',ls='--',lw=1);ax.set_xlabel('Authorship steering strength');ax.set_title(title);ax.set_ylabel('Change from baseline')
fig.tight_layout();fig.savefig('paper_draft/figures/dose.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,3.8))
cc=pd.DataFrame(paired);lab={'auth':'Authorship','random101':'Random 101','random202':'Random 202','formal':'Formality','persona':'Persona proxy','orth':'Orthogonalized'}
for ax,met,title in zip(axs,['detector','ai_style'],['Frozen detector','Blinded AI-style judge']):
 ss=cc[(cc.alpha==1)&(cc.metric==met)];x=np.arange(len(ss));ax.errorbar(ss.delta_plus_minus,x,xerr=[ss.delta_plus_minus-ss.lo,ss.hi-ss.delta_plus_minus],fmt='o',capsize=3);ax.set_yticks(x,[lab[v] for v in ss.vector]);ax.axvline(0,color='gray',ls='--');ax.set_title(title);ax.set_xlabel('Positive minus negative steering')
fig.tight_layout();fig.savefig('paper_draft/figures/controls.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3.5));ro=pd.read_csv('results/readout.csv')
for kind in ['base','instruct']:
 for ds,ls in [('HC3_test','-'),('HAPE_gpt4mini','--'),('HAPE_llama8i',':')]:
  sub=ro[(ro.model==kind)&(ro.method=='mean')&(ro.dataset==ds)];ax.plot(sub.layer,sub.auc,ls,marker='o',label=kind+' / '+ds.replace('HAPE_','').replace('HC3_test','HC3'))
ax.axhline(.5,color='gray',lw=1);ax.set(xlabel='Residual block',ylabel='AUROC',ylim=(.35,1.02));ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig('paper_draft/figures/readout.pdf');plt.close(fig)
print(s[['condition','detector','ai_style','coherence','adequacy','quality_pass','matched_n']].to_string(index=False))
print(pd.DataFrame(cr).to_string(index=False));print(info)
