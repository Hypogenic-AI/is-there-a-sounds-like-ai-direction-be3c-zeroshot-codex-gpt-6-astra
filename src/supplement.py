"""Additional diagnostics: domain transfer, fixed-length detector, and intervention implementation checks."""
from common import *
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import pandas as pd
corp=readj('data/corpus.jsonl');info=json.load(open('results/direction_info.json'));layer=info['layer'];y=np.array([r['label'] for r in corp]);dom=np.array([r['domain'] for r in corp]);tr=np.array([r['split']=='train' for r in corp]);te=np.array([r['split']=='test' for r in corp]);results=[]
for kind in ['base','instruct']:
 f=np.load('results/features_'+kind+'.npz');X=f[str(layer)][:len(corp)]
 for dd in sorted(set(dom)):
  ix=tr&(dom!=dd);d=X[ix&(y==1)].mean(0)-X[ix&(y==0)].mean(0);test=te&(dom==dd)
  results.append({'model':kind,'heldout_domain':dd,'train_pairs':int(ix.sum()/2),'test_pairs':int(test.sum()/2),'auc':roc_auc_score(y[test],X[test]@d)})
pd.DataFrame(results).to_csv('results/cross_domain.csv',index=False)
# Counterfactual strength check: residual hook modifies exactly the requested vector.
tok,m=load();vec=np.load('results/directions.npz');x=tok('A short test paragraph about the weather.',return_tensors='pt').to('cuda');captured={}
def capture(_,inp,out):captured['before']=out[0].detach().clone()
h=m.model.layers[layer-1].register_forward_hook(capture)
with torch.inference_mode():o=m(**x,output_hidden_states=True)
h.remove();hidden=o.hidden_states[layer];assert torch.equal(hidden,captured['before'])
v=torch.tensor(vec['auth'],device='cuda',dtype=m.dtype)
def alter(_,inp,out):
 z=out[0]+v;captured['after']=z.detach().clone();return (z,)+out[1:]
h=m.model.layers[layer-1].register_forward_hook(alter)
with torch.inference_mode():o2=m(**x,output_hidden_states=True)
h.remove();err=(o2.hidden_states[layer].float()-hidden.float()-v.float()).abs().max().item()
report={'layer_hook_matches_hidden_states':True,'max_bfloat16_addition_roundoff':err,'requested_vector_norm':float(v.float().norm()),'observed_perturbation_norm_mean':float((o2.hidden_states[layer].float()-hidden.float()).norm(dim=-1).mean())}
json.dump(report,open('results/hook_validation.json','w'),indent=2)
gen=readj('results/generations.jsonl')
gf,gn=extract(tok,m,[norm(r['text']) for r in gen]);gs=gf[str(layer)]@vec['auth']/np.linalg.norm(vec['auth']);writej('results/generation_readout.jsonl',[{'condition':r['condition'],'id':r['id'],'score':float(s)} for r,s in zip(gen,gs)])
pd.DataFrame(readj('results/generation_readout.jsonl')).groupby('condition').score.agg(['mean','std']).to_csv('results/generation_readout_summary.csv')
del m;torch.cuda.empty_cache()
# Frozen detector applied to exactly 64 Qwen tokens, if both outputs are long enough.
from transformers import AutoModelForSequenceClassification
qt=tok;name='openai-community/roberta-base-openai-detector';dt=AutoTokenizer.from_pretrained(name,revision=REVISIONS[name]);m=AutoModelForSequenceClassification.from_pretrained(name,revision=REVISIONS[name],use_safetensors=True).cuda().eval();fake=[int(k) for k,v in m.config.id2label.items() if v.lower()=='fake'][0]
gen=readj('results/generations.jsonl');rr=[]
for r in gen:
 ids=qt.encode(r['text'],add_special_tokens=False)
 if len(ids)>=64:rr.append({'condition':r['condition'],'id':r['id'],'text':qt.decode(ids[:64])})
out=[]
with torch.inference_mode():
 for i in range(0,len(rr),32):
  rows=rr[i:i+32];x=dt([r['text'] for r in rows],padding=True,return_tensors='pt').to('cuda');sc=m(**x).logits.softmax(-1)[:,fake].cpu().tolist()
  out.extend([{**r,'score':s} for r,s in zip(rows,sc)])
writej('results/fixed64_scores.jsonl',out);df=pd.DataFrame(out);base=df[df.condition=='baseline'].set_index('id');res=[];rng=np.random.default_rng(765)
for cond,g in df.groupby('condition',sort=False):
 g=g.set_index('id');ids=g.index.intersection(base.index);delta=(g.loc[ids].score-base.loc[ids].score).to_numpy();boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1);lo,hi=np.quantile(boot,[.025,.975]);res.append({'condition':cond,'n':len(ids),'delta':delta.mean(),'lo':lo,'hi':hi})
pd.DataFrame(res).to_csv('results/fixed64_summary.csv',index=False)
print('supplement complete',report,flush=True)
