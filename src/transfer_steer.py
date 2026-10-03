"""Exploratory cross-corpus causal replication, disjoint from HAP-E evaluation documents."""
from common import *
import pandas as pd
from sklearn.metrics import roc_auc_score
seed();tok,m=load();frames={}
for key,file in [('human','human-chunk-2'),('gpt4mini','gpt-4o-mini-2024-07-18'),('llama8i','llama-3-8B-Instruct')]:
 df=pd.read_parquet('data/hape/text_data/hape-text_'+file+'.parquet');frames[key]={r.doc_id.split('@')[0]:r.text for r in df.itertuples()}
test=readj('data/ood.jsonl');excluded={r['id'] for r in test};ids=sorted(set.intersection(*(set(f) for f in frames.values()))-excluded);random.Random(907).shuffle(ids);count={};rows=[]
for id in ids:
 dom=id.split('_')[0]
 if count.get(dom,0)>=30:continue
 toks={key:tok.encode(norm(f[id]),add_special_tokens=False) for key,f in frames.items()}
 if min(map(len,toks.values()))<96:continue
 for gen,tt in toks.items():rows.append({'id':id,'domain':dom,'generator':gen,'label':int(gen!='human'),'text':tok.decode(tt[:96])})
 count[dom]=count.get(dom,0)+1
assert len(rows)==540 and not({r['id'] for r in rows}&excluded)
writej('data/hape_fit.jsonl',rows);writej('results/hape_fit_manifest.jsonl',[{k:v for k,v in r.items() if k!='text'} for r in rows]);f,n=extract(tok,m,[r['text'] for r in rows]);np.savez_compressed('results/hape_fit_features.npz',**f,nll=n)
info=json.load(open('results/direction_info.json'));l=info['layer'];X=f[str(l)];y=np.array([r['label'] for r in rows]);raw=X[y==1].mean(0)-X[y==0].mean(0);d0=np.load('results/directions.npz')['auth'];d=raw/np.linalg.norm(raw)*np.linalg.norm(d0);cos=float(raw@d0/np.linalg.norm(raw)/np.linalg.norm(d0))
np.savez_compressed('results/hape_direction.npz',raw=raw,equal_norm=d)
corp=readj('data/corpus.jsonl');ff=np.load('results/features_instruct.npz')[str(l)];cr=[]
for name,ind in [('HC3_test',[i for i,r in enumerate(corp) if r['split']=='test']),('HAPE_gpt4mini',[len(corp)+i for i,r in enumerate(test) if r['generator'] in ['human','gpt4mini']]),('HAPE_llama8i',[len(corp)+i for i,r in enumerate(test) if r['generator'] in ['human','llama8i']])]:
 allr=corp+test;cr.append({'dataset':name,'auc':roc_auc_score([allr[i]['label'] for i in ind],ff[ind]@raw)})
json.dump({'cosine_with_hc3':cos,'raw_norm':float(np.linalg.norm(raw)),'equal_norm':float(np.linalg.norm(d)),'train_documents':180,'train_texts':540,'readout':cr},open('results/hape_direction_info.json','w'),indent=2)
qs=readj('results/generation_questions.jsonl');dv=torch.tensor(d,dtype=m.dtype,device='cuda');out=[]
for alpha in [-1,1]:
 def hook(module,inputs,output):return (output[0]+alpha*dv,)+output[1:]
 h=m.model.layers[l-1].register_forward_hook(hook);ans=generate(tok,m,[prompt(tok,q['question']) for q in qs]);h.remove()
 out.extend([{**r,**a,'condition':'hape_'+str(alpha),'alpha':alpha,'vector':'hape'} for r,a in zip(qs,ans)]);print('HAPE steering',alpha,flush=True)
writej('results/transfer_generations.jsonl',out)
