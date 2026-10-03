from common import *
import pandas as pd
from huggingface_hub import hf_hub_download
seed();tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
rows=readj('data/hc3/all.jsonl')
for i,r in enumerate(rows):r['id']=str(i)
random.shuffle(rows);seen=set();seen_text=set();groups={}
for r in rows:
 q=norm(r['question'])
 if q.lower() in seen:continue
 seen.add(q.lower());pair=[]
 for key in ['human_answers','chatgpt_answers']:
  good=[norm(s) for s in r[key] if len(tok.encode(norm(s),add_special_tokens=False))>=96]
  if not good:break
  s=good[0];ids=tok.encode(s,add_special_tokens=False)
  pair.append({'text':tok.decode(ids[:96]),'original_tokens':len(ids)})
 if len(pair)!=2:continue
 if any(p['text'] in seen_text for p in pair) or pair[0]['text']==pair[1]['text']:continue
 seen_text.update(p['text'] for p in pair)
 groups.setdefault(r['source'],[]).append({'id':str(r['id']),'question':q,'domain':r['source'],'pair':pair})
print('eligible', {k:len(v) for k,v in groups.items()},flush=True)
out=[];manifest=[]
for dom,rr in sorted(groups.items()):
 if dom=='open_qa':continue
 assert len(rr)>=200
 for split,a,b in [('train',0,125),('val',125,150),('test',150,200)]:
  for r in rr[a:b]:
   manifest.append({'dataset':'HC3','id':r['id'],'domain':dom,'split':split})
   for y,p in enumerate(r['pair']):out.append({**p,'id':r['id'],'question':r['question'],'domain':dom,'split':split,'label':y,'dataset':'HC3'})
writej('data/corpus.jsonl',out);writej('results/split_manifest.jsonl',manifest)
p=hf_hub_download('browndw/human-ai-parallel-corpus','text_data/hape-text_human-chunk-2.parquet',revision='b514ff64988d9e322fd81c5d70d69a38e78491f5',repo_type='dataset',local_dir='data/hape')
frames={}
for key,file in [('human','human-chunk-2'),('gpt4mini','gpt-4o-mini-2024-07-18'),('llama8i','llama-3-8B-Instruct')]:
 d=pd.read_parquet('data/hape/text_data/hape-text_'+file+'.parquet')
 frames[key]={r.doc_id.split('@')[0]:r.text for r in d.itertuples()}
ids=sorted(set.intersection(*(set(f) for f in frames.values())))
groups={}
for id in ids:
 if all(len(tok.encode(norm(f[id]),add_special_tokens=False))>=96 for f in frames.values()):groups.setdefault(id.split('_')[0],[]).append(id)
ood=[]
for dom,ids in sorted(groups.items()):
 random.shuffle(ids)
 for id in ids[:30]:
  for gen,dd in frames.items():
   s=norm(dd[id]);tt=tok.encode(s,add_special_tokens=False)
   ood.append({'dataset':'HAPE','id':id,'domain':dom,'generator':gen,'label':int(gen!='human'),'text':tok.decode(tt[:96]),'original_tokens':len(tt)})
writej('data/ood.jsonl',ood)
writej('results/ood_manifest.jsonl',[{k:v for k,v in r.items() if k not in ['text']} for r in ood])
print('corpus',len(out),'ood',len(ood),flush=True)
