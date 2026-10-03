from common import *
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
import pandas as pd
rows=[]
for r in readj('data/calibration.jsonl'):rows.append(r)
for r in readj('results/generations.jsonl'):rows.append({**r,'key':r['condition']+'__'+r['id']})
if Path('results/transfer_generations.jsonl').exists():
 for r in readj('results/transfer_generations.jsonl'):rows.append({**r,'key':r['condition']+'__'+r['id']})
name='openai-community/roberta-base-openai-detector';tok=AutoTokenizer.from_pretrained(name,revision=REVISIONS[name]);m=AutoModelForSequenceClassification.from_pretrained(name,revision=REVISIONS[name],use_safetensors=True).cuda().eval();print(m.config.id2label,flush=True)
fake=[int(k) for k,v in m.config.id2label.items() if v.lower()=='fake'][0];out=[]
with torch.inference_mode():
 for i in range(0,len(rows),32):
  rr=rows[i:i+32];x=tok([r['text'] for r in rr],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda');sc=m(**x).logits.softmax(-1)[:,fake].cpu().tolist()
  for r,s in zip(rr,sc):out.append({'key':r['key'],'detector':s})
writej('results/detector_scores.jsonl',out)
json.dump({'model':name,'revision':m.config._commit_hash,'id2label':m.config.id2label},open('results/detector_info.json','w'),indent=2)
del m;torch.cuda.empty_cache()
tok,m=load('base')
# Independent base-model likelihood on first 96 tokens, with common extraction cap.
f,n=extract(tok,m,[r['text'] for r in rows]);writej('results/generation_nll.jsonl',[{'key':r['key'],'nll':float(v)} for r,v in zip(rows,n)])
