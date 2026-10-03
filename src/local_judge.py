from common import *
from judge import SYSTEM
import argparse,time
p=argparse.ArgumentParser();p.add_argument('--calibration-only',action='store_true');args=p.parse_args()
corp=readj('data/corpus.jsonl');ood=readj('data/ood.jsonl');cal=[]
for dom in sorted(set(r['domain'] for r in corp)):
 for r in [r for r in corp if r['split']=='test' and r['domain']==dom][:40]:cal.append({**r,'key':'cal_hc3_'+r['id']+'_'+str(r['label'])})
for dom in sorted(set(r['domain'] for r in ood)):
 for r in [r for r in ood if r['domain']==dom][:15]:cal.append({**r,'key':'cal_hape_'+r['id']+'_'+r['generator']})
writej('data/calibration.jsonl',cal)
rows=cal
if not args.calibration_only:
 rows +=[{**r,'key':r['condition']+'__'+r['id']} for r in readj('results/generations.jsonl')]
 if Path('results/transfer_generations.jsonl').exists():rows +=[{**r,'key':r['condition']+'__'+r['id']} for r in readj('results/transfer_generations.jsonl')]
if not args.calibration_only and Path('results/quality_audit_inputs.jsonl').exists():rows +=readj('results/quality_audit_inputs.jsonl')
path='results/judge_raw.jsonl';done={r['key'] for r in readj(path) if 'ratings' in r} if Path(path).exists() else set();rows=[r for r in rows if r['key'] not in done];random.Random(812).shuffle(rows)
name='mistralai/Mistral-7B-Instruct-v0.3';tok=AutoTokenizer.from_pretrained(name,revision=REVISIONS[name]);tok.pad_token=tok.eos_token;tok.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(name,revision=REVISIONS[name],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
json.dump({'model':name,'revision':m.config._commit_hash,'rubric':SYSTEM,'temperature':0,'max_new_tokens':150,'batch_size':12},open('results/judge_info.json','w'),indent=2)
print('pending',len(rows),flush=True)
with torch.inference_mode(),open(path,'a') as out:
 for i in range(0,len(rows),12):
  rr=rows[i:i+12];prompts=[]
  for r in rr:
   text=SYSTEM+'\n\n'+json.dumps({'question':r.get('question','[No question: continuation excerpt]'),'sample':r['text']},ensure_ascii=False)+'\n\nReturn the JSON ratings now.'
   prompts.append(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True))
  x=tok(prompts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');y=m.generate(**x,max_new_tokens=150,do_sample=False,pad_token_id=tok.pad_token_id)
  for r,ids in zip(rr,y[:,x.input_ids.shape[1]:]):
   raw=tok.decode(ids,skip_special_tokens=True);record={'key':r['key'],'raw_text':raw}
   try:
    v=json.loads(raw[raw.index('{'):raw.rindex('}')+1]);assert all(isinstance(v[k],(int,float)) for k in ['ai_style','coherence','adequacy','formality']);assert 0<=v['ai_style']<=100 and all(1<=v[k]<=5 for k in ['coherence','adequacy','formality']);record['ratings']=v
   except Exception as e:record['error']=type(e).__name__
   out.write(json.dumps(record)+'\n');out.flush()
  print('judged',min(i+12,len(rows)),'/',len(rows),flush=True)
