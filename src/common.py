import os,json,re,random
from pathlib import Path
os.environ['HF_HOME']=str(Path('models/hf').resolve())
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, torch
from transformers import AutoTokenizer,AutoModelForCausalLM
REVISIONS={'Qwen/Qwen2.5-1.5B': '8faed761d45a263340a0528343f099c05c9a4323', 'Qwen/Qwen2.5-1.5B-Instruct': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306', 'mistralai/Mistral-7B-Instruct-v0.3': 'c170c708c41dac9275d15a8fff4eca08d52bab71', 'openai-community/roberta-base-openai-detector': '6cba99c003b711c7fe94f8a3aa2be35a792cb6fa', 'Hello-SimpleAI/HC3': '4d0ff18143b5a7e1b1e79beb540c04549d1e59d3', 'browndw/human-ai-parallel-corpus': 'b514ff64988d9e322fd81c5d70d69a38e78491f5'}
REVISIONS['desklib/ai-text-detector-v1.01']='5fdea974cd4287c61674951ec78803aa274e2fb7'
SEED=42
LAYERS=[7,14,21]
torch.set_num_threads(8)
def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
def readj(p):
 return [json.loads(s) for s in open(p)]
def writej(p,rows):
 with open(p,'w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def norm(s):
 s=re.sub(r'\s+',' ',s).strip()
 s=re.sub(r'\s+([,.!?;:])',r'\1',s)
 s=re.sub(r"\s+(['’](?:s|t|re|ve|ll|d|m))\b",r'\1',s)
 s=s.replace(" n't","n't")
 return s
def load(kind='instruct'):
 name='Qwen/Qwen2.5-1.5B'+('-Instruct' if kind=='instruct' else '')
 tok=AutoTokenizer.from_pretrained(name,revision=REVISIONS[name]);tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(name,revision=REVISIONS[name],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 return tok,model
def extract(tok,model,texts,batch=16):
 feats={l:[] for l in LAYERS};nll=[]
 tok.padding_side='right'
 with torch.inference_mode():
  for start in range(0,len(texts),batch):
   x=tok(texts[start:start+batch],padding=True,truncation=True,max_length=96,return_tensors='pt').to('cuda')
   out=model(**x,output_hidden_states=True)
   mask=x.attention_mask.float()
   for l in LAYERS:
    feats[l].append((out.hidden_states[l].float()*mask[...,None]).sum(1).div(mask.sum(1)[:,None]).cpu().numpy())
   loss=torch.nn.functional.cross_entropy(out.logits[:,:-1,:].float().transpose(1,2),x.input_ids[:,1:],reduction='none')
   nll.extend(((loss*mask[:,1:]).sum(1)/mask[:,1:].sum(1)).cpu().tolist())
   if start%320==0:print('extract',start,'/',len(texts),flush=True)
 return {str(l):np.concatenate(v) for l,v in feats.items()},np.array(nll)
def generate(tok,model,prompts,hook=None,batch=12):
 tok.padding_side='left';ans=[]
 with torch.inference_mode():
  for start in range(0,len(prompts),batch):
   xx=tok(prompts[start:start+batch],padding=True,return_tensors='pt').to('cuda')
   yy=model.generate(**xx,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id)
   for ids in yy[:,xx.input_ids.shape[1]:]:
    text=tok.decode(ids,skip_special_tokens=True)
    ans.append({'text':text,'tokens':int((ids!=tok.pad_token_id).sum())})
 return ans
def prompt(tok,q,instruction='Answer the question clearly in one short paragraph of about 80 words.'):
 return tok.apply_chat_template([{'role':'user','content':instruction+'\n\nQuestion: '+q}],tokenize=False,add_generation_prompt=True)
