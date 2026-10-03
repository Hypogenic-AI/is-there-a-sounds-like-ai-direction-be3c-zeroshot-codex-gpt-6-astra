"""Supplemental public detector. Architecture follows the model card (MIT)."""
from common import *
from transformers import AutoConfig,AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
NAME='desklib/ai-text-detector-v1.01';REV='5fdea974cd4287c61674951ec78803aa274e2fb7'
class Detector(torch.nn.Module):
 def __init__(self):
  super().__init__();cfg=AutoConfig.from_pretrained(NAME,revision=REV);self.model=AutoModel.from_config(cfg);self.classifier=torch.nn.Linear(cfg.hidden_size,1)
 def forward(self,ids,mask):
  hidden=self.model(input_ids=ids,attention_mask=mask).last_hidden_state
  pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
  return self.classifier(pooled).squeeze(-1)
tok=AutoTokenizer.from_pretrained(NAME,revision=REV);m=Detector();status=m.load_state_dict(load_file(hf_hub_download(NAME,'model.safetensors',revision=REV)),strict=True);print(status,flush=True);m=m.cuda().eval()
rows=readj('data/calibration.jsonl')
for file in ['results/generations.jsonl','results/transfer_generations.jsonl']:
 rows.extend([{**r,'key':r['condition']+'__'+r['id']} for r in readj(file)])
qt=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',revision=REVISIONS['Qwen/Qwen2.5-1.5B-Instruct']);items=[]
for r in rows:
 items.append({'key':r['key'],'mode':'full','text':r['text']})
 if not r['key'].startswith('cal_'):
  tt=qt.encode(r['text'],add_special_tokens=False)
  if len(tt)>=64:items.append({'key':r['key'],'mode':'fixed64','text':qt.decode(tt[:64])})
out=[]
with torch.inference_mode():
 for i in range(0,len(items),16):
  batch=items[i:i+16];x=tok([r['text'] for r in batch],padding=True,truncation=True,max_length=768,return_tensors='pt').to('cuda');logits=m(x.input_ids,x.attention_mask).cpu();s=logits.sigmoid().tolist();out.extend([{'key':r['key'],'mode':r['mode'],'score':float(v),'logit':float(z)} for r,v,z in zip(batch,s,logits.tolist())])
  if i%320==0:print('modern detector',i,'/',len(items),flush=True)
writej('results/modern_detector_scores.jsonl',out);json.dump({'model':NAME,'revision':REV,'max_length':768,'dtype':'float32','strict_weight_loading':str(status)},open('results/modern_detector_info.json','w'),indent=2)
