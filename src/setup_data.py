import os,json,requests,pandas as pd
from pathlib import Path
os.environ['HF_HOME']=str(Path('models/hf').resolve())
from huggingface_hub import hf_hub_download
from common import REVISIONS
Path('results').mkdir(exist_ok=True)
json.dump({k:{'type':'dataset' if k in ['Hello-SimpleAI/HC3','browndw/human-ai-parallel-corpus'] else 'model','revision':v} for k,v in REVISIONS.items()},open('results/revisions.json','w'),indent=2)
for name in ['all.jsonl']:
 p=hf_hub_download('Hello-SimpleAI/HC3',name,revision='4d0ff18143b5a7e1b1e79beb540c04549d1e59d3',repo_type='dataset',local_dir='data/hc3');print(p,flush=True)
for name in ['human-chunk-2','gpt-4o-mini-2024-07-18','llama-3-8B-Instruct']:
 p=hf_hub_download('browndw/human-ai-parallel-corpus','text_data/hape-text_'+name+'.parquet',revision='b514ff64988d9e322fd81c5d70d69a38e78491f5',repo_type='dataset',local_dir='data/hape')
 d=pd.read_parquet(p); print(name,d.shape,list(d.columns),flush=True)
