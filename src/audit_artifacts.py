"""Audit complete deliverables without exposing credential contents."""
from pathlib import Path
import json,hashlib,os,collections
import pymupdf as fitz
counts={name:sum(1 for _ in open('results/'+name+'.jsonl')) for name in ['generations','control_generations','transfer_generations','judge_raw','detector_scores','modern_detector_scores','claude_judge_raw']}
assert counts['generations']==1140 and counts['control_generations']==160 and counts['transfer_generations']==120
json.dump(counts,open('results/final_counts.json','w'),indent=2)
api=[json.loads(x) for x in open('results/claude_judge_raw.jsonl')];api=[x for x in api if 'ratings' in x];assert len({x['key'] for x in api})==1570
usage={'saved_rating_responses':len(api),'model_ids':sorted({x['raw']['model'] for x in api}),'prompt_tokens':sum(x['raw']['usage']['prompt_tokens'] for x in api),'completion_tokens':sum(x['raw']['usage']['completion_tokens'] for x in api),'provider_reported_cost_usd':sum(x['raw']['usage'].get('cost',0) for x in api),'finish_reasons':dict(collections.Counter(x['raw']['choices'][0]['finish_reason'] for x in api)),'note':'Usage of saved rating responses only; excludes quota probes and any unlogged request retries. All used rating objects passed schema/range checks; trailing explanations may be truncated.'}
json.dump(usage,open('results/api_usage.json','w'),indent=2)
root=Path('.');files=[p for base in ['src','results','paper_draft'] for p in Path(base).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
for name in ['OPENROUTER_KEY','OPENAI_API_KEY','HF_TOKEN']:
 secret=os.environ.get(name,'')
 if not secret:continue
 for p in files+[Path('README.md')]:
  if secret.encode() in p.read_bytes():raise RuntimeError('Credential found in '+str(p))
pdf=fitz.open('paper_draft/main.pdf');text='\n'.join(p.get_text() for p in pdf)
assert len(pdf)>=5 and len(text)>10000
assert '??' not in text
Path('results/paper_extracted.txt').write_text(text)
report={'pages':len(pdf),'characters':len(text),'credential_scan':'passed','files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.suffix in ['.py','.json','.jsonl','.csv','.npz','.tex','.pdf'] and p.name!='artifact_audit.json'}}
json.dump(report,open('results/artifact_audit.json','w'),indent=2)
for i in [0, min(4,len(pdf)-1)]:pdf[i].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save('data/paper_preview_'+str(i+1)+'.png')
print('Audit passed:',len(pdf),'PDF pages;',len(report['files']),'hashed artifacts')
