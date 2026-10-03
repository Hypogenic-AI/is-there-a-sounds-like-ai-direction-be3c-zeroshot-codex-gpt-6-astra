"""Late external audit using the original rubric after daily API quota reset."""
from common import readj,writej
from judge import request,SYSTEM,MODEL
from pathlib import Path
import json,random,concurrent.futures
rows=readj('data/calibration.jsonl')
for file in ['results/generations.jsonl','results/transfer_generations.jsonl']:
 rows.extend([{**r,'key':r['condition']+'__'+r['id']} for r in readj(file)])
rows+=readj('results/quality_audit_inputs.jsonl');path='results/claude_judge_raw.jsonl';done={r['key'] for r in readj(path) if 'ratings' in r} if Path(path).exists() else set();rows=[r for r in rows if r['key'] not in done];random.Random(643).shuffle(rows)
json.dump({'model':MODEL,'rubric':SYSTEM,'temperature':0,'max_tokens':180,'timing':'late audit after UTC-day quota reset; all interventions fixed'},open('results/claude_judge_info.json','w'),indent=2)
print('pending',len(rows),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex,open(path,'a') as f:
 for i,r in enumerate(ex.map(request,rows)):
  f.write(json.dumps(r)+'\n');f.flush()
  if i%25==0:print('judged',i+1,'/',len(rows),'error' if 'error' in r else 'ok',flush=True)
