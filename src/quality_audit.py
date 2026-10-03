"""Known-corruption diagnostic for the automated quality rubric, not natural-text data."""
from common import *
import pandas as pd
base=[r for r in readj('results/generations.jsonl') if r['condition']=='baseline'];selected=[]
for dom in sorted(set(r['domain'] for r in base)):selected.extend([r for r in base if r['domain']==dom][:5])
rng=random.Random(927);audit=[]
for i,r in enumerate(selected):
 words=r['text'].split();rng.shuffle(words)
 variants={'shuffled':' '.join(words),'unrelated':selected[(i+5)%len(selected)]['text'],'repeated':(' '.join(r['text'].split()[:12])+' ')*10}
 for kind,txt in variants.items():audit.append({'key':'quality_'+kind+'__'+r['id'],'id':r['id'],'condition':kind,'question':r['question'],'text':txt})
writej('results/quality_audit_inputs.jsonl',audit)
if Path('results/judge_raw.jsonl').exists():
 j={r['key']:r['ratings'] for r in readj('results/judge_raw.jsonl') if 'ratings' in r}
 if all(r['key'] in j for r in audit):
  rows=[{**r,**j[r['key']]} for r in audit];rows +=[{**r,**j['baseline__'+r['id']],'condition':'original'} for r in selected];df=pd.DataFrame(rows);df.to_csv('results/quality_audit_scores.csv',index=False);df.groupby('condition')[['coherence','adequacy','ai_style']].mean().to_csv('results/quality_audit_summary.csv');print(df.groupby('condition')[['coherence','adequacy','ai_style']].mean())
