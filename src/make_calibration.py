"""Reconstruct calibration inputs without loading an evaluator."""
from common import readj,writej
corp=readj('data/corpus.jsonl');ood=readj('data/ood.jsonl');rows=[]
for dom in sorted({r['domain'] for r in corp}):
 for r in [r for r in corp if r['split']=='test' and r['domain']==dom][:40]:rows.append({**r,'key':'cal_hc3_'+r['id']+'_'+str(r['label'])})
for dom in sorted({r['domain'] for r in ood}):
 for r in [r for r in ood if r['domain']==dom][:15]:rows.append({**r,'key':'cal_hape_'+r['id']+'_'+r['generator']})
writej('data/calibration.jsonl',rows);print('Reconstructed',len(rows),'calibration texts')
