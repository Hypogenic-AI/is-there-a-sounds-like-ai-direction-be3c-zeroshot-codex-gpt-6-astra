from common import *
from collections import Counter
corp=readj('data/corpus.jsonl');ood=readj('data/ood.jsonl');tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
report={'corpus_rows':len(corp),'ood_rows':len(ood)}
ids={s:{r['id'] for r in corp if r['split']==s} for s in ['train','val','test']}
assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
assert all(Counter(r['id'] for r in corp if r['split']==s).get(id)==2 for s,ii in ids.items() for id in ii)
report['split_pairs']={k:len(v) for k,v in ids.items()}
report['retokenized_lengths']=dict(Counter(len(tok.encode(r['text'],add_special_tokens=False)) for r in corp+ood))
# Exact repeated texts can leak even when questions are disjoint; report rather than silently remove after outcomes.
texts={s:{r['text'] for r in corp if r['split']==s} for s in ids}
report['exact_train_test_text_overlap']=len(texts['train']&texts['test']);report['exact_train_val_text_overlap']=len(texts['train']&texts['val'])
if Path('results/directions.npz').exists():
 d=np.load('results/directions.npz');norms={k:float(np.linalg.norm(d[k])) for k in ['auth','formal','persona','orth','random101','random202']};assert max(norms.values())-min(norms.values())<1e-4
 report['vector_norms']=norms;report['orth_basis_max_cosine']=float(np.abs(d['nuisance_basis'].T@d['orth']/np.linalg.norm(d['orth'])).max());assert report['orth_basis_max_cosine']<1e-5
if Path('results/generations.jsonl').exists():
 g=readj('results/generations.jsonl');c=Counter(r['condition'] for r in g);assert len(c)==19 and set(c.values())=={60};assert len({(r['id'],r['condition']) for r in g})==1140;assert all(r['id'] in ids['test'] for r in g)
 report['generation_counts']=dict(c)
if Path('data/hape_fit.jsonl').exists():
 fit=readj('data/hape_fit.jsonl');report['hape_fit_evaluation_document_overlap']=len({r['id'] for r in fit}&{r['id'] for r in ood});report['hape_fit_evaluation_text_overlap']=len({r['text'] for r in fit}&{r['text'] for r in ood});assert report['hape_fit_evaluation_document_overlap']==0 and report['hape_fit_evaluation_text_overlap']==0
json.dump(report,open('results/validation.json','w'),indent=2);print(report)
