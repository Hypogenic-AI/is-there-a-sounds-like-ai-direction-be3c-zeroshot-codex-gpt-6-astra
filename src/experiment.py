from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import argparse,time
p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','controls','steer']);p.add_argument('--kind',default='instruct');args=p.parse_args();seed()
corpus=readj('data/corpus.jsonl');ood=readj('data/ood.jsonl')
if args.stage=='extract':
 tok,model=load(args.kind)
 feats,nll=extract(tok,model,[r['text'] for r in corpus+ood])
 np.savez_compressed('results/features_'+args.kind+'.npz',**feats,nll=nll)
 json.dump({'model':model.config._name_or_path,'revision':model.config._commit_hash,'layers':LAYERS,'dtype':str(model.dtype)},open('results/model_'+args.kind+'.json','w'),indent=2)
elif args.stage=='controls':
 tok,model=load();train=[r for r in corpus if r['split']=='train' and r['label']==0];qs=[]
 for dom in sorted(set(r['domain'] for r in train)):qs.extend([r for r in train if r['domain']==dom][:10])
 styles={'formal':'Answer in a formal academic register, with precise impersonal language. Use one paragraph of about 80 words.', 'casual':'Answer in a casual conversational register, like talking to a friend. Use one paragraph of about 80 words.', 'assistant':'You are a helpful, professional AI assistant. Answer accurately as that assistant in one paragraph of about 80 words.', 'storyteller':'You are a whimsical fictional storyteller. Answer accurately in that character\'s voice in one paragraph of about 80 words.'}
 rows=[]
 for name,instr in styles.items():
  print('control',name,flush=True)
  ans=generate(tok,model,[prompt(tok,r['question'],instr) for r in qs])
  rows.extend([{**a,'style':name,'id':r['id'],'question':r['question']} for a,r in zip(ans,qs)])
 writej('results/control_generations.jsonl',rows)
 f,n=extract(tok,model,[norm(r['text']) for r in rows]);np.savez_compressed('results/control_features.npz',**f,nll=n)
elif args.stage=='steer':
 tok,model=load();f=np.load('results/features_instruct.npz');ctrl=np.load('results/control_features.npz');cr=readj('results/control_generations.jsonl')
 y=np.array([r['label'] for r in corpus]);tr=np.array([r['split']=='train' for r in corpus]);va=np.array([r['split']=='val' for r in corpus])
 ds={l:f[str(l)][:len(corpus)][tr&(y==1)].mean(0)-f[str(l)][:len(corpus)][tr&(y==0)].mean(0) for l in LAYERS}
 auc={l:roc_auc_score(y[va],f[str(l)][:len(corpus)][va]@ds[l]) for l in LAYERS};layer=max(LAYERS,key=lambda l:auc[l]);d=ds[layer];size=np.linalg.norm(d);u=d/size;X=f[str(layer)][:len(corpus)];cf=ctrl[str(layer)]
 means={style:cf[np.array([r['style']==style for r in cr])].mean(0) for style in ['formal','casual','assistant','storyteller']}
 form=means['formal']-means['casual'];persona=means['assistant']-means['storyteller']
 # Train-only residual nuisance covariance axes within each authorship class.
 axes=[form,persona];axis_names=['formality','persona_proxy']
 for name,vals in [('log_original_length',np.log([r['original_tokens'] for r in corpus])),('nll',f['nll'][:len(corpus)])]:
  z=vals[tr].copy();xx=X[tr].copy()
  for label in [0,1]:
   ix=y[tr]==label;z[ix]-=z[ix].mean();xx[ix]-=xx[ix].mean(0)
  v=xx.T@z;axes.append(v);axis_names.append(name)
 doms=sorted(set(r['domain'] for r in corpus));dom=np.array([r['domain'] for r in corpus]);globalmean=X[tr].mean(0)
 for name in doms[:-1]:axes.append(X[tr&(dom==name)].mean(0)-globalmean);axis_names.append(name)
 A=np.stack([v/np.linalg.norm(v) for v in axes],axis=1);Q=np.linalg.qr(A)[0];res=d-Q@(Q.T@d);ret=np.linalg.norm(res)/size;res=res/np.linalg.norm(res)*size
 vecs={'auth':d,'formal':form/np.linalg.norm(form)*size,'persona':persona/np.linalg.norm(persona)*size,'orth':res}
 for rs in [101,202]:
  v=np.random.default_rng(rs).normal(size=len(d));vecs['random'+str(rs)]=v/np.linalg.norm(v)*size
 center=(X[tr&(y==1)].mean(0)+X[tr&(y==0)].mean(0))/2
 np.savez_compressed('results/directions.npz',**vecs,center=center,nuisance_basis=Q)
 info={'layer':layer,'validation_auc':auc,'auth_norm':float(size),'train_mean_residual_norm':float(np.linalg.norm(X[tr],axis=1).mean()),'orth_retained_norm_fraction':float(ret),'cosines':{name:float(u@A[:,i]) for i,name in enumerate(axis_names)}}
 json.dump(info,open('results/direction_info.json','w'),indent=2);print(info,flush=True)
 qs=[]
 for dom in doms:qs.extend([r for r in corpus if r['split']=='test' and r['label']==0 and r['domain']==dom][:15])
 writej('results/generation_questions.jsonl',[{k:r[k] for k in ['id','question','domain']} for r in qs])
 conditions=[('baseline','auth',0)]+[('auth_'+str(a),'auth',a) for a in [-2,-1,-.5,.5,1,2]]
 conditions +=[(v+'_'+str(a),v,a) for v in ['random101','random202','formal','persona','orth'] for a in [-1,1]]
 conditions +=[('ablate','ablate',0),('prompt_human','auth',0)]
 path='results/generations.jsonl';done={r['condition'] for r in readj(path)} if Path(path).exists() else set()
 for name,v,a in conditions:
  if name in done:continue
  handle=None
  if a or v=='ablate':
   dv=torch.tensor(vecs.get(v,d),dtype=model.dtype,device='cuda');ut=torch.tensor(u,dtype=model.dtype,device='cuda');ct=torch.tensor(center,dtype=model.dtype,device='cuda')
   def hook(module,inputs,output):
    h=output[0] if isinstance(output,tuple) else output
    if v=='ablate':new=h-((h-ct)@ut)[...,None]*ut
    else:new=h+a*dv
    return (new,)+output[1:] if isinstance(output,tuple) else new
   handle=model.model.layers[layer-1].register_forward_hook(hook)
  instr='Answer the question clearly in one short paragraph of about 80 words.'
  if name=='prompt_human':instr+=' Write like a human, with natural wording rather than a typical AI-generated answer.'
  start=time.time();ans=generate(tok,model,[prompt(tok,r['question'],instr) for r in qs])
  if handle:handle.remove()
  with open(path,'a') as out:
   for r,an in zip(qs,ans):out.write(json.dumps({**an,'id':r['id'],'question':r['question'],'domain':r['domain'],'condition':name,'alpha':a,'vector':v})+'\n')
  print(name,len(ans),'seconds',round(time.time()-start,1),flush=True)
