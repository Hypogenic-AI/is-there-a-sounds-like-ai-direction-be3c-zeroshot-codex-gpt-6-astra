from pathlib import Path
import pandas as pd,json
P=Path('paper_draft/tables');P.mkdir(exist_ok=True)
def table(file,columns,header,rows):
 text='\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'
 text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n';(P/file).write_text(text)
def interval(x,lo,hi,d=3):return f'{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'
r=pd.read_csv('results/readout_ci.csv');rows=[]
for model in ['base','instruct']:
 for method in ['mean','logistic']:
  row=[model.capitalize(), 'Mean difference' if method=='mean' else 'Logistic']
  for ds in ['HC3_test','HAPE_gpt4mini','HAPE_llama8i']:
   x=r[(r.model==model)&(r.method==method)&(r.dataset==ds)].iloc[0];row.append(interval(x.auc,x.lo,x.hi))
  rows.append(row)
table('readout.tex','llccc','Model & Readout & HC3 & HAP-E GPT-4o-mini & HAP-E Llama-3-8B-I',rows)
s=pd.read_csv('results/summary.csv');names={'baseline':'Baseline','auth_-2':r'Authorship $-2$','auth_-1':r'Authorship $-1$','auth_-0.5':r'Authorship $-0.5$','auth_0.5':r'Authorship $+0.5$','auth_1':r'Authorship $+1$','auth_2':r'Authorship $+2$','prompt_human':'Human-style prompt','ablate':'Centered ablation','orth_-1':r'Orthogonalized $-1$','orth_1':r'Orthogonalized $+1$','formal_-1':r'Formality $-1$','formal_1':r'Formality $+1$','persona_-1':r'Persona proxy $-1$','persona_1':r'Persona proxy $+1$','random101_-1':r'Random 101 $-1$','random101_1':r'Random 101 $+1$','random202_-1':r'Random 202 $-1$','random202_1':r'Random 202 $+1$'}
rows=[]
for x in s.itertuples():rows.append([names[x.condition],f'{x.detector:.3f}',f'{x.ai_style:.1f}',f'{x.coherence:.2f}',f'{x.adequacy:.2f}',f'{100*x.quality_pass:.0f}\\%',f'{x.tokens:.1f}'])
table('all_conditions.tex','lrrrrrr','Condition & Detector & AI style & Coherence & Adequacy & Pass & Tokens',rows)
rows=[]
for x in s.itertuples():
 if x.condition=='baseline':continue
 rows.append([names[x.condition],interval(x.detector_delta,x.detector_lo,x.detector_hi),interval(x.ai_style_delta,x.ai_style_lo,x.ai_style_hi,1),str(x.matched_n),interval(x.detector_matched_delta,x.detector_matched_lo,x.detector_matched_hi),interval(x.ai_style_matched_delta,x.ai_style_matched_lo,x.ai_style_matched_hi,1)])
table('effects.tex','lccrcc',r'Condition & $\Delta$ detector & $\Delta$ AI style & $n_Q$ & $\Delta$ detector ($Q$) & $\Delta$ AI style ($Q$)',rows)
c=pd.concat([pd.read_csv('results/calibration_summary.csv'),pd.read_csv('results/modern_calibration.csv'),pd.read_csv('results/claude_calibration.csv')],ignore_index=True);rows=[]
for x in c.itertuples():rows.append([x.dataset,{'detector':'RoBERTa','ai_style':'Mistral judge','Desklib':'Desklib','Claude':'Claude (late)'}[x.measure],x.n,interval(x.auc,x.lo,x.hi)])
table('calibration.tex','llrc','Dataset & Measure & Texts & AUROC [95\\% CI]',rows)
o=pd.read_csv('results/orth_readout.csv');rows=[]
for direction in ['auth','formal','persona','orth']:
 row=[{'auth':'Authorship','formal':'Formality','persona':'Persona proxy','orth':'Orthogonalized'}[direction]]
 for ds in ['HC3_test','HAPE_gpt4mini','HAPE_llama8i']:row.append(f'{o[(o.direction==direction)&(o.dataset==ds)].iloc[0].auc:.3f}')
 rows.append(row)
table('orth.tex','lccc','Direction & HC3 & GPT-4o-mini & Llama-3-8B-I',rows)
if Path('results/fixed64_summary.csv').exists():
 ff=pd.read_csv('results/fixed64_summary.csv');rows=[]
 for x in ff.itertuples():rows.append([names[x.condition],x.n,interval(x.delta,x.lo,x.hi)])
 table('fixed64.tex','lrc','Condition & Pairs & Detector change [95\\% CI]',rows)
if Path('results/cross_domain.csv').exists():
 ff=pd.read_csv('results/cross_domain.csv');rows=[]
 for dom in ff.heldout_domain.unique():rows.append([dom.replace('_',r'\_')]+[f'{ff[(ff.heldout_domain==dom)&(ff.model==m)].iloc[0].auc:.3f}' for m in ['base','instruct']])
 table('cross_domain.tex','lcc','Held-out domain & Base & Instruct',rows)
info=json.load(open('results/direction_info.json'));rows=[[k.replace('_',r'\_'),f'{v:.3f}'] for k,v in info['cosines'].items()];table('cosines.tex','lr','Nuisance direction & Cosine with authorship',rows)
names.update({'hape_-1':r'HAP-E $-1$','hape_1':r'HAP-E $+1$'})
mm=pd.read_csv('results/modern_summary.csv');rows=[]
for x in mm.itertuples():
 rows.append([names[x.condition],f'{x.mean:.3f}',interval(x.delta,x.lo,x.hi),interval(x.logit_delta,x.logit_lo,x.logit_hi,2),f'{100*x.detection_rate:.1f}\\%',str(x.fixed_n),interval(x.fixed_logit_delta,x.fixed_logit_lo,x.fixed_logit_hi,2)])
table('modern.tex','lrccrrc','Condition & Score & Score change & Logit change & Detected & $n_{64}$ & Logit change (64 tokens)',rows)
cc=pd.read_csv('results/claude_summary.csv');rows=[]
for x in cc.itertuples():
 rows.append([names[x.condition],f'{x.ai_style:.1f}',interval(x.ai_style_delta,x.ai_style_lo,x.ai_style_hi,2),f'{x.adequacy:.2f}',f'{100*x.quality_pass:.1f}\\%',x.matched_n,interval(x.ai_style_matched_delta,x.ai_style_matched_lo,x.ai_style_matched_hi,2)])
table('claude.tex','lrcdrrc'.replace('d','r'),r'Condition & AI style & $\Delta$ AI style & Adequacy & Pass & $n_Q$ & $\Delta$ AI style ($Q$)',rows)
print('Generated result tables')
