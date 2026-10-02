#!/usr/bin/env python3
"""Create statistical summaries, tables, and figures from executed runs."""
from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest,wilcoxon

SEED=7832
def read(p):
    with open(p) as f:return [json.loads(x) for x in f if x.strip()]
def boot_delta(a,b,nboot=10000):
    """Paired b-a mean difference and percentile CI."""
    a=np.asarray(a,float);b=np.asarray(b,float);d=b-a;rng=np.random.default_rng(SEED)
    vals=d[rng.integers(0,len(d),size=(nboot,len(d)))].mean(1)
    return float(d.mean()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))
def paired_test(a,b,binary=False):
    a=np.asarray(a,float);b=np.asarray(b,float);d=b-a
    if binary:
        pos=int((d>0).sum());neg=int((d<0).sum());n=pos+neg
        return float(binomtest(pos,n,.5).pvalue) if n else 1.0
    if np.allclose(d,0):return 1.0
    return float(wilcoxon(d,alternative='two-sided',zero_method='pratt').pvalue)
def contrast(df,experiment,outcome,a,b,label,binary=False,subset=None):
    q=df[df.experiment.eq(experiment)]
    if subset is not None:q=q[subset(q)]
    p=q.pivot(index='item_id',columns='condition',values=outcome).dropna(subset=[a,b])
    d,lo,hi=boot_delta(p[a],p[b])
    return {"contrast":label,"experiment":experiment,"outcome":outcome,"condition_a":a,
            "condition_b":b,"n":len(p),"mean_a":float(p[a].mean()),"mean_b":float(p[b].mean()),
            "delta_b_minus_a":d,"ci_low":lo,"ci_high":hi,"p_value":paired_test(p[a],p[b],binary)}
def main():
    out=read('results/local_model_outputs.jsonl'); prompts={x['item_id']:x for x in read('results/prompts.jsonl')}; judgments=read('results/safety_judgments.jsonl')
    rows=[]
    for x in out:
        base={"experiment":x['experiment'],"item_id":x['item_id'],"condition":x['condition'],"task":x['task']}
        m=prompts[x['item_id']]
        if x['experiment']!='authorship' and x['task'] in ('accuracy','sycophancy') and 'probabilities' in x:
            target=m['label'];base['target_probability']=x['probabilities'][target];base['binary_target']=float(x['prediction']==target)
        if x['experiment']=='authorship':base['ai_style_probability']=x['probabilities']['A'];base['ai_style_choice']=float(x['prediction']=='A')
        rows.append(base)
    df=pd.DataFrame(rows)
    j=pd.DataFrame(judgments)
    if len(j):
        j=j.drop_duplicates(['experiment','item_id','condition'],keep='last')
        j=j[j.item_id.isin(df.item_id.unique())]
        j=j.merge(pd.DataFrame([{"item_id":k,"harmful":v.get('harmful')} for k,v in prompts.items()]),on='item_id',how='left')
        j['task']='safety'
    summaries=[]
    summaries.append(contrast(df,'authorship','ai_style_probability','human','llm','Perceived AI style: LLM vs human'))
    for task,label in [('accuracy','Answer probability'),('sycophancy','Sycophantic-answer probability')]:
        summaries.append(contrast(df,'style','target_probability','human','llm',label,subset=lambda q,t=task:q.task.eq(t)))
        summaries.append(contrast(df,'style','binary_target','human','llm',label+' (choice)',True,subset=lambda q,t=task:q.task.eq(t)))
    for harm,label in [(True,'Harmful refusal'),(False,'Benign over-refusal')]:
        summaries.append(contrast(j,'style','refusal','human','llm',label,True,subset=lambda q,h=harm:q.harmful.eq(h)))
    # Steering: each directed/random vector against exactly the same baseline prompt.
    for cond in ['llm_direction','human_direction','random_positive','random_negative']:
        for task,label in [('accuracy','Answer probability'),('sycophancy','Sycophantic-answer probability')]:
            summaries.append(contrast(df,'steering','target_probability','baseline',cond,f'{label}: {cond}',subset=lambda q,t=task:q.task.eq(t)))
        for harm,label in [(True,'Harmful refusal'),(False,'Benign over-refusal')]:
            summaries.append(contrast(j,'steering','refusal','baseline',cond,f'{label}: {cond}',True,subset=lambda q,h=harm:q.harmful.eq(h)))
    for a,b,name in [('human_direction','llm_direction','directed LLM minus human'),
                     ('random_negative','random_positive','random positive minus negative')]:
        for task,label in [('accuracy','Answer probability'),('sycophancy','Sycophantic-answer probability')]:
            summaries.append(contrast(df,'steering','target_probability',a,b,f'{label}: {name}',subset=lambda q,t=task:q.task.eq(t)))
        for harm,label in [(True,'Harmful refusal'),(False,'Benign over-refusal')]:
            summaries.append(contrast(j,'steering','refusal',a,b,f'{label}: {name}',True,subset=lambda q,h=harm:q.harmful.eq(h)))
    tab=pd.DataFrame(summaries);Path('results/tables').mkdir(parents=True,exist_ok=True);tab.to_csv('results/tables/contrasts.csv',index=False)

    valid_ids=set(df.item_id)
    vals={x['item_id']:x for x in read('results/rewrite_validation.jsonl')}
    surface=[]
    for iid in valid_ids:
        p=prompts[iid]
        for style in ['original','human','llm']:
            s=p[f'prompt_{style}'];surface.append({"item_id":iid,"task":p['task'],"style":style,"words":len(re.findall(r"\b\w+\b",s)),"chars":len(s),"lines":s.count('\n')+1,"judge_style_score":vals.get(iid,{}).get(f'{style}_style_score')})
    surf=pd.DataFrame(surface);surf.to_csv('results/tables/prompt_surface.csv',index=False)
    surface_summary=surf.groupby('style').agg(n=('item_id','size'),words_mean=('words','mean'),words_sd=('words','std'),chars_mean=('chars','mean'),lines_mean=('lines','mean'),judge_style_mean=('judge_style_score','mean')).reset_index();surface_summary.to_csv('results/tables/surface_summary.csv',index=False)
    lens=surf[surf['style'].isin(['human','llm'])].pivot(index='item_id',columns='style',values='words')
    matched=set(lens.index[(lens.max(axis=1)/lens.min(axis=1))<=1.5])
    sensitivity=[]
    for task,label in [('accuracy','Answer probability'),('sycophancy','Sycophantic-answer probability')]:
        sensitivity.append(contrast(df,'style','target_probability','human','llm',label,
            subset=lambda q,t=task:q.task.eq(t)&q.item_id.isin(matched)))
    for harm,label in [(True,'Harmful refusal'),(False,'Benign over-refusal')]:
        sensitivity.append(contrast(j,'style','refusal','human','llm',label,True,
            subset=lambda q,h=harm:q.harmful.eq(h)&q.item_id.isin(matched)))
    pd.DataFrame(sensitivity).to_csv('results/tables/length_matched_contrasts.csv',index=False)

    probe=json.loads(Path('results/probe_metrics.json').read_text());pdat=pd.DataFrame(probe['layers']);cdat=pd.DataFrame(probe['evaluation_awareness_cosines'])
    sns.set_theme(style='whitegrid',context='paper')
    fig,ax=plt.subplots(figsize=(5.4,3.2));ax.plot(pdat.layer,pdat.auc,marker='o',label='Style probe AUC');ax.axhline(.5,color='gray',ls='--',lw=1);ax.set(xlabel='Transformer block',ylabel='Held-out AUC',ylim=(.45,1.03));ax2=ax.twinx();ax2.plot(cdat.layer,cdat.cosine,marker='s',color='#c44e52',label='Cosine with eval-awareness');ax2.axhline(0,color='#c44e52',ls=':',lw=1);ax2.set_ylabel('Direction cosine',color='#c44e52');lines=ax.lines[:1]+ax2.lines[:1];ax.legend(lines,[z.get_label() for z in lines],loc='lower left',frameon=True);fig.tight_layout();fig.savefig('paper_draft/figures/probe.pdf');fig.savefig('paper_draft/figures/probe.png',dpi=180);plt.close(fig)

    forest=tab[tab.contrast.isin(['Answer probability','Sycophantic-answer probability','Harmful refusal','Benign over-refusal'])].copy();forest['pretty']=forest.contrast
    fig,ax=plt.subplots(figsize=(5.4,3.1));y=np.arange(len(forest));ax.errorbar(forest.delta_b_minus_a,y,xerr=[forest.delta_b_minus_a-forest.ci_low,forest.ci_high-forest.delta_b_minus_a],fmt='o',color='#4c72b0',capsize=3);ax.axvline(0,color='black',lw=1);ax.set_yticks(y,forest.pretty);ax.invert_yaxis();ax.set_xlabel('LLM-style minus human-style (paired mean difference)');fig.tight_layout();fig.savefig('paper_draft/figures/style_effects.pdf');fig.savefig('paper_draft/figures/style_effects.png',dpi=180);plt.close(fig)

    steer=tab[(tab.experiment=='steering') & (tab.condition_a=='baseline') & tab.contrast.str.contains('Answer probability|Sycophantic|Harmful refusal',regex=True)].copy();steer['direction']=steer.condition_b.map({'llm_direction':'LLM direction','human_direction':'Human direction','random_positive':'Random +','random_negative':'Random -'});steer['metric']=steer.contrast.str.replace(r': .*','',regex=True)
    fig,ax=plt.subplots(figsize=(6.2,3.7));colors=dict(zip(steer.direction.unique(),sns.color_palette('colorblind',4)));offsets=dict(zip(steer.direction.unique(),[-.24,-.08,.08,.24]));metrics=list(steer.metric.unique())
    for direction,g in steer.groupby('direction'):
        yy=np.array([metrics.index(x) for x in g.metric])+offsets[direction];ax.errorbar(g.delta_b_minus_a,yy,xerr=[g.delta_b_minus_a-g.ci_low,g.ci_high-g.delta_b_minus_a],fmt='o',capsize=2,label=direction,color=colors[direction])
    ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(metrics)),metrics);ax.invert_yaxis();ax.set_xlabel('Steered minus baseline (paired mean difference)');ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig('paper_draft/figures/steering_effects.pdf');fig.savefig('paper_draft/figures/steering_effects.png',dpi=180);plt.close(fig)

    # Machine-readable headline results.
    summary={"n_evaluation":int(df.item_id.nunique()),"n_safety_judgments":len(j),"n_length_matched":len(matched),"probe":probe,"contrasts":summaries,"length_matched_contrasts":sensitivity,"surface":surface_summary.to_dict('records')}
    Path('results/summary.json').write_text(json.dumps(summary,indent=2))
    print(tab.to_string(index=False));print('\nSurface\n',surface_summary.to_string(index=False))
if __name__=='__main__':main()
