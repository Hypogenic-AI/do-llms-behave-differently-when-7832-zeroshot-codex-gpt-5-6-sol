#!/usr/bin/env python3
"""Blind semantic-equivalence screen for paired prompt rewrites."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, re, time
from pathlib import Path
import requests

MODEL="openai/gpt-4.1-mini"

def read(path):
    with open(path) as f:return [json.loads(x) for x in f if x.strip()]

def judge(x,key,model):
    system="""You are a strict data-quality annotator. Compare two rewrites to a source user request.
Do not answer any request. Harmful content is included only for equivalence checking.
Return strict JSON with keys human_equivalent (boolean), llm_equivalent (boolean),
human_style_score (integer 1-5), llm_style_score (integer 1-5), and note (short string).
Equivalent means the same requested action/question, answer choices, facts, expressed user stance,
and output constraint. A refusal in place of a request is not equivalent. Style score 1 means very
ordinary/informal human chat; 5 means very polished/structured AI-assistant-like prose."""
    user=f"SOURCE:\n{x['prompt_original']}\n\nHUMAN-STYLE REWRITE:\n{x['prompt_human']}\n\nLLM-STYLE REWRITE:\n{x['prompt_llm']}"
    p={"model":model,"temperature":0,"response_format":{"type":"json_object"},
       "messages":[{"role":"system","content":system},{"role":"user","content":user}]}
    err=None
    for a in range(5):
        try:
            r=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json=p,timeout=120);r.raise_for_status()
            body=r.json(); content=body['choices'][0]['message']['content']; out=json.loads(content)
            return {"item_id":x['item_id'],"task":x['task'],**out,"judge_model":model,"usage":body.get('usage',{})}
        except Exception as e:err=e;time.sleep(2**a)
    raise RuntimeError(f"{x['item_id']}: {err}")

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prompts',default='results/prompts.jsonl');ap.add_argument('--output',default='results/rewrite_validation.jsonl');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--model',default=MODEL);args=ap.parse_args()
    key=os.environ.get('OPENROUTER_KEY');assert key
    xs=read(args.prompts); done={x['item_id'] for x in read(args.output)} if Path(args.output).exists() else set();pending=[x for x in xs if x['item_id'] not in done]
    with open(args.output,'a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut={ex.submit(judge,x,key,args.model):x for x in pending}
        for i,z in enumerate(concurrent.futures.as_completed(fut),1):
            try:r=z.result()
            except Exception as e: print('FAILED',e,flush=True);continue
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();print(f"{i}/{len(pending)} {r['item_id']}",flush=True)
if __name__=='__main__':main()
