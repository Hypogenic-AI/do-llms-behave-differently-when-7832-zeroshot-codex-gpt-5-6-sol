#!/usr/bin/env python3
"""Recover complete, screened, unique records after concurrent writers were detected."""
import json,sys
prompts=sys.argv[1] if len(sys.argv)>1 else 'results/prompts.jsonl'
validation=sys.argv[2] if len(sys.argv)>2 else 'results/rewrite_validation.jsonl'
source=sys.argv[3] if len(sys.argv)>3 else 'results/local_model_outputs_interleaved.jsonl'
dest=sys.argv[4] if len(sys.argv)>4 else 'results/local_model_outputs.jsonl'
def read(path,skip_bad=False):
 rows=[]
 for line in open(path):
  try: rows.append(json.loads(line))
  except json.JSONDecodeError:
   if not skip_bad:raise
 return rows
valid={x['item_id'] for x in read(validation) if x.get('human_equivalent') is True and x.get('llm_equivalent') is True}
unique={};bad=0
for line in open(source):
 try:x=json.loads(line)
 except json.JSONDecodeError:bad+=1;continue
 if x.get('item_id') not in valid:continue
 key=(x.get('experiment'),x.get('item_id'),x.get('condition'))
 unique.setdefault(key,x)
with open(dest,'w') as f:
 for key in sorted(unique):f.write(json.dumps(unique[key],ensure_ascii=False)+'\n')
print(json.dumps({'recovered':len(unique),'malformed_lines_skipped':bad,'valid_items':len(valid)}))
