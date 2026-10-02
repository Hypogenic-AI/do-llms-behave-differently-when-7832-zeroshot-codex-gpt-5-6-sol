#!/usr/bin/env python3
"""Deduplicate JSONL records by experiment, item, and condition."""
import json,sys
source,dest=sys.argv[1:3];rows={};bad=0
for line in open(source):
 try:x=json.loads(line)
 except json.JSONDecodeError:bad+=1;continue
 rows[(x.get('experiment'),x.get('item_id'),x.get('condition'))]=x
with open(dest,'w') as f:
 for k in sorted(rows):f.write(json.dumps(rows[k],ensure_ascii=False)+'\n')
print(json.dumps({'unique':len(rows),'bad':bad}))
