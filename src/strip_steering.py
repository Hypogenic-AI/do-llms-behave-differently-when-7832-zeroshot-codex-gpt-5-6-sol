#!/usr/bin/env python3
"""Keep non-steering records before a clean causal rerun."""
import json,sys
source=sys.argv[1];dest=sys.argv[2]
rows=[]
for line in open(source):
 try:x=json.loads(line)
 except json.JSONDecodeError:continue
 if x.get('experiment')!='steering':rows.append(x)
with open(dest,'w') as f:
 for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print(f'kept {len(rows)} non-steering records')
