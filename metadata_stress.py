"""Controlled post-hoc metadata fault injection; ground truth remains unchanged."""
import json
from dataclasses import replace
from pathlib import Path
from reference_rag import Engine,generate,evaluate

docs,qs=generate();test=[q for q in qs if q.asset>=20]
report=[]
for modulus,label in [(4,'25% текущих флагов'),(2,'50% текущих флагов'),(1,'100% текущих флагов')]:
 changed=[];counter=0;altered=0
 for doc in docs:
  if doc.current:
   damaged=counter%modulus==0;counter+=1
   if damaged:doc=replace(doc,current=False);altered+=1
  changed.append(doc)
 engine=Engine(changed);bm,sem=engine.score(test)
 summary,_=evaluate(engine,test,bm,sem,'filtered',10)
 report.append({'condition':label,'changed_records':altered,**summary})
Path('results').mkdir(exist_ok=True)
Path('results/metadata_stress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
