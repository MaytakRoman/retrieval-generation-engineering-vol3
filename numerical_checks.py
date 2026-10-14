"""Independent arithmetic controls for worked examples in the monograph."""
import json,math
from pathlib import Path

values={
 'bm25_tf':{str(tf):tf*2.2/(tf+1.2) for tf in (1,2,4,8)},
 'rrf_advanced':{str(c):{'A':1/(c+1),'B':2/(c+5),'V':1/(c+2)+1/(c+30)} for c in (1,10,60)},
 'context_required_tokens':80+70+40,
 'context_invalid_tokens':80+70+50,
 'memory_one_bytes':250000*(768*4+8+128),
 'memory_two_bytes':2*250000*(768*4+8+128),
 'precision4':2/4,'recall4':2/3,
 'ap4':(1+2/3)/3,
 'dcg3':7+3/2,
 'idcg3':7+3/math.log2(3),
 'macro_recall':(1+1/9)/2,
 'micro_recall':2/10,
 'parallel_ms':max(120,200)+80,
 'serial_ms':120+200+80}
values['ndcg3']=values['dcg3']/values['idcg3']
Path('results').mkdir(exist_ok=True)
Path('results/numerical_checks.json').write_text(json.dumps(values,ensure_ascii=False,indent=2))
assert abs(values['ndcg3']-.95583058934618)<1e-10
assert values['context_required_tokens']==190
assert values['memory_two_bytes']==1604000000
assert values['parallel_ms']==280
print(json.dumps(values,ensure_ascii=False,indent=2))
