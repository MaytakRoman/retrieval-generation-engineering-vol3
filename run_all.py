"""Run the book's four modules and independently reconcile saved artifacts."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def check_results():
    out=ROOT/'results';summary=json.loads((out/'summary.json').read_text())
    docs=[json.loads(s) for s in (out/'documents.jsonl').read_text().splitlines()]
    qs=[json.loads(s) for s in (out/'questions.jsonl').read_text().splitlines()]
    predictions=json.loads((out/'predictions.json').read_text())
    assert len(docs)==1440 and len(qs)==600
    for stem in ('documents','questions'):
        assert hashlib.sha256((out/f'{stem}.jsonl').read_bytes()).hexdigest()==summary[f'{stem}_sha256']
    test=[q for q in qs if q['asset']>=20];assert len(test)==400
    assert len({q['id'] for q in test})==400
    ids={q['id'] for q in test};known={d['id'] for d in docs}
    # Inspection of the written schema is intentional: fail rather than infer IDs.
    counts={}
    for mode,rows in predictions.items():
        assert len(rows)==400
        key='qid'
        assert {r[key] for r in rows}==ids
        assert len({r[key] for r in rows})==400
        for row in rows:
            assert set(row['retrieved'])<=known
            assert row['expected'] is None or row['expected'] in known
            assert row['answer'] is None or row['answer']['source'] in known
        counts[mode]={'correct':sum(r['correct'] for r in rows),'leaks':sum(r['leak'] for r in rows)}
        expected=next(s for s in summary['summaries'] if s['mode']==mode)
        assert abs(counts[mode]['correct']/400-expected['correct_outcome'])<1e-12
        assert counts[mode]['leaks']==expected['leaks']
    assert {k:v['correct'] for k,v in counts.items()}=={'bm25':128,'lsa':256,'hybrid':224,'filtered':400,'no_acl':336,'no_version':368}
    stress=json.loads((out/'metadata_stress.json').read_text())
    assert [x['changed_records'] for x in stress]==[120,240,480]
    return {'documents':len(docs),'questions':len(qs),'test_questions':len(test),
            'matching_data_hashes':True,'prediction_counts':counts,
            'metadata_flags_changed':[120,240,480]}

def main():
    logs=ROOT/'execution_logs';logs.mkdir(exist_ok=True)
    commands=[['reference_rag.py','--output','results'],['tests.py'],['metadata_stress.py'],['numerical_checks.py']]
    report={'python':platform.python_version(),'platform':platform.platform(),'runs':[]}
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONIOENCODING':'utf-8'}
    for args in commands:
        t=time.perf_counter();r=subprocess.run([sys.executable]+args,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        (logs/(Path(args[0]).stem+'.txt')).write_text(r.stdout+r.stderr,encoding='utf-8')
        report['runs'].append({'command':args,'returncode':r.returncode,'seconds':round(time.perf_counter()-t,4),'stdout':r.stdout,'stderr':r.stderr})
        print(args[0],r.returncode,flush=True)
    report['passed']=sum(r['returncode']==0 for r in report['runs']);report['total']=len(commands)
    if report['passed']==report['total']:
        report['artifact_checks']=check_results()
    (ROOT/'execution_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return int(report['passed']!=report['total'])

if __name__=='__main__':sys.exit(main())
