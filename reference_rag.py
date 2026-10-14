"""Closed synthetic retrieval experiment; no neural generator or real manuals."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, re, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

SEED=20261012
TOPICS={
 'seal':('замена уплотнения','смена манжеты','Проверить герметичность по режиму выбранной модификации.'),
 'sensor':('замена датчика','смена измерителя','Сопоставить идентификатор и диапазон датчика с карточкой.'),
 'filter':('очистка фильтра','чистка сетки','Осмотреть фильтрующий элемент по действующему регламенту.'),
 'controller':('обновление контроллера','перепрошивка блока','Проверить утверждённую версию конфигурации контроллера.')}

@dataclass(frozen=True)
class Document:
 id:str
 asset:int
 variant:str
 topic:str
 revision:int
 current:bool
 restricted:bool
 text:str
 answer:str

@dataclass(frozen=True)
class Question:
 id:str
 asset:int
 variant:str
 topic:str
 role:str
 text:str
 expected:str|None

def generate():
 docs=[];questions=[]
 for asset in range(60):
  code=f'Н-{asset:03d}'
  restricted=asset%5==0
  for variant in ('А','Б'):
   for topic,(formal,spoken,answer) in TOPICS.items():
    for rev in (1,2,3):
     ident=f'{asset:03d}-{variant}-{topic}-r{rev}'
     # Old editions repeat formal terms; this creates a controlled lexical trap.
     repeat=(formal+' ')*(4 if rev<3 else 1)
     text=(f'Регламент {code}, модификация {variant}. {repeat}'
           f'Событие: {formal}. Редакция {rev}. '
           f'Требование: {answer} '
           +('Историческая запись; заменена редакцией 3.' if rev<3 else 'Действующая редакция.'))
     docs.append(Document(ident,asset,variant,topic,rev,rev==3,restricted,text,answer))
    # Alternating forms are fixed before evaluation.
    wording=formal if (asset+list(TOPICS).index(topic))%2 else spoken
    text=f'Что проверяют после операции «{wording}» для {code}, модификация {variant}?'
    expected=None if restricted else f'{asset:03d}-{variant}-{topic}-r3'
    questions.append(Question(f'q-{asset:03d}-{variant}-{topic}',asset,variant,topic,'reader',text,expected))
   questions.append(Question(f'q-{asset:03d}-{variant}-unknown',asset,variant,'unknown','reader',
     f'Какова неизвестная диагностика для {code}, модификация {variant}?',None))
 return docs,questions

def tokens(text:str):
 return re.findall(r'[а-яёa-z]+(?:-\d+)?|\d+',text.lower())

FORMS={
 'seal':('замены уплотнения','смены манжеты'),
 'sensor':('замены датчика','смены измерителя'),
 'filter':('очистки фильтра','чистки сетки'),
 'controller':('обновления контроллера','перепрошивки блока')}

def canonical(text:str):
 text=text.lower()
 for key,(formal,spoken,_) in TOPICS.items():
  for form in (formal,spoken,*FORMS[key]):
   text=text.replace(form,key)
 return text

def parse_scope(text:str):
 asset=re.search(r'н-(\d{3})',text.lower())
 variant=re.search(r'модификация\s+([аб])',text.lower())
 topic=next((k for k,(f,s,_) in TOPICS.items()
             if any(form in text.lower() for form in (f,s,*FORMS[k]))),None)
 return (int(asset.group(1)) if asset else None,
         variant.group(1).upper() if variant else None,topic)

class Engine:
 def __init__(self,documents:list[Document],dimensions=64,canonicalize=True):
  self.docs=documents
  self.canonicalize=canonicalize
  texts=[d.text for d in documents]
  self.cv=CountVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False)
  counts=self.cv.fit_transform(texts).tocsr().astype(float)
  n=counts.shape[0];df=np.asarray((counts>0).sum(axis=0)).ravel()
  self.idf=np.log(1+(n-df+.5)/(df+.5))
  lengths=np.asarray(counts.sum(axis=1)).ravel();avg=lengths.mean()
  weighted=counts.copy();k1=1.2;b=.75
  for i in range(n):
   lo,hi=weighted.indptr[i:i+2];tf=weighted.data[lo:hi]
   weighted.data[lo:hi]=tf*(k1+1)/(tf+k1*(1-b+b*lengths[i]/avg))
  self.bm=weighted.multiply(self.idf).tocsr()
  self.tv=TfidfVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False)
  tfidf=self.tv.fit_transform([canonical(t) if canonicalize else t.lower() for t in texts])
  # LSA is a fitted linear text representation, not a pretrained neural embedding.
  self.svd=TruncatedSVD(n_components=dimensions,random_state=SEED)
  self.dense=normalize(self.svd.fit_transform(tfidf))
  self.assets=np.array([d.asset for d in documents])
  self.variants=np.array([d.variant for d in documents])
  self.current=np.array([d.current for d in documents])
  self.restricted=np.array([d.restricted for d in documents])

 def score(self,questions:Iterable[Question]):
  qs=list(questions);raw=[q.text for q in qs]
  qcounts=self.cv.transform(raw)
  qpresence=(qcounts>0).astype(float)
  bm=(qpresence@self.bm.T).toarray()
  qvec=normalize(self.svd.transform(self.tv.transform([canonical(t) if self.canonicalize else t.lower() for t in raw])))
  semantic=qvec@self.dense.T
  return bm,semantic

 def order(self,scores):
  # Stable order follows corpus identifiers when scores tie.
  return np.argsort(-scores,kind='stable')

 def fuse(self,bm,semantic,depth=30):
  result=np.zeros(len(self.docs))
  for scores in (bm,semantic):
   for rank,index in enumerate(self.order(scores)[:depth],1):
    result[index]+=1/(60+rank)
  return result

 def retrieve(self,q,bm,semantic,mode,depth=30):
  scores=bm if mode=='bm25' else semantic if mode=='lsa' else self.fuse(bm,semantic,depth)
  order=self.order(scores)
  if mode in ('filtered','no_acl','no_version'):
   asset,variant,topic=parse_scope(q.text)
   if asset is None or variant is None or topic is None:return []
   allowed=(self.assets==asset)&(self.variants==variant)
   if mode!='no_version':allowed&=self.current
   if mode!='no_acl' and q.role=='reader':allowed&=~self.restricted
   # Filters are applied to the whole ranked corpus, not only the short pool.
   order=order[allowed[order]]
  return list(map(int,order[:5]))

 def answer(self,q,indices,checked=False):
  if not indices:return None
  candidate=self.docs[indices[0]]
  if checked:
   asset,variant,topic=parse_scope(q.text)
   if (candidate.asset,candidate.variant,candidate.topic)!=(asset,variant,topic):return None
  return {'source':candidate.id,'text':candidate.answer,'revision':candidate.revision}

def evaluate(engine,qs,bm,semantic,mode,depth):
 rows=[];durations=[]
 for q,b,s in zip(qs,bm,semantic):
  start=time.perf_counter()
  idx=engine.retrieve(q,b,s,mode,depth)
  ans=engine.answer(q,idx,checked=mode in ('filtered','no_acl','no_version'))
  durations.append((time.perf_counter()-start)*1000)
  ids=[engine.docs[i].id for i in idx]
  rank=ids.index(q.expected)+1 if q.expected in ids else 0
  correct=(ans is None) if q.expected is None else (ans is not None and ans['source']==q.expected)
  leak=bool(ans and engine.docs[idx[0]].restricted and q.role=='reader')
  rows.append({'qid':q.id,'asset':q.asset,'answerable':q.expected is not None,
   'hit5':int(rank>0),'mrr':1/rank if rank else 0,'ndcg':1/math.log2(rank+1) if rank else 0,
   'correct':int(correct),'answered':int(ans is not None),'leak':int(leak),
   'expected':q.expected,'answer':ans,'retrieved':ids})
 answerable=[r for r in rows if r['answerable']]
 unavailable=[r for r in rows if not r['answerable']]
 mean=lambda key,x:float(np.mean([r[key] for r in x])) if x else None
 summary={'mode':mode,'questions':len(rows),'answerable':len(answerable),
  'recall5':mean('hit5',answerable),'mrr5':mean('mrr',answerable),'ndcg5':mean('ndcg',answerable),
  'correct_outcome':mean('correct',rows),'answer_coverage':mean('answered',rows),
  'correct_refusal':mean('correct',unavailable),'leaks':sum(r['leak'] for r in rows),
  'routing_p50_ms':float(np.median(durations)),'routing_p95_ms':float(np.percentile(durations,95))}
 return summary,rows

def clustered_interval(rows,iterations=2000):
 # Resample equipment groups; queries in a group remain together.
 assets=sorted({r['asset'] for r in rows})
 group=np.array([np.mean([r['correct'] for r in rows if r['asset']==a]) for a in assets])
 rng=np.random.default_rng(SEED)
 sampled=group[rng.integers(0,len(group),size=(iterations,len(group)))].mean(axis=1)
 return [float(x) for x in np.quantile(sampled,[.025,.975])]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='results');args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 docs,allq=generate();engine=Engine(docs)
 # Families 0..19 are used only for choosing fusion depth; 20..59 are independent test groups.
 tuning=[q for q in allq if q.asset<20];test=[q for q in allq if q.asset>=20]
 tb,ts=engine.score(tuning);vb,vs=engine.score(test)
 choices=[]
 for depth in (10,30,60):
  summary,_=evaluate(engine,tuning,tb,ts,'filtered',depth);choices.append((summary['correct_outcome'],depth))
 selected=max(choices,key=lambda p:(p[0],-p[1]))[1]
 summaries=[];full={}
 for mode in ('bm25','lsa','hybrid','filtered','no_acl','no_version'):
  summary,rows=evaluate(engine,test,vb,vs,mode,selected)
  summary['outcome_ci95']=clustered_interval(rows);summaries.append(summary);full[mode]=rows
 meta={'seed':SEED,'documents':len(docs),'questions_total':len(allq),
  'tuning_questions':len(tuning),'test_questions':len(test),'tuning_assets':20,'test_assets':40,
  'selected_fusion_depth':selected,'lsa_dimensions':64,'generator':'deterministic extractive',
  'corpus':'synthetic; no real engineering limits','timing_scope':'ranking/filtering/extractive output only; scores precomputed',
  'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
  'summaries':summaries}
 for name,objects in [('documents',docs),('questions',allq)]:
  content=''.join(json.dumps(asdict(o),ensure_ascii=False)+'\n' for o in objects)
  (out/f'{name}.jsonl').write_text(content,encoding='utf-8')
  meta[f'{name}_sha256']=hashlib.sha256(content.encode()).hexdigest()
 sensitivity=[]
 for dim,canon in ((16,True),(32,True),(64,True),(64,False)):
  alternative=Engine(docs,dimensions=dim,canonicalize=canon)
  ab,ass=alternative.score(test)
  extra,_=evaluate(alternative,test,ab,ass,'filtered',selected)
  sensitivity.append({'dimensions':dim,'canonicalization':canon,**extra})
 meta['sensitivity']=sensitivity
 (out/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'predictions.json').write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
