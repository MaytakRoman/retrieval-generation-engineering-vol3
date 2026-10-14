import unittest
from reference_rag import Engine,generate,parse_scope,Question

class ContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.docs,cls.qs=generate();cls.engine=Engine(cls.docs)

 def run_one(self,q,mode='filtered'):
  b,s=self.engine.score([q]);idx=self.engine.retrieve(q,b[0],s[0],mode,10)
  return self.engine.answer(q,idx,checked=True),idx

 def test_corpus_identity_and_partition(self):
  self.assertEqual(len(self.docs),1440)
  self.assertEqual(len({d.id for d in self.docs}),1440)
  tuning={q.asset for q in self.qs if q.asset<20}
  test={q.asset for q in self.qs if q.asset>=20}
  self.assertFalse(tuning & test)
  self.assertEqual(sum(q.asset>=20 for q in self.qs),400)

 def test_reader_cannot_receive_restricted_material(self):
  q=next(q for q in self.qs if q.asset==20 and q.topic=='seal')
  ans,idx=self.run_one(q)
  self.assertIsNone(ans);self.assertEqual(idx,[])

 def test_unknown_operation_is_not_invented(self):
  q=next(q for q in self.qs if q.asset==21 and q.topic=='unknown')
  ans,idx=self.run_one(q)
  self.assertIsNone(ans);self.assertEqual(idx,[])

 def test_current_source_and_variant_are_preserved(self):
  q=next(q for q in self.qs if q.asset==21 and q.variant=='Б' and q.topic=='filter')
  ans,idx=self.run_one(q)
  self.assertIsNotNone(ans)
  self.assertEqual(ans['source'],q.expected)
  self.assertEqual(ans['revision'],3)
  self.assertTrue(all(self.docs[i].variant=='Б' and self.docs[i].current for i in idx))

 def test_paraphrase_does_not_change_identity(self):
  scope=parse_scope('Что после смены манжеты для Н-021, модификация Б?')
  self.assertEqual(scope,(21,'Б','seal'))

 def test_missing_identifier_requires_no_automatic_source(self):
  q=Question('missing',21,'А','seal','reader','Что после замены уплотнения?',None)
  ans,idx=self.run_one(q)
  self.assertIsNone(ans);self.assertEqual(idx,[])

if __name__=='__main__':unittest.main()
