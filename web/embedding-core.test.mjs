import test from 'node:test';
import assert from 'node:assert/strict';
import {validateIndex,filterSegments,nearest} from './embedding-core.mjs';

const vector = (v) => Array.from({length:768}, (_,i)=>i===v?1:0);
const segment = (id, category, slot) => ({id, sample_id:id, session_id:`s${slot}`,split:'TEST',category,start_s:0,end_s:4,embeddings:{video:vector(slot),audio:vector(0)}});
test('filters and ranks by selected modality without class prediction',()=>{
  const segments=[segment('good','Good',0),segment('overlap','Overlap',0),segment('other','Overlap',1)];
  validateIndex({schema_version:1,model:'google/embeddinggemma-2',segments});
  assert.equal(filterSegments(segments,{category:'Overlap'}).length,2);
  assert.equal(nearest(segments,segments[0],'video')[0].segment.id,'overlap');
});
test('rejects malformed or duplicate vectors',()=>{
  const s=segment('a','Good',0);
  assert.throws(()=>validateIndex({schema_version:1,model:'google/embeddinggemma-2',segments:[s,s]}));
  s.embeddings.video[0]=NaN;
  assert.throws(()=>validateIndex({schema_version:1,model:'google/embeddinggemma-2',segments:[s]}));
});
