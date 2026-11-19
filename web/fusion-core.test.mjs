import test from 'node:test';import assert from 'node:assert/strict';import {modes,validateReport,episodes,rowsFor} from './fusion-core.mjs';
test('validates synthetic provenance and slices episode rows',()=>{const scores=Object.fromEntries(modes.map(k=>[k,0.5]));const results=Object.fromEntries(modes.map(k=>[k,{auc:0.7}]));const profile={median_rate_hz:5,missing_window_fraction:0};const report={protocol:'synthetic_multimodal_fusion_v1',data_kind:'simulation',session_count:1,results,clock_evidence:{source:'synthetic_landmarks',independent_holdout_passed:true},fusion_gate:{aligned_fusion_allowed:true},modality_profiles:{video:profile,audio:profile,sensor:profile},test_windows:[{episode_id:'a',category:'Good',time_s:1.5,label:0,scores},{episode_id:'a',category:'Good',time_s:0.5,label:0,scores},{episode_id:'b',category:'Overlap',time_s:0.5,label:1,scores}]};
  validateReport(report);assert.deepEqual(episodes(report,'Overlap'),['a','b']);assert.equal(rowsFor(report,'a')[0].time_s,0.5);
  assert.throws(()=>validateReport({...report,data_kind:'recorded'}));
  assert.throws(()=>validateReport({...report,fusion_gate:{aligned_fusion_allowed:false}}),/clock evidence/);
});
