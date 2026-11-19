export const modes=['late_video','late_audio','late_sensor','late_fusion','continuous','tokenized','uncalibrated_continuous'];
export function validateReport(report){
  if(report?.protocol!=='synthetic_multimodal_fusion_v1'||report.data_kind!=='simulation'||!Array.isArray(report.test_windows))throw Error('Expected synthetic fusion report v1');
  if(!report.test_windows.length||!Number.isInteger(report.session_count)||report.session_count<1||modes.some(k=>!Number.isFinite(report.results?.[k]?.auc)))throw Error('Incomplete score report');
  if(report.fusion_gate?.aligned_fusion_allowed!==true||report.clock_evidence?.source!=='synthetic_landmarks'||!report.clock_evidence?.independent_holdout_passed)throw Error('Aligned fusion lacks independent synthetic clock evidence');
  if(['video','audio','sensor'].some(k=>!Number.isFinite(report.modality_profiles?.[k]?.median_rate_hz)||!Number.isFinite(report.modality_profiles[k]?.missing_window_fraction)))throw Error('Missing modality profile');
  if(report.test_windows.some(w=>!w.episode_id||!Number.isFinite(w.time_s)||![0,1].includes(w.label)||modes.some(k=>!Number.isFinite(w.scores?.[k]))))throw Error('Invalid window score');
  return report;
}
export function episodes(report,category='All'){
  const rows=report.test_windows.filter(w=>category==='All'||w.category===category||w.category==='Good');
  return [...new Set(rows.map(w=>w.episode_id))];
}
export function rowsFor(report,id){return report.test_windows.filter(w=>w.episode_id===id).sort((a,b)=>a.time_s-b.time_s);}
