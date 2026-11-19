export function validateIndex(index) {
  if (index?.schema_version !== 1 || index.model !== "google/embeddinggemma-2" || !Array.isArray(index.segments)) throw Error("Expected EmbeddingGemma 2 index v1");
  const ids = new Set();
  for (const s of index.segments) {
    if (!s.id || !s.sample_id || !s.category || !s.split || !s.session_id || !Number.isFinite(s.start_s) || !Number.isFinite(s.end_s) || s.end_s <= s.start_s || ids.has(s.id)) throw Error("Invalid or duplicate segment");
    ids.add(s.id);
    for (const mode of ["audio", "video"]) if (!Array.isArray(s.embeddings?.[mode]) || s.embeddings[mode].length !== 768 || !s.embeddings[mode].every(Number.isFinite)) throw Error(`Invalid ${mode} vector in ${s.id}`);
  }
  return index;
}
export function filterSegments(segments, {split="",category="",session="",search=""}={}) {
  return segments.filter(s=>(!split||s.split===split)&&(!category||s.category===category)&&(!session||s.session_id===session)&&(!search||`${s.sample_id} ${s.id}`.toLowerCase().includes(search.toLowerCase())));
}
export function cosine(a,b) {
  let dot=0, aa=0, bb=0;
  for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}
  return aa&&bb?dot/Math.sqrt(aa*bb):0;
}
export function nearest(segments, selected, mode, limit=10) {
  if(!["audio","video"].includes(mode))throw Error("Unknown modality");
  return segments.filter(s=>s.id!==selected.id).map(s=>({segment:s,similarity:cosine(selected.embeddings[mode],s.embeddings[mode])})).sort((a,b)=>b.similarity-a.similarity).slice(0,limit);
}
