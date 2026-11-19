import {validateIndex,filterSegments,nearest} from "./embedding-core.mjs";
const $=id=>document.getElementById(id);
let index=null,selected=null,demo=false;
function optionValues(id, values){const all={split:'All splits',category:'All categories',session:'All sessions'};$(id).replaceChildren(new Option(all[id],""), ...[...new Set(values)].sort().map(v=>new Option(v,v)));}
function filters(){return {split:$('split').value,category:$('category').value,session:$('session').value,search:$('search').value};}
function render(){
  if(!index)return;
  const visible=filterSegments(index.segments,filters());
  $('segments').textContent=visible.length.toLocaleString();
  $('welds').textContent=new Set(visible.map(s=>s.sample_id)).size;
  $('classes').textContent=new Set(visible.map(s=>s.category)).size;
  $('alignment').textContent=index.clock_alignment_verified===true?'Verified':'Unverified';
  $('rows').replaceChildren();
  for(const s of visible.slice(0,1000)){
    const tr=document.createElement('tr');tr.classList.toggle('selected',s.id===selected?.id);
    for(const val of [s.sample_id,`${s.start_s.toFixed(2)}–${s.end_s.toFixed(2)} s`,s.split,s.category,s.session_id]){const td=document.createElement('td');td.textContent=val;tr.append(td);}
    tr.addEventListener('click',()=>{selected=s;render();});$('rows').append(tr);
  }
  $('status').textContent=`${demo?'SYNTHETIC DEMO · no model inference · ':''}${index.segments.length.toLocaleString()} segments loaded locally · ${visible.length>1000?'Showing first 1,000 rows · ':''}source clocks ${index.clock_alignment_verified?'verified':'unverified'}`;
  $('neighbors').replaceChildren();
  $('selection').textContent=selected?`${selected.sample_id} · ${selected.start_s.toFixed(2)}–${selected.end_s.toFixed(2)} s · ${$('mode').value} similarity`:'Select a segment to inspect cosine similarity.';
  if(selected)for(const result of nearest(visible,selected,$('mode').value)){
    const li=document.createElement('li');const title=document.createElement('b');title.textContent=result.segment.sample_id;
    const sub=document.createElement('small');sub.textContent=`${result.segment.start_s.toFixed(2)}–${result.segment.end_s.toFixed(2)} s · ${result.segment.category} · cosine ${result.similarity.toFixed(3)}`;
    li.append(title,sub);$('neighbors').append(li);
  }
}
function load(next,isDemo){index=validateIndex(next);demo=isDemo;selected=null;optionValues('split',index.segments.map(s=>s.split));optionValues('category',index.segments.map(s=>s.category));optionValues('session',index.segments.map(s=>s.session_id));$('search').value='';render();}
$('file').addEventListener('change',async e=>{try{if(e.target.files?.[0])load(JSON.parse(await e.target.files[0].text()),false);}catch(error){$('status').textContent=`Cannot open index: ${error.message}`;}e.target.value='';});
$('demo').addEventListener('click',()=>{
  const basis=n=>Array.from({length:768},(_,i)=>i===n?1:0);
  const sample=(id,session,split,category,slot,v,a)=>({id:`${id}:${slot}`,sample_id:id,session_id:session,split,category,start_s:slot*2,end_s:slot*2+4,embeddings:{video:basis(v),audio:basis(a)}});
  load({schema_version:1,model:'google/embeddinggemma-2',clock_alignment_verified:false,segments:[sample('demo/good-a','demo-a','TRAIN','Good',0,0,0),sample('demo/good-b','demo-b','VALIDATION','Good',0,0,1),sample('demo/overlap-b','demo-b','VALIDATION','Overlap',0,1,2),sample('demo/good-c','demo-c','TEST','Good',0,0,1),sample('demo/overlap-c','demo-c','TEST','Overlap',0,1,2),sample('demo/overlap-c','demo-c','TEST','Overlap',1,1,2)]},true);
});
for(const id of ['split','category','session','mode','search'])$(id).addEventListener(id==='search'?'input':'change',render);
