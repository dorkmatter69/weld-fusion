import { runVirtualEpisode, reviewEpisode } from "./cell-core.mjs";

const $ = id => document.getElementById(id);
const cell = $("cell"), chart = $("chart");
let episode = runVirtualEpisode(), cursor = 0, timer = null;

function line(ctx, pts, color) {
  if (!pts.length) return;
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke();
}

function drawCell(row) {
  const ctx = cell.getContext("2d"), { width:w, height:h } = cell;
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#0e1a22"; ctx.fillRect(0,0,w,h);
  if (episode.source.kind !== "simulation") {
    ctx.fillStyle="#b4c8d4"; ctx.font="20px system-ui";
    ctx.fillText("Recorded episode · no torch pose available",42,145); return;
  }
  ctx.fillStyle="#507080"; ctx.fillRect(38,155,w-76,88);
  ctx.fillStyle="#637f8b"; ctx.fillRect(38,155,w-76,3);
  ctx.setLineDash([5,5]); line(ctx, [[38,200],[w-38,200]],"#f3a068"); ctx.setLineDash([]);
  const x = 52 + (w-104)*Math.min(row.torch_x_m == null ? row.time_s/38 : row.torch_x_m/0.38,1);
  ctx.strokeStyle="#c1d7e1"; ctx.lineWidth=7; ctx.beginPath(); ctx.moveTo(x,45); ctx.lineTo(x,139); ctx.stroke();
  ctx.fillStyle=row.plc_state==="Weld"?"#ff993f":"#678493";
  ctx.beginPath(); ctx.arc(x,149,row.plc_state==="Weld"?11:6,0,Math.PI*2); ctx.fill();
  ctx.fillStyle="#aec4cf"; ctx.font="14px system-ui";
  ctx.fillText("SIMULATED FLAT COUPON",38,285);
}

function drawChart() {
  const ctx=chart.getContext("2d"), {width:w,height:h}=chart;
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#0e1a22"; ctx.fillRect(0,0,w,h);
  const rows=episode.telemetry, duration=Math.max(rows.at(-1).time_s,1);
  for(let i=0;i<5;i++){const y=20+i*45;line(ctx,[[45,y],[w-15,y]],"#28404b");}
  const project=(key,max)=>rows.map(r=>[45+(w-70)*r.time_s/duration,h-20-(h-45)*Math.max(0,Number(r[key])||0)/max]);
  const hasCurrent=rows.some(r=>Number.isFinite(r.current_a));
  const hasGas=rows.some(r=>Number.isFinite(r.gas_bar));
  if(hasCurrent) line(ctx,project("current_a",220),"#ff9853");
  if(hasGas) line(ctx,project("gas_bar",6),"#5fc5c3");
  const marker=45+(w-70)*rows[cursor].time_s/duration;
  line(ctx,[[marker,15],[marker,h-15]],"#eaf2f6");
  ctx.fillStyle="#ff9853";ctx.font="13px system-ui";ctx.fillText(hasCurrent?"Current (A)":"Current unavailable",55,20);
  ctx.fillStyle="#5fc5c3";ctx.fillText(hasGas?"Gas (bar)":"Gas unavailable",220,20);
}

function render() {
  const row=episode.telemetry[cursor];
  $("clock").textContent=`${row.time_s.toFixed(1)} s`;
  $("state").textContent=row.plc_state||"Recorded";
  $("state").classList.toggle("fault",row.plc_state==="Fault");
  for(const [id,key,unit] of [["current","current_a","A"],["voltage","voltage_v","V"],["gas","gas_bar","bar"],["seam","seam_offset_mm","mm"],["vision","vision_age_s","s"]]){
    $(id).textContent=Number.isFinite(row[key])?`${row[key].toFixed(2)} ${unit}`:"Not recorded";
  }
  drawCell(row);drawChart();
}

function load(next) {
  if(next?.schema_version!=="1.0.0"||!next.source?.kind||!Array.isArray(next.telemetry)||!next.telemetry.length||
    (next.reviews != null && !Array.isArray(next.reviews))||
    next.telemetry.some(r=>!Number.isFinite(r.time_s))){throw new Error("Expected WeldEpisode v1 with time_s telemetry");}
  clearInterval(timer);timer=null;episode=next;cursor=0;
  $("scrub").max=next.telemetry.length-1;$("scrub").value=0;
  if(next.source.kind==="simulation" && !next.reviews?.length) next.reviews=reviewEpisode(next);
  $("reviews").replaceChildren();
  for(const r of next.reviews||[]){
    const box=document.createElement("div");box.className=`review ${r.verdict}`;
    const title=document.createElement("b");title.textContent=`${r.agent}: ${r.verdict}`;
    const detail=document.createElement("p");detail.textContent=r.finding;
    const evidence=document.createElement("small");evidence.textContent=r.evidence_time_s==null?"":`Evidence at ${r.evidence_time_s}s`;
    box.append(title,detail,evidence);$("reviews").append(box);
  }
  if(!next.reviews?.length) $("reviews").textContent="No review has been recorded for this episode.";
  $("provenance").textContent=JSON.stringify({episode_id:next.episode_id,source:next.source,process:next.process,timeline:next.timeline,streams:next.streams},null,2);
  render();
}

$("run").addEventListener("click",()=>{
  load(runVirtualEpisode($("fault").value));
  timer=setInterval(()=>{if(cursor>=episode.telemetry.length-1){clearInterval(timer);timer=null;return;}cursor++;$("scrub").value=cursor;render();},65);
});
$("scrub").addEventListener("input",event=>{cursor=Number(event.target.value);render();});
$("download").addEventListener("click",()=>{
  const blob=new Blob([JSON.stringify(episode,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=`${episode.episode_id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$("import").addEventListener("change",async event=>{
  const file=event.target.files[0];if(!file)return;
  try{load(JSON.parse(await file.text()));}catch(error){alert(error.message);}
  event.target.value="";
});
load(episode);
