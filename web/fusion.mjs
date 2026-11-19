import {modes,validateReport,episodes,rowsFor} from './fusion-core.mjs';
const $=id=>document.getElementById(id);
const labels={late_video:'Video only',late_audio:'Audio only',late_fusion:'Late audio–video',continuous:'Calibrated continuous',tokenized:'Calibrated tokens',uncalibrated_continuous:'Uncalibrated continuous'};
let report=null;
function option(id,value,text){$(id).append(new Option(text,value));}
function init(data){
  report=validateReport(data);$('category').replaceChildren(new Option('All defects','All'));
  for(const category of Object.keys(report.per_category))option('category',category,category);
  $('model').replaceChildren();for(const mode of modes)option('model',mode,labels[mode]);$('model').value='continuous';
  $('sessions').textContent=report.session_count;
  $('windows').textContent=report.test_windows.length;
  $('clock').textContent='3 + 1 landmarks';
  $('status').textContent=`Synthetic seed ${report.seed} · ${report.splits.TRAIN} training, ${report.splits.VALIDATION} validation and ${report.splits.TEST} test welds · late-video weight ${report.late_video_weight_selected_on_validation.toFixed(2)}`;
  render();
}
function render(){
  if(!report)return;
  const category=$('category').value,model=$('model').value;
  const metric=category==='All'?report.results:report.per_category[category];
  $('score').textContent=metric[model].auc.toFixed(3);
  $('bars').replaceChildren();for(const mode of modes){const row=document.createElement('div');row.className=`bar-row ${mode.startsWith('late_')?'baseline':''} ${mode.startsWith('uncalibrated')?'uncalibrated':''}`;
    const title=document.createElement('span');title.textContent=labels[mode];const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=`${Math.max(0,Math.min(100,100*metric[mode].auc))}%`;track.append(fill);const value=document.createElement('strong');value.textContent=metric[mode].auc.toFixed(3);row.append(title,track,value);$('bars').append(row);}
  const previous=$('episode').value;$('episode').replaceChildren();for(const id of episodes(report,category))option('episode',id,id);
  if([...$('episode').options].some(o=>o.value===previous))$('episode').value=previous;
  $('slice-rows').replaceChildren();for(const [name,results] of Object.entries(report.per_category)){const tr=document.createElement('tr');const cell=document.createElement('td');cell.textContent=name;tr.append(cell);
    for(const key of modes){const td=document.createElement('td');td.textContent=results[key].auc.toFixed(3);tr.append(td);}$('slice-rows').append(tr);}
  draw();
}
function draw(){
  if(!report)return;const rows=rowsFor(report,$('episode').value),canvas=$('timeline'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  ctx.fillStyle='#0c1e26';ctx.fillRect(0,0,w,h);if(!rows.length)return;
  const x=i=>45+i*(w-65)/rows.length,y=v=>h-28-v*(h-58);
  rows.forEach((r,i)=>{if(r.label){ctx.fillStyle='#5e342c';ctx.fillRect(x(i),18,(w-65)/rows.length,h-46);}});
  for(const q of [0,0.25,0.5,0.75,1]){ctx.strokeStyle='#2b4650';ctx.beginPath();ctx.moveTo(45,y(q));ctx.lineTo(w-20,y(q));ctx.stroke();ctx.fillStyle='#91abb2';ctx.font='14px system-ui';ctx.fillText(q.toFixed(2),5,y(q)+4);}
  for(const [key,color] of [[$('model').value,'#6fddba'],['late_fusion','#edaa7f']]){ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();rows.forEach((r,i)=>i?ctx.lineTo(x(i)+8,y(r.scores[key])):ctx.moveTo(x(i)+8,y(r.scores[key])));ctx.stroke();}
  ctx.fillStyle='#bcd1d4';ctx.fillText('0 s',45,h-6);ctx.fillText(`${rows.length} s`,w-62,h-6);
  $('episode-detail').textContent=`${rows[0].episode_id} · ${rows[0].category} · green: ${labels[$('model').value]} · orange: late fusion · shaded: synthetic defect interval`;
}
$('category').addEventListener('change',render);$('model').addEventListener('change',render);$('episode').addEventListener('change',draw);
$('file').addEventListener('change',async e=>{try{if(e.target.files?.[0])init(JSON.parse(await e.target.files[0].text()));}catch(err){$('status').textContent=`Invalid report: ${err.message}`;}e.target.value='';});
fetch('demo-multimodal-report.json').then(r=>{if(!r.ok)throw Error('demo report unavailable');return r.json();}).then(init).catch(e=>{$('status').textContent=e.message;});
