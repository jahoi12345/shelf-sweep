// Controlled browser primitives; NO real devices or network. Ported from the reference's client-regressions.cjs
// (Apache-2.0, see app/NOTICE.md) and adapted to app/static/app.js. Mock validation only.
const vm=require('node:vm'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const src=fs.readFileSync(path.join(__dirname,'../app/static/app.js'),'utf8').replace(/\ncreateSession\(true\);\s*$/,'\n');
function env(){
 const elements=new Map(), sockets=[], sources=[], storage=new Map();let counter=0;
 const el=()=>({textContent:'',innerHTML:'',className:'',hidden:false,disabled:false,href:'',style:{},dataset:{},value:'',width:0,height:0,videoWidth:1920,videoHeight:1080,currentTime:3.5,
   classList:{add(){},remove(){},toggle(){}},querySelector(){return el()},querySelectorAll(){return[]},addEventListener(){},focus(){},showModal(){},close(){},getTracks(){return[]},
   getContext(){return {drawImage(){}}},toDataURL(){return 'data:image/jpeg;base64,QUJD'},play:async()=>{},pause(){},removeAttribute(){},setAttribute(){},getAttribute(){return null},readyState:4,removeEventListener(){}});
 class WS{static OPEN=1;constructor(url){this.url=url;this.readyState=0;sockets.push(this);setImmediate(()=>{if(this.closed)return;this.readyState=1;this.onmessage?.({data:JSON.stringify({type:'ready',voice:true,session_id:new URL(url).searchParams.get('session_id')})})})}send(x){this.sent??=[];this.sent.push(JSON.parse(x))}close(){this.closed=true;this.readyState=3;this.onclose?.()}}
 class Audio{constructor(){this.currentTime=10;this.destination={}}async resume(){}createBuffer(c,n,r){return {duration:n/r,getChannelData(){return new Float32Array(n)}}}createBufferSource(){const x={stopped:false,connect(){},start(at){this.startAt=at},stop(){this.stopped=true}};sources.push(x);return x}}
 const ctx=vm.createContext({console,crypto,URLSearchParams,URL,document:{querySelector(k){if(!elements.has(k))elements.set(k,el());return elements.get(k)},querySelectorAll(){return[]},addEventListener(){}},history:{replaceState(){}},location:{hash:'',protocol:'http:',origin:'http://127.0.0.1:4188'},setInterval(){return 1},clearInterval(){},window:{location:{protocol:'http:',origin:'http://127.0.0.1:4188'},AudioContext:Audio,addEventListener(){},setInterval(){return 1},clearInterval(){}},sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},navigator:{},WebSocket:WS,AudioContext:Audio,fetch:async url=>({ok:true,json:async()=>url.endsWith('/health')?{replay_videos:['v.mov'],has_api_key:true}:url.endsWith('/api/scans')?{scans:[{label:'YugoTest1',sha256:'c4041f08aa',preview_available:true},{label:'YugoTest2',sha256:'5dca9395bb',preview_available:true}]}:url.endsWith('/api/build')?{pid:1,backend_fingerprint_loaded:'abc',static_fingerprint_current:'def'}:url.includes('/api/hosted')?{disabled:false,calls:0,blocked:0,tokens:{total_tokens:0},cost_usd:0,live_sessions_open:[]}:{session_id:'session-'+(++counter),has_api_key:true,state:{session_id:'session-'+counter,transcript:[],tool_activity:[]}}}),setTimeout,clearTimeout,atob:s=>Buffer.from(s,'base64').toString('binary'),btoa:s=>Buffer.from(s,'binary').toString('base64')});
 vm.runInContext(src,ctx);vm.runInContext('state={...emptyState};sessionId="initial";',ctx);return {ctx,sockets,sources,elements,run:x=>vm.runInContext(x,ctx)};
}
const tests=[];const test=(name,fn)=>tests.push([name,fn]);
test('Books table shows merged members with their evidence and a split control that posts the member id',async()=>{
 const e=env();e.run('sessionId="s1";window.__posts=[];const _f=fetch;fetch=async(u,o)=>{if(u.endsWith("/instances/split")){window.__posts.push(JSON.parse(o.body));return{ok:true,json:async()=>({correction_id:"c009",attach:{attached:true},state:{...state,books:[]}})}}return _f(u,o)};state={...emptyState,totals:{book_count:1},books:[{id:"bk-f00001-o000",status:"unidentified",sightings:9,member_group_ids:["bk-f00001-o000","bk-f00007-o003"],merge_evidence_kinds:["cross_segment_registration"],merge_evidence:[{kind:"cross_segment_registration",groups:["bk-f00001-o000","bk-f00007-o003"],frame_ids:["f00001","f00007"],inliers:66,appearance_score:0.71}],evidence:{frame_id:"f00001",bbox_xyxy:[0,0,10,10],frame_width_px:100,frame_height_px:100,timestamp_s:1}}]};renderInventory()');
 const html=e.elements.get('#books').innerHTML;assert(html.includes('merged from 2 track groups (cross segment registration)'));assert(html.includes('data-split-member="bk-f00007-o003"'));assert(html.includes('66 inliers, appearance 0.71'));assert(html.includes('9 sightings'));
 await e.run('splitMember("bk-f00007-o003")');assert.equal(e.run("JSON.stringify(window.__posts)"),JSON.stringify([{member_group_id:"bk-f00007-o003"}]));
 e.run('state={...emptyState,books:[{id:"bk-f00001-o000",status:"unidentified",member_group_ids:["bk-f00001-o000"],evidence:{}}]};renderInventory()');assert(!e.elements.get('#books').innerHTML.includes('merged from'));
});
test('Books table labels a converted price with its native amount and ECB rate; a native US price carries no label',()=>{
 const e=env();
 e.run('sessionId="s1";state={...emptyState,totals:{book_count:2},books:[{id:"bk-1",status:"identified",title:"Acosada",replacement_cost:{amount:13.52,currency:"USD",converted:true,original_amount:12,original_currency:"EUR",fx:{rate:1.1269,rate_date:"2026-10-06"},source:"Iberlibro (AbeBooks Spain) seller in Spain",url:"https://www.iberlibro.com/x",retrieved_at:"2026-10-07",condition_assumed:"New"},used_value:{amount:7,currency:"USD",converted:false,source:"AbeBooks seller",url:"https://www.abebooks.com/x",retrieved_at:"2026-10-07",condition_assumed:"Used"},evidence:{}}]};renderInventory()');
 const html=e.elements.get('#books').innerHTML;
 assert(html.includes('13.52 USD<br><small><b>converted</b> from EUR 12 at 1.1269 (ECB 2026-10-06)</small>'));
 assert(html.includes('7 USD<br><a href="https://www.abebooks.com/x"'));assert.equal((html.match(/<b>converted<\/b>/g)||[]).length,1);
});
test('Books table names the catalogue that identified a book and links to its record',async()=>{
 const e=env();e.run('sessionId="s1";state={...emptyState,totals:{book_count:2},books:[{id:"bk-1",status:"identified",title:"Rayuela",author:"Cortázar",catalogue:"google_books",identity_source_url:"https://books.google.com/books?id=RaYuElA12345",evidence:{}},{id:"bk-2",status:"identified",title:"Ficciones",catalogue:"open_library",identity_source_url:"https://openlibrary.org/works/OL1W",evidence:{}},{id:"bk-3",status:"unidentified",title:null,catalogue:null,evidence:{}}]};renderInventory()');
 const html=e.elements.get('#books').innerHTML;assert(html.includes('Google Books <a href="https://books.google.com/books?id=RaYuElA12345"'));assert(html.includes('Open Library <a href="https://openlibrary.org/works/OL1W"'));assert.equal((html.match(/Open Library|Google Books/g)||[]).length,2);
});
test('new sweep closes original socket and rejects late messages',async()=>{const e=env();await e.run('connectLive()');const old=e.sockets[0];await e.run('createSession(false)');old.onmessage({data:JSON.stringify({type:'transcript',session_id:'initial',id:'late',speaker:'Claimant',text:'Old'})});await e.run('sendClaimantTurn("Second")');assert(old.closed);assert.equal(e.sockets.length,2);assert.equal(e.run('state.transcript.length'),1);assert.equal(old.sent,undefined);});
test('New sweep asks before leaving a sweep that is still processing, and only then starts the new one',async()=>{
 const e=env();e.run('window.__asked=[];window.__calls=[];window.confirm=m=>{window.__asked.push(m);return window.__answer};const _f=fetch;fetch=async(u,o)=>{window.__calls.push((o&&o.method||"GET")+" "+u);if(u.endsWith("/processing"))return{ok:true,json:async()=>window.__info};return _f(u,o)}');
 e.run('window.__info={processing:true,run_id:"sweep_old",phase:"finalizing",message:"packet is still being written"};window.__answer=false');
 await e.run('requestNewSweep()');
 assert.equal(e.run('window.__asked.length'),1);assert.match(e.run('window.__asked[0]'),/packet is still being written/);
 assert.equal(e.run('window.__calls.filter(c=>c.startsWith("DELETE")||c==="POST "+API_ORIGIN+"/api/sessions").length'),0);   // declined: nothing deleted, no new session
 assert.equal(e.run('sessionId'),'initial');
 e.run('window.__answer=true');await e.run('requestNewSweep()');
 assert.equal(e.run('window.__asked.length'),2);
 assert(e.run('window.__calls.some(c=>c.startsWith("DELETE")&&c.endsWith("/api/sessions/initial"))'));assert.notEqual(e.run('sessionId'),'initial');
 assert.match(e.run('state.transcript.map(t=>t.text).join("|")'),/previous sweep \(sweep_old\) keeps finishing its packet and maps in the background/);
});
test('New sweep does not ask when nothing is processing, and still works when the status cannot be read',async()=>{
 const e=env();e.run('window.__asked=0;window.confirm=()=>{window.__asked++;return true};window.__info={processing:false};const _f=fetch;fetch=async(u,o)=>u.endsWith("/processing")?(window.__fail?{ok:false,json:async()=>({detail:"gone"})}:{ok:true,json:async()=>window.__info}):_f(u,o)');
 await e.run('requestNewSweep()');assert.equal(e.run('window.__asked'),0);assert.notEqual(e.run('sessionId'),'initial');
 const e2=env();e2.run('window.__asked=0;window.confirm=()=>{window.__asked++;return false};window.__fail=true;const _f=fetch;fetch=async(u,o)=>u.endsWith("/processing")?{ok:false,json:async()=>({detail:"gone"})}:_f(u,o)');
 await e2.run('requestNewSweep()');assert.equal(e2.run('window.__asked'),0);assert.notEqual(e2.run('sessionId'),'initial');
});
test('concurrent connects share a socket',async()=>{const e=env();await Promise.all([e.run('connectLive()'),e.run('connectLive()')]);assert.equal(e.sockets.length,1);e.run('disconnectLive()');});
test('reconnect uses the same sweep session',async()=>{const e=env();await e.run('connectLive()');e.run('stopLiveVoice()');await e.run('connectLive()');assert.equal(e.sockets[0].url,e.sockets[1].url);e.run('disconnectLive()');});
test('interruption stops all scheduled audio',async()=>{const e=env();await e.run('connectLive()');e.run('playPcm24(btoa("\\0\\0".repeat(24000)));playPcm24(btoa("\\0\\0".repeat(24000)))');e.sockets[0].onmessage({data:JSON.stringify({type:'interrupted',session_id:'initial'})});assert.equal(e.sources.filter(x=>x.stopped).length,2);e.run('disconnectLive()');});
test('reset stops queued speech',async()=>{const e=env();e.run('playPcm24(btoa("\\0\\0".repeat(24000)))');await e.run('createSession(false)');assert(e.sources[0].stopped);});
test('typed interruption stops playback before sending the new turn',async()=>{
 const e=env();await e.run('connectLive()');e.run('playPcm24(btoa("\\0\\0".repeat(24000)))');
 const socket=e.sockets[0],send=socket.send.bind(socket);socket.send=message=>{assert(e.sources[0].stopped);send(message)};
 await e.run('sendClaimantTurn("Stop, please wait.")');assert.equal(socket.sent.at(-1).text,'Stop, please wait.');e.run('disconnectLive()');
});
test('disconnect releases microphone and camera',async()=>{const e=env();await e.run('connectLive()');e.run('let stopped=0;isRecording=true;audioStream={getTracks:()=>[{stop(){stopped++}}]};cameraStream={getTracks:()=>[{stop(){stopped++}}]};');e.sockets[0].close();assert.equal(e.run('stopped'),2);assert.equal(e.run('isRecording'),false);assert.equal(e.run('cameraStream'),null);});
test('state merge preserves final turns not yet in snapshot',()=>{const e=env();assert.equal(e.run('mergeTranscript([{id:"1",speaker:"Agent",text:"Earlier"}],[{id:"2",speaker:"Claimant",text:"Correction",streaming:false}]).length'),2);});
test('short answers are distinct; repeated final ID is idempotent',()=>{const e=env();e.run('upsertStreamingTurn("Claimant","No",true,"a");upsertStreamingTurn("Claimant","No",true,"b");upsertStreamingTurn("Claimant","No",true,"b")');assert.equal(e.run('state.transcript.length'),2);});
test('finished and failed tools clear writing indicator',()=>{const e=env();for(const phase of ['done','error','cancelled']){e.run(`state={...emptyState,tool_activity:[{id:"1",name:"record_correction",phase:"${phase}"}]};setWriting(true);renderTeam()`);assert.equal(e.run('writing'),false);}});
test('errors release busy state',async()=>{const e=env();await e.run('connectLive()');e.run('processing=true;render()');e.sockets[0].onmessage({data:JSON.stringify({type:'error',session_id:'initial',message:'failed'})});assert.equal(e.run('writing'),false);e.run('disconnectLive()');});
test('camera activation and stop send explicit mode changes',async()=>{
 const e=env();e.run('let tracksStopped=0;const videoTrack={stop(){tracksStopped++},addEventListener(){}};navigator.mediaDevices={getUserMedia:async()=>({getTracks:()=>[videoTrack],getVideoTracks:()=>[videoTrack]})}');
 await e.run('startCamera()');assert.deepEqual(e.sockets[0].sent,[{type:'camera_state',enabled:true}]);
 e.run('stopCamera()');assert.deepEqual(e.sockets[0].sent.at(-1),{type:'camera_state',enabled:false});assert.equal(e.run('tracksStopped'),1);e.run('disconnectLive()');
});
test('denied camera does not announce camera on',async()=>{const e=env();e.run('navigator.mediaDevices={getUserMedia:async()=>{throw new Error("Permission denied")}}');await e.run('startCamera()');assert.equal(e.sockets[0].sent,undefined);assert.equal(e.run('cameraStream'),null);e.run('disconnectLive()');});
// ---- new for the sweep app
test('camera requests high resolution, evidence frame is full-size JPEG and distinct from the 512px voice frame',async()=>{
 const e=env();e.run('let constraints;const t={stop(){},addEventListener(){}};navigator.mediaDevices={getUserMedia:async(c)=>{constraints=c;return {getTracks:()=>[t],getVideoTracks:()=>[t]}}}');
 await e.run('startCamera()');assert.equal(e.run('constraints.video.width.ideal'),1920);
 e.run('sendVoiceFrame();sendEvidenceFrame()');const sent=e.sockets[0].sent;
 assert.equal(sent.at(-2).type,'video');assert.equal(sent.at(-1).type,'evidence_frame');assert.equal(sent.at(-1).source,'camera');
 assert.equal(e.run('frameCanvas.width'),512);assert.equal(e.run('evidenceCanvas.width'),1920);assert.equal(e.run('evidenceCanvas.height'),1080);
 e.run('disconnectLive()');
});
test('replay uses the same frame messages as the camera and carries the video clock',async()=>{
 const e=env();await e.run('startReplay("YugoTest1")');
 e.run('sendEvidenceFrame()');const m=e.sockets[0].sent.at(-1);
 assert.equal(m.type,'evidence_frame');assert.equal(m.source,'replay');assert.equal(m.t_s,3.5);
 assert.deepEqual(e.sockets[0].sent[0],{type:'camera_state',enabled:true});
 e.run('stopCamera()');assert.deepEqual(e.sockets[0].sent.at(-1),{type:'camera_state',enabled:false});assert.equal(e.run('replayActive'),false);e.run('disconnectLive()');
});
test('replay and camera are mutually exclusive',async()=>{const e=env();await e.run('startReplay("YugoTest1")');e.run('navigator.mediaDevices={getUserMedia:async()=>{throw new Error("should not be called")}}');await e.run('startCamera()');assert.equal(e.run('cameraStream'),null);e.run('disconnectLive()');});
test('unknown money and measurements render as "unknown" (never 0) once the sweep is final',()=>{
 const e=env();e.run('state={...emptyState,status:"finished",totals:{book_count:3,books_identified:1,books_unidentified:2,item_count:1}};renderInventory()');
 const html=e.elements.get('#totals').innerHTML;assert(html.includes('Replacement / used value'));assert(/<b>unknown<\/b><span>Replacement/.test(html));assert(/<b>unknown<\/b><span>Shelf run/.test(html));assert(/<b>unknown<\/b><span>Room floor/.test(html));
 assert(/<b>1<\/b><span>Identified/.test(html));
});
test('values that only exist after End sweep read "pending" while capturing or finalizing, with a not-final mark',()=>{
 const e=env();for(const status of ['capturing','finalizing']){
  e.run(`state={...emptyState,status:"${status}",totals:{book_count:3,books_identified:1,books_unidentified:2,item_count:1}};renderInventory()`);
  const html=e.elements.get('#totals').innerHTML;
  for(const label of ['Identified','Unidentified','Needs appraisal','Replacement','Shelf run','Room floor'])assert(new RegExp('<b>pending</b><span>'+label).test(html),status+' '+label);
  assert(/<b>3<\/b><span>Books/.test(html));assert(!html.includes('✓'));assert(html.includes('◐'));assert(!/<b>unknown/.test(html));}
});
test('final tiles carry the done mark; a failed sweep never shows pending',()=>{
 const e=env();e.run('state={...emptyState,status:"finished",totals:{book_count:3}};renderInventory()');
 assert(e.elements.get('#totals').innerHTML.includes('✓'));assert(!e.elements.get('#totals').innerHTML.includes('pending'));
 e.run('state={...emptyState,status:"failed",finalize:{error:"boom"},totals:{book_count:3}};renderInventory()');
 const html=e.elements.get('#totals').innerHTML;assert(!html.includes('pending'));assert(html.includes('✕'));
});
test('sweep status: recording, processing with the finalize stage and title progress, done with report link, failed with reason',()=>{
 const e=env();const phase=(st,live,fail)=>JSON.stringify(e.run(`sweepPhase(${st},${live},${fail||null})`));
 let v=JSON.parse(phase('{...emptyState,progress:{frames_saved:12,keyframes_selected:3,stages_in_flight:1}}',true));
 assert.equal(v.phase,'recording');assert.equal(v.icon,'●');assert.equal(v.word,'Recording');assert(v.detail.includes('12 frames'));
 v=JSON.parse(phase('{...emptyState,status:"finalizing",finalize:{stage:"reading titles",read:{attempted:120,total:400}}}',false));
 assert.equal(v.phase,'processing');assert.equal(v.icon,'◐');assert.equal(v.word,'Sweep finished — processing');assert.equal(v.detail,'reading titles 120/400');
 v=JSON.parse(phase('{...emptyState,status:"finalizing",finalize:{stage:"pricing"}}',false));assert.equal(v.detail,'pricing');
 v=JSON.parse(phase('{...emptyState,status:"finalizing"}',false));assert.equal(v.phase,'processing');
 v=JSON.parse(phase('{...emptyState,status:"finished",finalization:{packet_status:"partial",finalization_s:41.2}}',false));
 assert.equal(v.phase,'done');assert.equal(v.icon,'✓');assert.equal(v.word,'Done');assert(v.detail.includes('packet ready'));
 v=JSON.parse(phase('{...emptyState,status:"failed",finalize:{error:"OSError: disk full"}}',false));
 assert.equal(v.phase,'failed');assert.equal(v.icon,'✕');assert(v.detail.includes('disk full'));
 v=JSON.parse(phase('{...emptyState,status:"capturing"}',false,'"End request refused"'));assert.equal(v.phase,'failed');assert(v.detail.includes('End request refused'));
 assert.equal(JSON.parse(phase('{...emptyState}',false)).phase,'idle');
});
test('status block shows the report link only when done, and finalize events update the stage live',async()=>{
 const e=env();await e.run('connectLive()');e.run('sessionId="initial"');
 e.run('state={...emptyState,status:"finalizing"};render()');
 assert.equal(e.elements.get('#sweepStatus').querySelector('.ss-word')===undefined,false);
 e.sockets[0].onmessage({data:JSON.stringify({type:'finalize',session_id:'initial',stage:'reading titles',read:{attempted:5,total:10}})});
 assert.equal(e.run('sweepPhase(state,false,null).detail'),'reading titles 5/10');
 e.run('state={...emptyState,status:"finished"};render()');assert.equal(e.elements.get('#ssReport').hidden,false);assert(e.elements.get('#ssReport').href.endsWith('/api/sessions/initial/report'));
 e.run('state={...emptyState,status:"capturing"};render()');assert.equal(e.elements.get('#ssReport').hidden,true);
 e.run('disconnectLive()');
});
test('inventory view shows counts from server state and excluded lines',()=>{
 const e=env();e.run('sessionId="s1";state={...emptyState,totals:{book_count:2,user_excluded_count:1},books:[{id:"bk-001",status:"unidentified",title:null,author:null,visible_text:null,language:null,evidence:{frame_id:"f00001",bbox_xyxy:[0,0,10,10],frame_width_px:100,frame_height_px:100,timestamp_s:1}}],excluded_by_user:[{instance_id:"bk-002",reason:"not mine",correction_id:"c001"}]};renderInventory()');
 assert(e.elements.get('#books').innerHTML.includes('/api/sessions/s1/frames/f00001.jpg'));assert(e.elements.get('#excluded').innerHTML.includes('bk-002'));
});
test('books table zooms to the detected book and outlines its exact polygon (debugger style), and toggles to the whole frame',()=>{
 const e=env();e.run('sessionId="s1";state={...emptyState,books:[{id:"bk-001",status:"unidentified",evidence:{frame_id:"f00001",bbox_xyxy:[400,100,460,400],polygon_xy:[[410,100],[460,110],[450,400],[400,390]],frame_width_px:1000,frame_height_px:500,timestamp_s:1}}]};renderInventory()');
 const zoom=e.elements.get('#books').innerHTML;
 // bbox 60x300 padded by 25% of the long side (75 px) -> region [325,25,535,475] as the SVG viewBox; the polygon (not the bbox) is drawn
 assert(zoom.includes('viewBox="325 25 210 450"'));assert(zoom.includes('aspect-ratio:210 / 450'));assert(zoom.includes('points="410.0,100.0 460.0,110.0 450.0,400.0 400.0,390.0"'));
 assert(zoom.includes('fill-rule="evenodd"'));assert(zoom.includes('whole frame'));
 e.run('wholeFrameBooks.add("bk-001");renderInventory()');
 const whole=e.elements.get('#books').innerHTML;
 assert(whole.includes('viewBox="0 0 1000 500"'));assert(whole.includes('points="410.0,100.0'));assert(whole.includes('zoom to book'));
 // no polygon -> the bbox corners are outlined; a book near the frame edge keeps its region inside the frame; no box -> plain thumbnail
 e.run('wholeFrameBooks.clear();state={...emptyState,books:[{id:"bk-002",evidence:{frame_id:"f00002",bbox_xyxy:[0,0,50,200],frame_width_px:1000,frame_height_px:500}},{id:"bk-003",evidence:{frame_id:"f00003",bbox_xyxy:null,frame_width_px:1000,frame_height_px:500}}]};renderInventory()');
 const edge=e.elements.get('#books').innerHTML;assert(edge.includes('viewBox="0 0 100 250"'));assert(edge.includes('points="0.0,0.0 50.0,0.0 50.0,200.0 0.0,200.0"'));assert(edge.includes('f00003.jpg'));
});
test('end sweep sends one message and stops capture',async()=>{
 const e=env();await e.run('connectLive()');e.run('endSweep()');assert.deepEqual(e.sockets[0].sent.at(-1),{type:'end_sweep'});e.run('disconnectLive()');
});
test('nudge events are shown without replacing server state',async()=>{
 const e=env();await e.run('connectLive()');e.sockets[0].onmessage({data:JSON.stringify({type:'nudge',session_id:'initial',code:'blur',message:'Slow down'})});
 assert(e.elements.get('#nudges').innerHTML.includes('Slow down'));e.run('disconnectLive()');
});
// ---- measurement panel (human inputs only; the client never estimates a dimension)
function calEnv(){const e=env();e.run('const _img=document.querySelector("#calImg");_img.naturalWidth=1920;_img.naturalHeight=1080;_img.getBoundingClientRect=()=>({left:10,top:20,width:960,height:540});window.__posts=[];const _f=fetch;fetch=async(u,o)=>{if(o&&o.method==="POST"){window.__posts.push({u,body:JSON.parse(o.body)});}return _f(u,o)};sessionId="s1";document.querySelector("#calFrame").value="f00003";document.querySelector("#calMode").value="plane_quad";document.querySelector("#calUnit").value="m";document.querySelector("#calTol").value="3";document.querySelector("#calWidth").value="1.8";document.querySelector("#calHeight").value="2";document.querySelector("#calUtterance").value="about 1.8 by 2 m";');return e;}
test('taps are converted to native frame pixels and limited to the needed count',()=>{
 const e=calEnv();e.run('calClick({clientX:10,clientY:20});calClick({clientX:970,clientY:560});calClick({clientX:490,clientY:290});calClick({clientX:10,clientY:560});calClick({clientX:1,clientY:1})');
 assert.equal(JSON.stringify(e.run('calPoints')),JSON.stringify([[0,0],[1920,1080],[960,540],[0,1080]]));
});
test('calibration submit sends plane_quad with units, tolerance and the human utterance; wrong point count is refused',async()=>{
 const e=calEnv();e.run('calClick({clientX:10,clientY:20})');await e.run('submitCalibration()');assert.equal(e.run('window.__posts.length'),0);assert(e.elements.get('#calMsg').textContent.includes('Tap exactly 4'));
 e.run('calPoints=[[1,1],[100,1],[100,100],[1,100]]');await e.run('submitCalibration()');
 const p=e.run('window.__posts[0]');assert(p.u.endsWith('/api/sessions/s1/calibrations'));
 assert.equal(p.body.kind,'plane_quad');assert.equal(p.body.width,1.8);assert.equal(p.body.height,2);assert.equal(p.body.unit,'m');assert.equal(p.body.tolerance_pct,3);assert.equal(p.body.utterance,'about 1.8 by 2 m');assert.equal(p.body.frame_id,'f00003');
});
test('wall marks go to room evidence with the floor-to-ceiling declaration',async()=>{
 const e=calEnv();e.run('document.querySelector("#calMode").value="wall";document.querySelector("#calWall").value="A";document.querySelector("#calCeiling").checked=true;calPoints=[[1,1],[100,1],[100,100],[1,100]]');await e.run('submitCalibration()');
 const p=e.run('window.__posts[0]');assert(p.u.endsWith('/room-evidence/walls'));assert.equal(p.body.wall_id,'A');assert.equal(p.body.spans_floor_to_ceiling,true);assert.equal(p.body.quad_px.length,4);
});
test('ceiling height is posted as an approximate user-supplied statement',async()=>{
 const e=calEnv();e.run('document.querySelector("#ceilValue").value="2.5"');await e.run('submitCeiling()');
 const p=e.run('window.__posts[0]');assert(p.u.endsWith('/room-evidence/height'));assert.equal(p.body.value,2.5);assert.equal(p.body.approximate,true);
});
test('missing coverage is shown live with the question to ask; unknown shelf run/room area stay unknown once final',()=>{
 const e=calEnv();e.run('state={...emptyState,status:"finished",totals:{book_count:1},room:{},measurement:{items:[{item:"ceiling_height",present:false,how:"Ask the ceiling height"},{item:"wall_faces_2_adjacent",present:true}],calibrations:[{calibration_id:"cal-001",kind:"plane_quad",status:"invalid",problems:["quad non-convex"],plane:{id:"p"}}],pending_dimensions:[],scale:{summary:{measured:0,abstained:3},shelf_run_m:null},room:{status:"abstained",abstention_reasons:["no_wall_evidence_supplied"]},validation:"software_only_unvalidated"},frames_index:[{frame_id:"f00003"}]};renderMeasurement();renderInventory()');
 const html=e.elements.get('#measureStatus').innerHTML;assert(html.includes('Ask the ceiling height'));assert(html.includes('Shelf run withheld'));assert(html.includes('quad non-convex'));assert(html.includes('software_only_unvalidated'));
 const t=e.elements.get('#totals').innerHTML;assert(/<b>unknown<\/b><span>Shelf run/.test(t));assert(/<b>unknown<\/b><span>Room floor area/.test(t));
});
test('automatic measurement: spine sizes with +-, PRIOR scale source, room reasons in plain words, unknown never 0',()=>{
 const e=env();const v={available:true,measure_dir:"sweep_x_measure_v1",attached_to_packet:true,validation:"software_only_unvalidated",
  scale:{source:"standard_format_prior",label:"PRIOR (assumed 22 cm median book; not a measurement)",not_a_measurement:true,relative_uncertainty_pct:15,why_not_catalogue:"no_identified_books"},
  summary:{books:2,measured:1,with_thickness:0,abstained:1},shelf:{shelf_run_m:1.46,rows:2},
  books:[{group_id:"bk-001",title:"Rayuela",status:"measured",spine_height_cm:26.9,height_pm_cm:4.2,spine_thickness_cm:null,thickness_reason:"thickness withheld: spines too narrow in the video to measure a thickness"},
         {group_id:"bk-002",visible_text:"[?]",status:"abstained",reason:"no usable view: cut off by the frame edge in every view"}],
  room:{status:"abstained",dimensions:{height_m:{value:null,unit:"m",reason:"needs both the floor line and the ceiling line in the video"},width_m:{value:null,unit:"m",reason:"needs both side walls in the video"},depth_m:{value:null,unit:"m",reason:"x"}},
        areas:{floor_area_m2:{value:null,unit:"m2",reason:"needs all four walls in the video"},wall_area_m2:{value:null,unit:"m2",reason:"y"},shelved_wall_area_m2:{value:0.63,unit:"m2",pct:30.4}},
        reasons:["floor line not filmed (no floor surface in the 3D map)"],film:["Tilt down so the floor along the wall is in view."]}};
 const html=e.run(`renderAutoMeasure(${JSON.stringify(v)})`);
 assert(html.includes('26.9 cm ± 4.2'));assert(html.includes('PRIOR'));assert(html.includes('not a measurement'));assert(html.includes('floor line not filmed'));
 assert(html.includes('spines too narrow'));assert(html.includes('cut off by the frame edge'));assert(html.includes('0.63 m2'));assert(html.includes('1.46 m'));
 assert(!/>0 m|>0 cm|: 0 /.test(html));assert(!/measure (the|it|your)|tape|mark the/i.test(html));
 assert.equal(e.run(`renderAutoMeasure({available:false,reason:"no 3D map of this sweep yet"})`),'<p>no 3D map of this sweep yet</p>');
});
// ---- unified site: navigation must not disturb the active session, scan identity stays explicit
test('the scan list is exactly the registry (no aliases or preview files)',async()=>{
 const e=env();await e.run('loadScans()');const html=e.elements.get('#replaySelect').innerHTML;
 assert(html.includes('YugoTest1')&&html.includes('YugoTest2'));assert(!/IMG_|h264|\.MOV/i.test(html));assert.equal((html.match(/<option/g)||[]).length,2);
});
test('replay binds the session to the scan before capture and uses the registered preview URL',async()=>{
 const e=env();const posts=[];e.run('window.__posts=[];const _f=fetch;fetch=async(u,o)=>{if(o&&o.method==="POST")window.__posts.push(u);return _f(u,o)}');
 await e.run('startReplay("YugoTest2")');assert(e.run('window.__posts[0]').endsWith('/api/sessions/initial/scan'));
 assert(e.elements.get('#cameraPreview').src.endsWith('/replay/scan/YugoTest2'));assert.equal(e.run('replayLabel'),'YugoTest2');e.run('disconnectLive()');
});
test('switching views keeps the socket, session, replay and calibration state',async()=>{
 const e=env();await e.run('startReplay("YugoTest1")');const socket=e.sockets[0];
 e.run('calPoints=[[1,1],[2,2]];iframeReady=async()=>({openFrameSource:(s,f)=>{window.__fs=[s,f]},catalogue:async()=>{},scans:[]})');
 const before=e.sockets.length;
 for(const v of ['frames','debug','measure','sweep'])await e.run(`showView("${v}")`);
 assert.equal(e.sockets.length,before);assert(!socket.closed);assert.equal(e.run('sessionId'),'initial');assert.equal(e.run('replayActive'),true);assert.equal(e.run('calPoints.length'),2);
 assert.equal(socket.sent.filter(m=>m.type==='camera_state'&&m.enabled===false).length,0);
 e.run('disconnectLive()');
});
test('only the selected view is visible and the recording strip stays truthful on other views',async()=>{
 const e=env();await e.run('startReplay("YugoTest1")');e.run('state={...state,identity:{session_id:"initial",run_dir_name:"sweep_initial",scan:{label:"YugoTest1",sha256:"c4041f08aa"},frames_saved:7}};iframeReady=async()=>({openFrameSource(){}})');
 await e.run('showView("frames")');assert.equal(e.elements.get('#view-frames').hidden,false);assert.equal(e.elements.get('#view-sweep').hidden,true);
 const txt=e.elements.get('#recIdentity').textContent;assert(txt.includes('session initial')&&txt.includes('scan YugoTest1')&&txt.includes('recording continues'));
 assert(e.elements.get('#recState').textContent.includes('REPLAY RECORDING'));e.run('disconnectLive()');
});
test('frame links open the same frame of the ACTIVE session; debugger links are pending until a run exists',async()=>{
 const e=env();e.run('sessionId="s1";state={...emptyState,status:"capturing",identity:{scan:{label:"YugoTest1"},run_id:"sweep_s1"},progress:{merges:0}};iframeReady=async()=>({openFrameSource:(s,f)=>{window.__fs=[s,f]}})');
 const line='{evidence:{frame_id:"f00003",observation_ids:["f00003-o001"]}}';
 let html=e.run(`evidenceLinks(${line})`);assert(html.includes('data-open-frame="f00003"'));assert(html.includes('debugger pending'));assert(!html.includes('data-open-debug'));
 e.run('state.progress.merges=2');html=e.run(`evidenceLinks(${line})`);assert(html.includes('data-open-debug="f00003-o001"'));
 e.run('showView("frames",{source:"session:s1",frame:"f00003"})');return new Promise(r=>setTimeout(()=>{assert.equal(JSON.stringify(e.run('window.__fs')),JSON.stringify(['session:s1','f00003']));r()},20));
});
test('hash routing parses view and parameters',()=>{
 const e=env();e.run('location.hash="#frames?source=run:yugotest2_frames_v1&frame=f00010"');
 const p=JSON.parse(JSON.stringify(e.run('parseHash()')));assert.equal(p.view,'frames');assert.equal(p.params.source,'run:yugotest2_frames_v1');assert.equal(p.params.frame,'f00010');
 e.run('location.hash="#bogus"');assert.equal(e.run('parseHash().view'),'sweep');
});
test('replay is driven by seeking (no play(), so a hidden tab cannot pause it) and stops at the end of the video',async()=>{
 const e=env();await e.run('startReplay("YugoTest1")');
 e.run('window.__seeks=[];seekTo=async(v,t)=>{window.__seeks.push(t)};cameraPreview.duration=0.5;cameraPreview.play=()=>{throw new Error("play must not be used")}');
 await e.run('stepReplay()');await e.run('stepReplay()');await e.run('stepReplay()');
 assert.equal(JSON.stringify(e.run('window.__seeks')),JSON.stringify([0,0.2,0.4]));   // 5 fps of video time
 const frames=e.sockets[0].sent.filter(m=>m.type==='evidence_frame');assert.equal(frames.length,3);assert(frames.every(m=>m.source==='replay'));
 await e.run('stepReplay()');assert.equal(e.run('replayActive'),false);assert.equal(e.sockets[0].sent.at(-1).enabled,false);e.run('disconnectLive()');
});
test('talk fails clearly BEFORE opening the microphone when hosted voice is unavailable',async()=>{
 const e=env();e.run('let micCalls=0;navigator.mediaDevices={getUserMedia:async()=>{micCalls++;return {getTracks:()=>[]}}};window.AudioContext=function(){};');
 await e.run('connectLive()');e.sockets[0].onmessage({data:JSON.stringify({type:'ready',session_id:'initial',voice:false,voice_reason:'LIBRARY_LIVE_MODEL is not set; no Live model id is assumed'})});
 e.run('voiceAvailable=false;voiceReason="LIBRARY_LIVE_MODEL is not set; no Live model id is assumed"');
 await e.run('startLiveVoice()');assert.equal(e.run('micCalls'),0);assert.equal(e.run('isRecording'),false);
 const sys=JSON.stringify(e.run('state.transcript.filter(t=>t.speaker==="System").map(t=>t.text)'));assert(sys.includes('Voice is unavailable')&&sys.includes('LIBRARY_LIVE_MODEL'));assert.equal(e.elements.get('#callStatus').textContent,'Voice unavailable');
 e.run('disconnectLive()');
});
test('held-out truth needs the truth mode and exactly one tap, posts to /truth/spines only and never to calibrations',async()=>{
 const e=calEnv();e.run('document.querySelector("#truthName").value="Alex";document.querySelector("#truthHeight").value="23.5";document.querySelector("#truthThickness").value="3.1";document.querySelector("#truthUnit").value="cm";document.querySelector("#truthLang").value="es";document.querySelector("#truthDesc").value="unit 2";calPoints=[[400,300]]');
 await e.run('saveTruthSpine()');assert.equal(e.run('window.__posts.length'),0);assert(e.elements.get('#truthMsg').textContent.includes('HELD-OUT'));
 e.run('document.querySelector("#calMode").value="truth_spine"');await e.run('saveTruthSpine()');
 const p=e.run('window.__posts[0]');assert(p.u.endsWith('/api/sessions/s1/truth/spines'));assert.equal(p.body.height,23.5);assert.equal(p.body.thickness,3.1);assert.equal(p.body.annotator,'Alex');assert.equal(p.body.language,'es');assert.equal(p.body.tap_px.length,2);assert.equal(p.body.frame_id,'f00003');
 e.run('window.__posts.length=0;calPoints=[[1,1]]');await e.run('submitCalibration()');assert.equal(e.run('window.__posts.length'),0);assert(e.elements.get('#calMsg').textContent.includes('held-out truth'));
});
test('truth room reading posts to /truth/room',async()=>{
 const e=calEnv();e.run('document.querySelector("#truthName").value="Alex";document.querySelector("#truthRoomL").value="4.2";document.querySelector("#truthRoomW").value="3.1";document.querySelector("#truthRoomH").value="2.5";document.querySelector("#truthRoomUnit").value="m";document.querySelector("#truthRoomNotes").value="door west"');
 await e.run('saveTruthRoom()');const p=e.run('window.__posts[0]');assert(p.u.endsWith('/truth/room'));assert.equal(p.body.length,4.2);assert.equal(p.body.unit,'m');
});
// ---- automatic 2D panorama + 3D map after End sweep
test('the maps status stays hidden until the sweep has finished',()=>{
 const e=env();e.run('sessionId="s1";state={...emptyState,status:"capturing"};renderSweepBuilds()');assert.equal(e.elements.get('#sweepBuilds').hidden,true);
 e.run('state={...emptyState,status:"finalizing"};renderSweepBuilds()');assert.equal(e.elements.get('#sweepBuilds').hidden,true);
});
test('maps status while building: visible, processing, polled; links only for finished steps',()=>{
 const e=env();const post=(state,pano,recon)=>`state={...emptyState,status:"finished",post_sweep:{state:"${state}",run_id:"sweep_s1",scan_label:"IESTest3",steps:{panorama:{state:"${pano}"},recon3d:{state:"${recon}"}},summary:"x"}};renderSweepBuilds()`;
 e.run('sessionId="s1"');
 e.run('state={...emptyState,status:"finished",post_sweep:null};renderSweepBuilds()');assert.equal(e.elements.get('#sweepBuilds').hidden,false);assert.equal(e.elements.get('#sweepBuilds').className,'sweep-status processing');assert.equal(e.run('buildPoll!==null'),true);   // queued: polled until the server reports
 e.run(post('building_2d','running','pending'));assert.equal(e.elements.get('#sbPano').hidden,true);assert.equal(e.elements.get('#sbRecon').hidden,true);
 e.run(post('building_3d','done','running'));assert.equal(e.elements.get('#sbPano').hidden,false);assert.equal(e.elements.get('#sbRecon').hidden,true);assert.equal(e.elements.get('#sweepBuilds').className,'sweep-status processing');
 e.run(post('done','done','done'));assert.equal(e.elements.get('#sweepBuilds').className,'sweep-status done');assert.equal(e.elements.get('#sbRecon').hidden,false);assert.equal(e.run('buildPoll'),null);   // terminal: polling stops
});
test('a failed build is shown as failed and stops polling; the map links open the Debugger on that run and section',()=>{
 const e=env();e.run('sessionId="s1";state={...emptyState,status:"finished",post_sweep:{state:"failed",run_id:"sweep_s1",scan_label:"Other runs (no canonical recording)",steps:{panorama:{state:"failed",error:"boom"},recon3d:{state:"done"}},summary:"2D panorama failed: boom"}};renderSweepBuilds()');
 assert.equal(e.elements.get('#sweepBuilds').className,'sweep-status failed');assert.equal(e.run('buildPoll'),null);assert.equal(e.elements.get('#sbPano').hidden,true);assert.equal(e.elements.get('#sbRecon').hidden,false);
 assert(src.includes('data-open-map')||fs.readFileSync(path.join(__dirname,'../app/static/index.html'),'utf8').includes('data-open-map="3d"'));
 assert(src.includes('section: m.dataset.openMap'));
});
// ---- progress bars for the automatic / manual 2D + 3D builds (Sweep tab "Maps" line and the Debugger next to the Build buttons)
test('Sweep tab Maps line: overall progress bar, stage label and ETA while building; queued steps show their position; hidden when not building',()=>{
 const e=env();e.run('sessionId="s1"');
 const post=(o)=>`state={...emptyState,status:"finished",post_sweep:${JSON.stringify(o)}};renderSweepBuilds()`;
 e.run(post({state:'building_3d',run_id:'r',scan_label:'x',overall:{fraction:0.63,eta_s:30},steps:{panorama:{state:'done',markers:155,book_count:155},recon3d:{state:'running',progress:{label:'registering frames (SfM) 40/61',fraction:0.55}}}}));
 assert.equal(e.elements.get('#sbBar').hidden,false);assert.equal(e.elements.get('#sbFill').style.width,'63%');
 const detail=e.run('mapsDetail(state.post_sweep)');
 assert(detail.includes('building 3D map: registering frames (SfM) 40/61 (55%)'));assert(detail.includes('2D panorama ready (155 markers = book_count 155)'));assert(detail.includes('63% overall'));assert(detail.includes('~30 s left'));
 const q=e.run(`mapsDetail(${JSON.stringify({state:'queued',overall:{fraction:0,eta_s:140},steps:{panorama:{state:'running',progress:{label:'aligning frames 20/60',fraction:0.2}},recon3d:{state:'queued',position:2}}})})`);
 assert(q.includes('3D map queued #2'));assert(q.includes('~2 min left'));
 e.run(post({state:'done',run_id:'r',scan_label:'x',overall:{fraction:1,eta_s:0},steps:{panorama:{state:'done'},recon3d:{state:'done'}}}));assert.equal(e.elements.get('#sbFill').style.width,'100%');assert.equal(e.elements.get('#sbBar').hidden,false);
 e.run('state={...emptyState,status:"capturing"};renderSweepBuilds()');assert.equal(e.elements.get('#sweepBuilds').hidden,true);
});
test('a measuring post-sweep state keeps polling and is labelled, not shown as failed',()=>{
 const e=env();e.run('sessionId="s1";state={...emptyState,status:"finished",post_sweep:{state:"measuring",run_id:"r",scan_label:"x",steps:{panorama:{state:"done"},recon3d:{state:"done"}},overall:{fraction:1,eta_s:0}}};renderSweepBuilds()');
 assert.equal(e.elements.get('#sweepBuilds').className,'sweep-status processing');assert.equal(e.run('buildPoll!==null'),true);
});
test('joining_3d (the 3D merge pass after the first map) keeps polling, is labelled and shows the join progress, not failed',()=>{
 const e=env();const post=(o)=>`sessionId="s1";state={...emptyState,status:"finished",post_sweep:${JSON.stringify(o)}};renderSweepBuilds()`;
 e.run('globalThis.__word={textContent:""};const _sb=document.querySelector("#sweepBuilds");_sb.querySelector=k=>k===".ss-word"?globalThis.__word:{textContent:""}');
 e.run(post({state:'joining_3d',run_id:'r',scan_label:'x',overall:{fraction:0.8,eta_s:120},steps:{panorama:{state:'done'},recon3d:{state:'done'}},merge3d:{state:'running',first_map:'r_recon3d_v1',progress:{label:'bridge matching',fraction:0.4}}}));
 assert.equal(e.elements.get('#sweepBuilds').className,'sweep-status processing');assert.equal(e.run('buildPoll!==null'),true);
 assert.equal(e.run('__word.textContent'),'Maps: joining the 3D map fragments');
 const d=e.run('mapsDetail(state.post_sweep)');assert.match(d,/joining 3D fragments: bridge matching \(40%\)/);assert.match(d,/80% overall/);
 assert.match(e.run(`mapsDetail(${JSON.stringify({state:'done',steps:{recon3d:{state:'done'}},merge3d:{state:'done',registered_ratio:0.795}})})`),/joined 3D map ready \(80% of frames\)/);
 assert.match(e.run(`mapsDetail(${JSON.stringify({state:'done',steps:{recon3d:{state:'done'}},merge3d:{state:'failed',error:'boom'}})})`),/first map stands\): boom/);
});
// ---- sharp live capture (iPhone Safari over HTTPS)
test('camera falls back through the constraint sets, asks for environment/1080p/60 fps first (?fps=30 opts out), and never retries a denied permission',async()=>{
 let e=env();e.run('const calls=[];const t={stop(){},addEventListener(){}};navigator.mediaDevices={getUserMedia:async(c)=>{calls.push(c);if(calls.length<3){const x=new Error("no match");x.name="OverconstrainedError";throw x}return {getTracks:()=>[t],getVideoTracks:()=>[t]}}}');
 await e.run('startCamera()');assert.equal(e.run('calls.length'),3);
 const first=e.run('calls[0].video');assert.equal(first.facingMode.ideal,'environment');assert.equal(first.width.ideal,1920);assert.equal(first.height.ideal,1080);assert.equal(first.frameRate.ideal,60);
 assert.equal(e.run('calls[2].video'),true);assert.equal(e.run('cameraStream!==null'),true);e.run('disconnectLive()');
 e=env();e.run('let n=0;navigator.mediaDevices={getUserMedia:async()=>{n++;const x=new Error("denied");x.name="NotAllowedError";throw x}}');
 await e.run('startCamera()');assert.equal(e.run('n'),1);assert.equal(e.run('cameraStream'),null);e.run('disconnectLive()');
});
test('the phone reports its camera label, stream settings and capture mode once, before the camera-on state is used',async()=>{
 const e=env();e.run('const t={label:"Back Camera",stop(){},addEventListener(){},getSettings(){return {width:1920,height:1080,frameRate:30,facingMode:"environment"}}};navigator.mediaDevices={getUserMedia:async()=>({getTracks:()=>[t],getVideoTracks:()=>[t]})};navigator.userAgent="iPhone Safari";window.isSecureContext=true');
 await e.run('startCamera()');const info=e.sockets[0].sent.find(m=>m.type==='capture_info');
 assert.equal(info.label,'Back Camera');assert.equal(info.width,1920);assert.equal(info.height,1080);assert.equal(info.frame_rate,30);assert.equal(info.facing_mode,'environment');
 assert.equal(info.capture_mode,'timer');assert.equal(info.jpeg_quality,0.95);assert.equal(info.secure_context,true);
 e.run('stopCamera()');assert.deepEqual(e.sockets[0].sent.at(-1),{type:'camera_state',enabled:false});e.run('disconnectLive()');
});
// best-of-window harness: a fake camera whose callback we drive with explicit times; performance.now is a settable clock (the camera callback's `now` is on the same clock in a browser)
const bowSetup=`let amp=10,pnow=0,step=0;const drawn=[],pending=[];globalThis.performance={now:()=>{pnow+=step;return pnow}};
  const mk=()=>({width:0,height:0,getContext:()=>({drawImage(v,x,y,w,h){drawn.push([w,h,amp])},getImageData(x,y,w,h){const d=new Uint8ClampedArray(w*h*4);for(let i=0;i<w*h;i++){const v=128+(i%2?amp:-amp);d[4*i]=d[4*i+1]=d[4*i+2]=v;d[4*i+3]=255}return {data:d}}}),
   toBlob(cb,type,q){const blob={tag:"amp"+drawn.filter(d=>d[0]>900).at(-1)[2],type,q};if(globalThis.holdBlobs)pending.push(()=>cb(blob));else cb(blob)}});
  const cbs=[];document.createElement=mk;evidenceCanvas.toBlob=()=>{};cameraPreview.requestVideoFrameCallback=cb=>{cbs.push(cb);return cbs.length};cameraPreview.cancelVideoFrameCallback=()=>{};
  FileReader=class{readAsDataURL(b){this.result="data:image/jpeg;base64,"+btoa(b.tag);setTimeout(()=>this.onload(),0)}};
  const t={label:"cam",stop(){},addEventListener(){},getSettings(){return {width:1920,height:1080}}};navigator.mediaDevices={getUserMedia:async()=>({getTracks:()=>[t],getVideoTracks:()=>[t]})};`;
const bowFrame=(e,a,t)=>e.run(`amp=${a};pnow=${t};cbs.at(-1)(${t},{captureTime:${t}})`);
const sentFrames=e=>e.sockets[0].sent.filter(x=>x.type==='evidence_frame');
const settle=()=>new Promise(r=>setTimeout(r,15));
test('best-of-window: the camera callback closes each 200 ms window with its sharpest frame (grid-aligned, asynchronous); a stalled camera is closed by the timer watchdog; a backed-up socket skips, never queues',async()=>{
 const e=env();e.run(bowSetup);
 await e.run('startCamera()');assert.equal(e.run('bow.active'),true);
 assert.equal(e.sockets[0].sent.find(m=>m.type==='capture_info').capture_mode,'best_of_window');
 bowFrame(e,10,1000);bowFrame(e,40,1033);bowFrame(e,20,1066);
 assert.equal(sentFrames(e).length,0);                                   // 66 ms in: the window is still open
 bowFrame(e,15,1200);await settle();                                     // the 1200 ms callback closes it (>=200 ms) and joins it
 let m=sentFrames(e);assert.equal(m.length,1);
 assert.equal(m[0].source,'camera');assert.equal(m[0].cand,4);assert.equal(m[0].seen,4);assert.equal(m[0].win_ms,200);assert.equal(m[0].client_t_s,0.033);   // the picked frame's own capture time, not the window end
 assert(m[0].sharp_small>0);assert(m[0].enc_ms>=0);assert.equal(m[0].q,0.95);
 assert.equal(Buffer.from(m[0].data,'base64').toString(),'amp40');
 assert.deepEqual([...e.run('drawn.filter(d=>d[0]===1920).map(d=>d[2])')],[10,40]);   // the amp-20 and amp-15 frames never reached the full-size canvas
 bowFrame(e,30,1233);bowFrame(e,25,1400);await settle();                 // window 2 starts on the 1200 ms grid (1000+200), not at the late callback
 assert.equal(sentFrames(e).length,2);assert.equal(sentFrames(e)[1].win_ms,200);assert.equal(sentFrames(e)[1].seen,2);
 e.run('pnow=1500;sendEvidenceFrame()');assert.equal(sentFrames(e).length,2);assert.equal(e.run('bow.stats.empty_windows'),0);   // watchdog: only after 400 ms without a camera callback
 e.run('pnow=2000;sendEvidenceFrame()');await settle();assert.equal(sentFrames(e).length,2);assert.equal(e.run('bow.stats.empty_windows'),1);   // no new frame -> nothing sent
 e.run('liveSocket.bufferedAmount=4000000');bowFrame(e,30,2100);bowFrame(e,30,2250);await settle();
 assert.equal(sentFrames(e).length,2);assert.equal(e.run('bow.stats.skipped_backlog'),1);
 e.run('liveSocket.bufferedAmount=0;stopCamera()');const stats=e.sockets[0].sent.filter(x=>x.type==='capture_info'&&x.client_stats).at(-1).client_stats;
 assert.equal(stats.frames_sent,2);assert.equal(stats.empty_windows,1);assert.equal(stats.skipped_socket_backlog,1);assert.equal(stats.inflight_max,1);assert.equal(stats.jpeg_quality_final,0.95);
 assert(stats.window_ms_p50>=200&&stats.window_ms_p50<=250);assert(stats.encode_ms_p50>=0);assert.equal(stats.main_thread_lag_ms_p90,null);   // flat scalars (null when never sampled: no interval ran in this mock)
 assert.equal(stats.sent_per_10s,'2');assert.deepEqual(e.sockets[0].sent.at(-1),{type:'camera_state',enabled:false});
 e.run('disconnectLive()');
});
test('best-of-window: two encodes overlap, a third window is skipped (counted), and sustained encode-bound skipping steps JPEG quality down to 0.85 and no further',async()=>{
 const e=env();e.run(bowSetup);e.run('globalThis.holdBlobs=true');await e.run('startCamera()');
 for(let w=0;w<3;w++){bowFrame(e,10,1000+200*w);bowFrame(e,10,1000+200*w+200)}   // windows close at 1200, 1400, 1600
 assert.equal(e.run('bow.inflight'),2);assert.equal(e.run('bow.stats.skipped_encoding'),1);assert.equal(e.run('bow.stats.inflight_max'),2);
 e.run('pending.splice(0).forEach(f=>f())');await settle();assert.equal(e.run('bow.inflight'),0);assert.equal(sentFrames(e).length,2);
 e.run('globalThis.holdBlobs=true');
 for(let w=3;w<80;w++)bowFrame(e,10,1000+200*w);                         // nothing completes: 2 stay in flight, every later window is skipped
 assert.equal(e.run('evidenceQuality'),0.85);assert.equal(e.run('bow.stats.quality_steps'),2);   // 0.95 -> 0.9 -> 0.85, never lower
 e.run('globalThis.holdBlobs=false;pending.splice(0).forEach(f=>f());stopCamera()');
 e.run('disconnectLive()');
});
test('best-of-window: ranking is budgeted - an expensive per-frame check samples every Nth camera frame (up to 6) and relaxes again when cheap',async()=>{
 const e=env();e.run(bowSetup);await e.run('startCamera()');
 e.run('step=30');                                                         // every performance.now() call advances 30 ms: ranking "costs" >=60 ms per camera frame
 for(let i=0;i<120;i++)bowFrame(e,10,1000+33*i);
 const slow=e.run('bow.rankEvery');assert(slow>=2&&slow<=6,'rankEvery '+slow);
 const ranked=e.run('bow.samples.rank_ms.length'),seen=e.run('bow.stats.callbacks');assert(ranked<seen);          // not every camera frame was ranked
 e.run('step=0');for(let i=120;i<420;i++)bowFrame(e,10,1000+33*i);
 assert.equal(e.run('bow.rankEvery'),1);
 e.run('stopCamera();disconnectLive()');
});
test('while the camera is on only the newest rows of a long book/item list are drawn (the table is rebuilt on every state push); the full list returns when the camera stops',async()=>{
 const e=env();const books=Array.from({length:120},(_,i)=>({id:'bk-'+String(i).padStart(3,'0'),status:'unidentified',title:null,author:null,visible_text:null,evidence:{}}));
 e.run(`state={...emptyState,status:"capturing",books:${JSON.stringify(books)},items:[],totals:{book_count:120}};cameraStream={};renderInventory()`);
 let html=e.elements.get('#books').innerHTML;assert.equal((html.match(/<tr><td>bk-/g)||[]).length,40);assert(html.includes('bk-119')&&!html.includes('bk-079'));assert(html.includes('Showing the latest 40 of 120'));
 e.run('state={...state,books:state.books.slice(-40),list_caps:{books:{shown:40,total:751}}};renderInventory()');   // a server-capped push still says how many there are
 html=e.elements.get('#books').innerHTML;assert.equal((html.match(/<tr><td>bk-/g)||[]).length,40);assert(html.includes('Showing the latest 40 of 751'));
 e.run(`state={...emptyState,status:"capturing",books:${JSON.stringify(books)},items:[],totals:{book_count:120}};cameraStream=null;renderInventory()`);
 html=e.elements.get('#books').innerHTML;assert.equal((html.match(/<tr><td>bk-/g)||[]).length,120);assert(!html.includes('Showing the latest'));
});
test('state messages are timed (bytes and apply ms) while the camera loop runs; the frame-timing lines show both clocks, the rate over time and the phone counters',async()=>{
 const e=env();e.run(bowSetup);await e.run('startCamera()');
 e.sockets[0].onmessage({data:JSON.stringify({type:'state',session_id:'initial',state:{session_id:'initial',status:'capturing',transcript:[],tool_activity:[]}})});
 assert.equal(e.run('bow.stats.state_msgs'),1);assert(e.run('bow.samples.state_kb[0]')>0);
 e.run('stopCamera();disconnectLive()');
 const lines=e.run(`captureTimingLines({frames:300,fps_effective:2.5,orientation:"portrait",frame_size:"1080x1920",device:{requested_resolution:"3840x2160",width:1080,height:1920,zoom:1},frame_timing:{arrival_clock:{p50:0.3,p90:0.9,max:2.4},capture_clock:{p50:0.33,p90:0.87,max:2.5},fps_first_third:3.8,fps_last_third:1.6,server_ingest_ms:{p50:80,p90:120,max:400}},client_counters:{camera_callback_fps:9.8,encode_ms_p50:120,encode_ms_p90:220,rank_ms_p50:18,rank_every_final:3,main_thread_lag_ms_p90:310,main_thread_lag_ms_max:900,state_msgs:200,state_kb_p90:2800,state_apply_ms_p90:140,skipped_while_encoding:12,jpeg_quality_final:0.9}})`);
 assert.match(lines[0],/Frame rate: 2\.50 fps \(target 5\), first third 3\.8 -> last third 1\.6 fps · gap p50\/p90\/max 0\.30\/0\.90\/2\.40 s \(server arrival\) · 0\.33\/0\.87\/2\.50 s \(phone clock\) · server ingest p50\/p90 80\/120 ms · portrait 1080x1920 · camera asked 3840x2160, granted 1080x1920, zoom 1x/);
 assert.match(lines[1],/Phone: camera 9\.8 callbacks\/s · encode p50\/p90 120\/220 ms.*every 3 frame.*lag p90\/max 310\/900 ms · state msgs 200 \(p90 2800 KB, 140 ms\).*JPEG q0\.9/);
 assert.match(lines[2],/Portrait frames: hold the phone sideways/);assert.deepEqual([...e.run('captureTimingLines({frames:0})')],[]);
});
test('?res=4k asks for 3840x2160 first and sends native pixels up to 3840 long; the default stays 1920 wide; the granted size and zoom are reported; the socket backlog allowance grows with the frame size',async()=>{
 const run=async(search,vw,vh)=>{
  const e=env();e.run(bowSetup);e.run(`window.location.search=${JSON.stringify(search)};cameraPreview.videoWidth=${vw};cameraPreview.videoHeight=${vh};
   let zoomApplied=null;const tr={label:"cam",stop(){},addEventListener(){},getSettings(){return {width:${vw},height:${vh},zoom:zoomApplied||1}},getCapabilities(){return {zoom:{min:1,max:5}}},applyConstraints:async(c)=>{if(c.advanced&&c.advanced[0].zoom)zoomApplied=c.advanced[0].zoom}};
   const seen=[];navigator.mediaDevices={getUserMedia:async(c)=>{seen.push(c);return {getTracks:()=>[tr],getVideoTracks:()=>[tr]}}}`);
  await e.run('startCamera()');return e;};
 let e=await run('?res=4k',2160,3840);
 assert.equal(e.run('seen[0].video.width.ideal'),3840);assert.equal(e.run('seen[0].video.height.ideal'),2160);
 let info=e.sockets[0].sent.find(m=>m.type==='capture_info');assert.equal(info.requested_resolution,'3840x2160');assert.equal(info.width,2160);assert.equal(info.height,3840);assert.equal(info.zoom,1);
 bowFrame(e,10,1000);bowFrame(e,10,1200);await settle();assert.deepEqual([...e.run('drawn.filter(d=>d[0]>900).at(-1).slice(0,2)')],[2160,3840]);   // native pixels, not scaled to 1920 wide
 e.run('stopCamera();disconnectLive()');
 e=await run('',1080,1920);assert.equal(e.run('seen[0].video.width.ideal'),1920);assert.equal(e.sockets[0].sent.find(m=>m.type==='capture_info').requested_resolution,'1920x1080');
 e.run('stopCamera();disconnectLive()');
 e=await run('?res=4k&zoom=2',1920,1080);                                // zoom is opt-in, clamped to the track's capability
 assert.equal(e.run('zoomApplied'),2);assert.equal(e.sockets[0].sent.find(m=>m.type==='capture_info').zoom_requested,2);
 e.run('stopCamera();disconnectLive()');
 e=await run('',1080,1920);assert.equal(e.run('zoomApplied'),null);
 bowFrame(e,10,1000);bowFrame(e,10,1200);await settle();                  // one frame sent: its size sets the backlog allowance (3 frames' worth)
 e.run('bow.lastMsgBytes=4000000;liveSocket.bufferedAmount=11000000');bowFrame(e,10,1400);await settle();assert.equal(e.run('bow.stats.skipped_backlog'),0);
 e.run('liveSocket.bufferedAmount=13000000');bowFrame(e,10,1600);await settle();assert.equal(e.run('bow.stats.skipped_backlog'),1);
 e.run('bufferedAmount=0;stopCamera();disconnectLive()');
});
function debugEnv(){
 const make=()=>{const el={children:[],textContent:'',className:'',style:{},dataset:{},attrs:{},hidden:false,value:'',checked:false,disabled:false,
   append(...n){this.children.push(...n)},replaceChildren(...n){this.children=[...n]},setAttribute(k,v){this.attrs[k]=v},getAttribute(k){return this.attrs[k]},addEventListener(){},removeAttribute(){},focus(){}};return el};
 const els=new Map();const get=id=>{if(!els.has(id))els.set(id,make());return els.get(id)};
 const ctx=vm.createContext({console,URL,URLSearchParams,location:{href:'http://127.0.0.1:4182/debug/'},document:{getElementById:get,createElement:()=>make(),createElementNS:()=>make(),addEventListener(){},querySelector:()=>make()},window:{addEventListener(){}},
  fetch:async()=>({ok:true,json:async()=>[],text:async()=>''}),setInterval:()=>1,clearInterval(){},innerHeight:900,DOMPoint:function(){}, Event:function(){}});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../app/static/debug-app.js'),'utf8'),ctx);
 return {ctx,els,run:x=>vm.runInContext(x,ctx)};
}
test('Debugger build status: a running build shows stage, percent, ETA and a progress bar; queued shows #n; failures are visible warnings',()=>{
 const d=debugEnv();
 const res={status:{run_id:'sweep_x',steps:{
   panorama:{state:'running',stage:'aligning: matching',elapsed_s:12.4,progress:{label:'aligning frames: matching 30/60',fraction:0.4,eta_s:15}},
   recon3d:{state:'queued',position:1,ahead:[{run_id:'sweep_x',kind:'panorama'}]}}},queue:{running:['sweep_x','panorama'],queued:[['sweep_x','recon3d']]}};
 d.run(`renderBuildStatus(${JSON.stringify(res)})`);
 const rows=d.els.get('buildStatus').children;assert.equal(rows.length,2);
 const text=r=>r.children[0].textContent;
 assert(text(rows[0]).includes('2D panorama · sweep_x: building · aligning frames: matching 30/60 · 40% · ~15 s left · 12 s elapsed'));
 const bar=rows[0].children[1];assert.equal(bar.className,'bar');assert.equal(bar.attrs.role,'progressbar');assert.equal(bar.attrs['aria-valuenow'],'40');assert.equal(bar.children[0].style.width,'40%');
 assert(text(rows[1]).includes('3D map · sweep_x: queued #1 (waiting for 2D panorama of sweep_x)'));assert.equal(rows[1].children.length,1);          // queued: position, no bar
 d.run('buildError="Cannot build the 3D map of r: boom"');d.run(`renderBuildStatus(${JSON.stringify({status:{run_id:'r',steps:{panorama:{state:'failed',error:'RuntimeError: boom'},recon3d:{state:'done',out_run:'r_recon3d_v1',markers:7,book_count:7}}},queue:{running:null,queued:[]}})})`);
 const r2=d.els.get('buildStatus').children;assert.equal(r2[0].className,'warning');assert(text(r2[0]).includes('Cannot build'));assert.equal(r2[1].className,'warning');assert(text(r2[1]).includes('FAILED: RuntimeError: boom'));
 assert(text(r2[2]).includes('ready: r_recon3d_v1 (7 markers = book_count 7)'));assert.equal(r2[2].children[1].children[0].style.width,'100%');
 assert.equal(d.run('pct(0.126)'),13);assert.equal(d.run('pct(7)'),100);assert.equal(d.run('eta(200)'),'~3 min left');assert.equal(d.run('eta(20)'),'~20 s left');
});
test('book count shows best guess, range and reliability on Sweep and Measurements',()=>{
 const e=env();e.run('state={...emptyState,status:"finished",totals:{book_count:174,book_count_range:[160,462],count_reliability:"low",book_count_lower:160,book_count_upper:462}};renderInventory();renderMeasurement()');
 const tot=e.elements.get('#totals').innerHTML;assert(tot.includes('~174 (range 160–462, reliability low)'));assert(!tot.includes('Plausible range'));
 assert.equal(e.elements.get('#measureSummary').textContent,'Books: ~174 (range 160–462, reliability low)');
 e.run('state={...emptyState,status:"finished",totals:{book_count:3}};renderInventory()');assert(e.elements.get('#totals').innerHTML.includes('<b>3</b>'));
});
(async()=>{for(const [name,fn]of tests){await fn();console.log('PASS '+name)}console.log(`${tests.length} client regression tests passed`)})().catch(e=>{console.error(e);process.exitCode=1});
