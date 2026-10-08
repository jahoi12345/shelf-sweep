// Debugger 3D view of runs/<run>_recon3d_vN/recon3d.json (pipeline/recon3d.py). UNSCALED SfM: no metric units.
// New files (alignment.applied) are shelf-aligned: plane z=0, camera side +z, spine-up +y. Older files are raw COLMAP frames (y down).
// three.js is vendored (static/vendor, MIT). Rendering is on demand; picking/hover is screen-space (no raycaster).
// Clicking a marker (or an unplaced group id) dispatches document 'recon3d-marker' {detail: {run_id, group_id}}.
import * as THREE from './vendor/three.module.min.js';

const $ = id => document.getElementById(id);
const api = p => new URL(p, location.href).href;
let renderer, scene, camera, root, pointsObj, camLine, camFrusta, markerObj, markerData = [], ready = false, dirty = true;
let runId = null, current = null, selected = -1, hovered = -1, loadSeq = 0, aligned = false, frame3d = null;
let labelEl = null, countEl = null, viewTools = null;
const orbit = { target: new THREE.Vector3(), radius: 3, theta: 0.12, phi: 1.5 };
const COLORS = { triangulated: 0x25ebbc, plane_projected: 0x6ea8ff, item: 0xf59db9 };
const FOV = 50, MARKER_PX = 11;
const rgb = hex => [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
const methodOf = m => (m.kind === 'item' ? 'item' : m.method === 'plane_projected' ? 'plane_projected' : 'triangulated');   // 'ray_triangulation' (old files) = triangulated

const MARKER_VERT = `attribute vec3 aColor; attribute float aState; uniform float uSize; varying vec3 vColor; varying float vState;
void main(){ vColor = aColor; vState = aState; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = uSize * (1.0 + 0.7 * aState); }`;
const MARKER_FRAG = `varying vec3 vColor; varying float vState;
void main(){ vec2 p = gl_PointCoord * 2.0 - 1.0; float r2 = dot(p, p); if (r2 > 1.0) discard;
  vec3 n = vec3(p.x, -p.y, sqrt(1.0 - r2)); float l = clamp(dot(n, normalize(vec3(-0.4, 0.6, 0.7))), 0.0, 1.0);
  vec3 c = vColor * (0.4 + 0.7 * l) + vec3(0.35) * pow(l, 24.0);
  if (r2 > 0.74) c = mix(c, vec3(0.03, 0.06, 0.1), 0.7);
  if (vState > 0.9) c = mix(c, vec3(1.0), 0.55);
  gl_FragColor = vec4(c, 1.0); }`;

function init() {
  if (ready) return true;
  const holder = $('reconCanvasHolder');
  try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); }
  catch (e) { $('reconNote').textContent = '3D unavailable: this browser could not create a WebGL context.'; return false; }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setClearColor(0x0b121b);
  holder.replaceChildren(renderer.domElement);
  holder.style.position = 'relative';
  labelEl = document.createElement('div'); labelEl.id = 'reconLabel';
  labelEl.style.cssText = 'position:absolute;left:0;top:0;display:none;pointer-events:none;padding:1px 6px;font:11px/1.5 ui-monospace,Menlo,monospace;background:rgba(11,18,27,.92);color:#e8f1ff;border:1px solid #25ebbc;border-radius:3px;white-space:nowrap;z-index:2';
  holder.append(labelEl);
  buildStatusUi();
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 1e6);
  root = new THREE.Group();
  scene.add(root);
  const el = renderer.domElement;
  let drag = null;
  el.addEventListener('contextmenu', e => e.preventDefault());
  el.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.button !== 0 || e.shiftKey, moved: 0 }; el.setPointerCapture(e.pointerId); setHover(-1); });
  el.addEventListener('pointermove', e => {
    if (!drag) { setHover(nearest(e, 10)); return; }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.pan) {
      const k = orbit.radius * 0.0016, right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      orbit.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
    } else { orbit.theta -= dx * 0.006; orbit.phi = Math.min(Math.PI - 0.05, Math.max(0.05, orbit.phi - dy * 0.006)); }
    update();
  });
  el.addEventListener('pointerup', e => { const d = drag; drag = null; if (d && d.moved < 5) pick(e); });
  el.addEventListener('pointerleave', () => setHover(-1));
  el.addEventListener('wheel', e => { e.preventDefault(); const lim = frame3d ? frame3d.extent : 1; orbit.radius = Math.min(lim * 40, Math.max(lim * 0.02, orbit.radius * (e.deltaY > 0 ? 1.12 : 1 / 1.12))); update(); }, { passive: false });
  new ResizeObserver(resize).observe(holder);
  ready = true; resize(); loop();
  return true;
}

// Status line + unplaced list + view buttons are created here so no HTML change is needed.
function buildStatusUi() {
  const note = $('reconNote');
  countEl = document.createElement('div'); countEl.id = 'reconCount'; countEl.style.cssText = 'margin:6px 0;font-size:13px';
  note.after(countEl);
  const tools = note.previousElementSibling;
  if (tools && tools.classList.contains('tools')) {
    viewTools = document.createElement('span'); viewTools.id = 'reconViews'; viewTools.style.cssText = 'display:inline-flex;gap:6px';
    for (const [name, label] of [['front', 'Front'], ['top', 'Top'], ['side', 'Side']]) {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = label; b.title = `${label} view of the shelf-aligned scene`; b.onclick = () => setView(name); viewTools.append(b);
    }
    tools.append(viewTools);
  }
}

function resize() {
  if (!ready) return;
  const holder = $('reconCanvasHolder'), w = Math.max(200, holder.clientWidth), h = Math.max(200, holder.clientHeight);
  renderer.setSize(w, h, false); renderer.domElement.style.width = '100%'; renderer.domElement.style.height = '100%';
  camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
}
function update() {
  const s = Math.sin(orbit.phi);
  camera.position.set(orbit.target.x + orbit.radius * s * Math.sin(orbit.theta), orbit.target.y + orbit.radius * Math.cos(orbit.phi), orbit.target.z + orbit.radius * s * Math.cos(orbit.theta));
  camera.lookAt(orbit.target); camera.updateMatrixWorld(); dirty = true;
}
function loop() { requestAnimationFrame(loop); if (dirty && ready) { renderer.render(scene, camera); dirty = false; placeLabel(); } }

function clear() {
  for (const o of [pointsObj, camLine, camFrusta, markerObj]) { if (o) { root.remove(o); o.geometry.dispose(); o.material.dispose(); } }
  pointsObj = camLine = camFrusta = markerObj = null; markerData = []; selected = hovered = -1; frame3d = null; $('reconDetail').replaceChildren();
  if (countEl) countEl.replaceChildren();
  if (labelEl) labelEl.style.display = 'none';
}
const percentile = (arr, q) => { const a = Float64Array.from(arr).sort(); return a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))]; };

// Smallest-variance axis of the cloud (the shelf-face normal for a planar shelf), by Jacobi eigen-decomposition of the 3x3 covariance.
// Only used for older (non-aligned) recon3d.json files.
function planeNormal(pts) {
  const n = pts.length; if (n < 10) return null;
  const mean = [0, 1, 2].map(i => pts.reduce((a, p) => a + p[i], 0) / n), a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of pts) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) a[i][j] += (p[i] - mean[i]) * (p[j] - mean[j]) / n;
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 30; sweep++) {
    for (let p = 0; p < 2; p++) for (let q = p + 1; q < 3; q++) {
      if (Math.abs(a[p][q]) < 1e-14) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 3; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 3; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
  }
  const vals = [a[0][0], a[1][1], a[2][2]], i = vals.indexOf(Math.min(...vals));
  return [v[0][i], v[1][i], v[2][i]];
}

const hasAlignment = data => !!(data.alignment && data.alignment.applied);

function show(data) {
  clear();
  aligned = hasAlignment(data);
  root.rotation.x = aligned ? 0 : Math.PI;     // old files: COLMAP is y-down / z-forward; flip so the shelf stands up
  const pts = data.points || [], colours = data.point_colors || [];
  if (pts.length < 3) { frame3d = null; describe(data); if (viewTools) viewTools.style.display = 'none'; dirty = true; return; }   // failed run: nothing to draw, the note says why
  // Robust frame: 2-98% box around the points, widened, so far depth outliers do not dominate the view.
  const lo = [0, 1, 2].map(i => percentile(pts.map(p => p[i]), 0.02)), hi = [0, 1, 2].map(i => percentile(pts.map(p => p[i]), 0.98));
  const pad = [0, 1, 2].map(i => (hi[i] - lo[i]) * 0.5), min = lo.map((v, i) => v - pad[i]), max = hi.map((v, i) => v + pad[i]);
  const inside = p => p[0] >= min[0] && p[0] <= max[0] && p[1] >= min[1] && p[1] <= max[1] && p[2] >= min[2] && p[2] <= max[2];
  const keep = []; pts.forEach((p, i) => { if (inside(p)) keep.push(i); });
  const pos = new Float32Array(keep.length * 3), col = new Float32Array(keep.length * 3);
  keep.forEach((i, k) => { pos.set(pts[i], k * 3); const c = colours[i] || [200, 200, 200]; col.set([c[0] / 255, c[1] / 255, c[2] / 255], k * 3); });
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  pointsObj = new THREE.Points(g, new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: true }));
  root.add(pointsObj);
  const centre = lo.map((v, i) => (v + hi[i]) / 2), span = hi.map((v, i) => v - lo[i]);
  const extent = Math.max(span[0], span[1], span[2], 1e-6);
  frame3d = { extent, centre, span, lo, hi, cams: (data.cameras || []).map(c => c.center) };
  camera.near = extent * 0.002; camera.far = extent * 400; camera.updateProjectionMatrix();
  // Camera path (thin line) + small frusta (depth 3% of the scene size).
  const cams = (data.cameras || []).slice().sort((a, b) => (a.frame_id || '').localeCompare(b.frame_id || ''));
  if (cams.length) {
    const path = new Float32Array(cams.length * 3); cams.forEach((c, i) => path.set(c.center, i * 3));
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(path, 3));
    camLine = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0xffca4e, transparent: true, opacity: 0.8 })); root.add(camLine);
    const s = extent * 0.03, seg = [];
    for (const c of cams) {
      const R = c.rotation_world_to_cam, o = c.center, f = (c.focal_px || 1000), hw = (c.width_px || 1920) / 2 / f, hh = (c.height_px || 1080) / 2 / f;
      const toWorld = v => [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2], R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2], R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]];
      const corners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => { const w = toWorld([x * s, y * s, s]); return [o[0] + w[0], o[1] + w[1], o[2] + w[2]]; });
      corners.forEach((p, i) => { seg.push(...o, ...p); const q = corners[(i + 1) % 4]; seg.push(...p, ...q); });
    }
    const fg = new THREE.BufferGeometry(); fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(seg), 3));
    camFrusta = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ color: 0xffca4e, transparent: true, opacity: 0.55 })); root.add(camFrusta);
  }
  // Markers: books first (data.markers), then optional item markers; sphere-shaded round sprites of constant screen size.
  const finite = m => Array.isArray(m.xyz) && m.xyz.every(Number.isFinite);
  markerData = [...(data.markers || []), ...(data.item_markers || [])].filter(finite);
  const mp = new Float32Array(markerData.length * 3), mc = new Float32Array(markerData.length * 3), ms = new Float32Array(markerData.length);
  markerData.forEach((m, i) => { mp.set(m.xyz, i * 3); mc.set(rgb(COLORS[methodOf(m)]), i * 3); });
  const mg = new THREE.BufferGeometry(); mg.setAttribute('position', new THREE.BufferAttribute(mp, 3)); mg.setAttribute('aColor', new THREE.BufferAttribute(mc, 3)); mg.setAttribute('aState', new THREE.BufferAttribute(ms, 1));
  markerObj = new THREE.Points(mg, new THREE.ShaderMaterial({ uniforms: { uSize: { value: MARKER_PX * renderer.getPixelRatio() } }, vertexShader: MARKER_VERT, fragmentShader: MARKER_FRAG, depthTest: false, transparent: false }));
  markerObj.renderOrder = 5; markerObj.frustumCulled = false; root.add(markerObj);
  applyToggles();
  if (aligned) setView('front');
  else {
    // Older file: target in the flipped frame (root rotated 180 deg about x), start frontal to the cloud's plane normal.
    orbit.target.set(centre[0], -centre[1], -centre[2]); orbit.radius = extent * 1.1; orbit.theta = 0.35; orbit.phi = 1.35;
    const normal = planeNormal(keep.slice(0, 6000).map(i => pts[i]));
    if (normal && cams.length) {
      const cc = [0, 1, 2].map(i => cams.reduce((a, c) => a + c.center[i], 0) / cams.length), toCams = [cc[0] - centre[0], cc[1] - centre[1], cc[2] - centre[2]];
      const sign = (normal[0] * toCams[0] + normal[1] * toCams[1] + normal[2] * toCams[2]) < 0 ? -1 : 1, d = [sign * normal[0], -sign * normal[1], -sign * normal[2]];
      orbit.phi = Math.acos(Math.max(-1, Math.min(1, d[1]))); orbit.theta = Math.atan2(d[0], d[2]);
      orbit.phi = Math.min(Math.max(orbit.phi, 0.3), Math.PI - 0.3);
    }
    update();
  }
  if (viewTools) viewTools.style.display = aligned ? 'inline-flex' : 'none';
  describe(data);
}

// Preset views of the shelf-aligned scene (Front = looking at the shelf face from the camera side, +y up). The orbit radius is the
// smallest that keeps the key points (cloud box; plus the camera path for Top/Side) inside the frustum.
function setView(name) {
  if (!frame3d || !aligned) return;
  const [theta, phi] = { top: [0, 0.08], side: [Math.PI / 2, Math.PI / 2] }[name] || [0.12, 1.5], withCams = name === 'top' || name === 'side';
  const { lo, hi, cams } = frame3d, kp = [];
  for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) kp.push([x, y, z]);
  if (withCams) kp.push(...cams);
  const mn = [0, 1, 2].map(i => Math.min(...kp.map(p => p[i]))), mx = [0, 1, 2].map(i => Math.max(...kp.map(p => p[i]))), c = mn.map((v, i) => (v + mx[i]) / 2);
  const s = Math.sin(phi), dir = new THREE.Vector3(s * Math.sin(theta), Math.cos(phi), s * Math.cos(theta)), up = new THREE.Vector3(0, 1, 0);
  const right = new THREE.Vector3().crossVectors(up, dir).normalize(), upv = new THREE.Vector3().crossVectors(dir, right).normalize();
  const t = Math.tan(FOV * Math.PI / 360) * 0.92, aspect = camera.aspect || 1.6;
  let r = 1e-6;
  for (const p of kp) {
    const q = new THREE.Vector3(p[0] - c[0], p[1] - c[1], p[2] - c[2]), along = q.dot(dir);
    r = Math.max(r, along + Math.abs(q.dot(right)) / (t * aspect), along + Math.abs(q.dot(upv)) / t);
  }
  orbit.target.set(c[0], c[1], c[2]); orbit.theta = theta; orbit.phi = phi; orbit.radius = r * 1.06 + frame3d.extent * 0.02;
  update();
}

function describe(data) {
  const reg = data.registered_frames, tot = data.frames_used || data.frames_total, plane = data.plane, al = data.alignment;
  const frameText = aligned ? `aligned to the fitted shelf plane (z=0, +y spine-up${al.up_source === 'camera_up' ? ', from camera up' : ''}) · UNSCALED: arbitrary units, no metric distances`
    : 'UNSCALED (arbitrary scale/orientation, no metric units)';
  const planeText = plane ? ` · plane rms ${plane.rms_over_extent != null ? (100 * plane.rms_over_extent).toFixed(1) + '% of scene extent' : plane.rms} (${Math.round(100 * plane.inlier_ratio)}% of points on it)` : '';
  const unreg = (data.unregistered_frames || []).length;
  const markersN = data.markers_total != null ? data.markers_total : data.group_markers;
  $('reconNote').textContent = `${String(data.status).toUpperCase()} · ${frameText} · ${reg}/${tot} frames registered (${Math.round(100 * data.registered_ratio)}%${unreg ? `, ${unreg} unregistered` : ''})${data.frames_used && data.frames_used < data.frames_total ? ` · frame diet: ${data.frames_used} of the run's ${data.frames_total} frames used (${Math.round(100 * (data.registered_of_total || 0))}% of all registered)` : ''} · ${data.points_kept} points · reprojection ${data.mean_reprojection_error_px == null ? 'n/a' : data.mean_reprojection_error_px.toFixed(2) + ' px'}${planeText}${data.book_count == null ? ` · ${markersN}/${data.groups_total} group markers` : ''}${data.reason ? ' · ' + data.reason : ''}`;
  countEl.replaceChildren();
  if (data.book_count == null) return;       // older file: no book_count / unplaced fields
  const books = data.markers || [], unplaced = data.unplaced_groups || [], tri = books.filter(m => methodOf(m) === 'triangulated').length, proj = books.filter(m => methodOf(m) === 'plane_projected').length;
  const line = document.createElement('div'); line.id = 'reconCountLine';
  const dot = (hex, text) => { const s = document.createElement('span'); s.style.cssText = `color:#${hex.toString(16).padStart(6, '0')}`; s.textContent = text; return s; };
  line.append(unplaced.length ? `${books.length} book markers + ${unplaced.length} unplaced = book_count ${data.book_count} (` : `${books.length} book markers = book_count ${data.book_count} (`,
    dot(COLORS.triangulated, `${tri} placed by triangulation`), ', ', dot(COLORS.plane_projected, `${proj} plane-projected`), `, ${unplaced.length} unplaced)`);
  if ((data.item_markers || []).length) line.append(dot(COLORS.item, ` · ${data.item_markers.length} item markers (not counted as books)`));
  countEl.append(line);
  if (unplaced.length) {
    const d = document.createElement('details'), sum = document.createElement('summary'); sum.textContent = `${unplaced.length} unplaced group(s): ${unplaced.slice(0, 8).map(u => u.group_id).join(', ')}${unplaced.length > 8 ? ', …' : ''}`; d.append(sum);
    for (const u of unplaced) {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = u.group_id; b.title = u.reason || ''; b.style.cssText = 'margin:2px;padding:2px 6px;font-size:12px';
      b.onclick = () => { select(-1); emit(u.group_id); showUnplaced(u); }; d.append(b);
    }
    countEl.append(d);
  }
}

function showUnplaced(u) {
  const el = $('reconDetail'); el.replaceChildren();
  const h = document.createElement('h3'); h.textContent = `${u.group_id} · unplaced`; const p = document.createElement('p'); p.textContent = u.reason || 'No 3D marker for this group.'; el.append(h, p);
}

function applyToggles() {
  if (pointsObj) pointsObj.visible = $('reconPoints').checked;
  if (camLine) camLine.visible = camFrusta.visible = $('reconCams').checked;
  if (markerObj) markerObj.visible = $('reconMarkers').checked;
  if (!markerObj || !markerObj.visible) setHover(-1);
  dirty = true;
}

function screenOf(i) {
  const m = markerData[i], v = new THREE.Vector3(...m.xyz).applyMatrix4(root.matrixWorld).project(camera);
  const r = renderer.domElement.getBoundingClientRect();
  return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height, z: v.z };
}
function nearest(e, radius) {
  if (!markerObj || !markerObj.visible) return -1;
  root.updateMatrixWorld(); let best = -1, bd = radius;
  markerData.forEach((m, i) => { const s = screenOf(i); if (s.z < 1) { const d = Math.hypot(s.x - e.clientX, s.y - e.clientY); if (d < bd) { bd = d; best = i; } } });
  return best;
}
function paintStates() {
  if (!markerObj) return;
  const a = markerObj.geometry.attributes.aState; markerData.forEach((m, k) => a.setX(k, k === selected ? 1 : k === hovered ? 0.5 : 0)); a.needsUpdate = true; dirty = true;
}
function setHover(i) { if (i === hovered) return; hovered = i; paintStates(); }
function placeLabel() {
  const i = hovered >= 0 ? hovered : selected;
  if (i < 0 || !markerObj || !markerObj.visible || !markerData[i]) { labelEl.style.display = 'none'; return; }
  const s = screenOf(i), hr = $('reconCanvasHolder').getBoundingClientRect(), m = markerData[i];
  if (s.z >= 1) { labelEl.style.display = 'none'; return; }
  labelEl.textContent = `${m.group_id}${m.method ? ' · ' + m.method.replace('ray_', '').replace('_', ' ') : ''}`;
  labelEl.style.borderColor = '#' + COLORS[methodOf(m)].toString(16).padStart(6, '0');
  labelEl.style.transform = `translate(${Math.round(s.x - hr.left + 9)}px, ${Math.round(s.y - hr.top - 24)}px)`; labelEl.style.display = 'block';
}
const emit = group_id => document.dispatchEvent(new CustomEvent('recon3d-marker', { detail: { run_id: runId, group_id } }));
function pick(e) {
  const best = nearest(e, 14);
  select(best);
  if (best >= 0) emit(markerData[best].group_id);
}
function select(i) {
  selected = i; const el = $('reconDetail'); el.replaceChildren(); paintStates();
  if (i < 0) return;
  const m = markerData[i], sights = m.sightings || m.frames || [];
  const h = document.createElement('h3'); h.textContent = `${m.group_id} · ${m.kind}${m.category ? ' · ' + m.category : ''}`; el.append(h);
  const p = document.createElement('p'), pos = m.xyz.map(v => v.toFixed(2)).join(', '), reg = m.registered_sightings != null ? `${m.registered_sightings}${m.sightings_total != null ? ' of ' + m.sightings_total : ''} registered sighting(s)` : '';
  p.textContent = methodOf(m) === 'plane_projected'
    ? `Projected onto the shelf plane from ${m.views} sighting ray(s) (${reg}; too little parallax or off-plane for triangulation) · position ${pos} (unscaled)`
    : `Triangulated from ${m.views} of ${m.registered_sightings} registered sightings · parallax ${m.baseline_angle_deg}° · ray residual ${m.ray_residual} (SfM units, unscaled) · position ${pos} (unscaled)`;
  el.append(p);
  const row = document.createElement('div'); row.className = 'tools'; el.append(row);
  const view = document.createElement('div'); view.id = 'reconFrame'; el.append(view);
  sights.forEach((f, k) => {
    const b = document.createElement('button'); b.textContent = f.frame_id; b.onclick = () => frame(m, f); row.append(b); if (k === 0) frame(m, f);
  });
  if (window.selectObservation && sights[0] && sights[0].observation_id) {
    const b = document.createElement('button'); b.textContent = 'Show in stitched/source views'; b.onclick = () => window.selectObservation(sights[0].observation_id); row.append(b);
  }
}
function frame(m, f) {
  const holder = $('reconFrame'); holder.replaceChildren();
  const [x0, y0, x1, y1] = f.bbox_xyxy, wrap = document.createElement('div'); wrap.style.position = 'relative'; wrap.style.maxWidth = '640px';
  const img = document.createElement('img'); img.src = api(`api/frame/${encodeURIComponent(runId)}/${encodeURIComponent(f.frame_id)}`); img.style.width = '100%'; img.alt = `${f.frame_id} source frame`;
  img.onload = () => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
    svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%';
    const shape = f.polygon_xy && f.polygon_xy.length >= 3 ? document.createElementNS('http://www.w3.org/2000/svg', 'polygon') : document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    const attrs = shape.tagName === 'polygon' ? { points: f.polygon_xy.map(q => q.join(',')).join(' ') } : { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
    for (const [k, v] of Object.entries({ ...attrs, fill: 'none', stroke: '#25ebbc', 'stroke-width': Math.max(2, img.naturalWidth / 400) })) shape.setAttribute(k, v);
    svg.append(shape); wrap.append(svg);
  };
  img.onerror = () => { holder.textContent = `Frame ${f.frame_id} unavailable`; };
  wrap.append(img); holder.append(wrap);
}
function selectGroup(groupId) {          // programmatic: no 'recon3d-marker' event (the caller already knows)
  const i = markerData.findIndex(m => m.group_id === groupId);
  if (i >= 0) { select(i); return true; }
  const u = ((current && current.unplaced_groups) || []).find(x => x.group_id === groupId);
  if (u) { select(-1); showUnplaced(u); }
  return false;
}

async function list(run) {
  const mine = ++loadSeq; runId = run; const sel = $('reconSelect'); sel.replaceChildren(); clear();
  if (!run) { $('reconNote').textContent = 'Select a run.'; return; }
  let data;
  try { const r = await fetch(api(`api/recon3d/${encodeURIComponent(run)}`)); if (!r.ok) throw Error(await r.text()); data = await r.json(); }
  catch (e) { if (mine === loadSeq) $('reconNote').textContent = `3D list unavailable: ${e.message}`; return; }
  if (mine !== loadSeq) return;
  data.recon.forEach(x => { const o = document.createElement('option'); o.value = x.recon_id; o.textContent = `${x.recon_id} · ${x.status}${x.registered_ratio != null ? ' · ' + Math.round(100 * x.registered_ratio) + '% registered' : ''}`; sel.append(o); });
  if (!data.recon.length) { $('reconNote').textContent = 'No 3D reconstruction for this run yet. Press Build 3D (background, low priority; takes a few minutes).'; return; }
  await load(sel.value, mine);
}
async function load(id, seq = ++loadSeq) {
  if (!id) return;
  $('reconNote').textContent = 'Loading reconstruction…';
  try {
    const r = await fetch(api(`api/recon3d/${encodeURIComponent(runId)}/${encodeURIComponent(id)}`)); if (!r.ok) throw Error(await r.text());
    const data = await r.json(); if (seq !== loadSeq) return; current = data;
    if (!init()) return;
    show(data);
  } catch (e) { $('reconNote').textContent = `3D unavailable: ${e.message}`; }
}

$('reconSelect').onchange = () => load($('reconSelect').value);
for (const id of ['reconPoints', 'reconCams', 'reconMarkers']) $(id).onchange = applyToggles;
window.Recon3D = { list, load, select, selectGroup, current: () => current };
document.dispatchEvent(new Event('recon3d-ready'));
