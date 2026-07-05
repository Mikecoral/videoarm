"""Run Omni pipeline on a single video with verification, memory, and viewer.

    python agent_v2/run_omni_seg.py [--video_id 1] [--no-verify] [--no-memory]
"""

from __future__ import annotations

import json, re, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_AI_LAB = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_AI_LAB))

from agent_v2 import api_client, config, media, memory, ontology
from agent_v2.pipeline_omni import process_video_omni
from agent_v2.runlog import RunLog

VIDEO_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
VIDEO_PATH = config.DATA_ROOT / "videos" / f"{VIDEO_ID}.mp4"
OUT_DIR = Path(_HERE / "outputs" / f"omni_{VIDEO_ID}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Video: {VIDEO_PATH}")
print(f"Omni model: {config.OMNI_MODEL}")
print(f"MLLM model: {config.VISION_MODEL}")

# ── Extract frames (needed for verification) ──
print("Extracting frames ...", flush=True)
video = media.load_video(VIDEO_ID, VIDEO_PATH, cache_root=config.FRAME_CACHE_DIR)
print(f"  {len(video.frames)} frames, {video.duration:.0f}s  {video.width}x{video.height}")

phases = ontology.load_ontology()
log = RunLog()

# ── Omni pipeline (with real verification since frames are available) ──
t0 = time.time()
prediction = process_video_omni(VIDEO_ID, video, phases, log)
dt = time.time() - t0

# Print segments summary
segments = prediction.get("segments", [])
for seg in segments:
    status = seg.get("verification_status", "?")
    conf = seg.get("confidence", "?")
    print(
        f"  [{seg['start']:.0f}s–{seg['end']:.0f}s] {seg.get('action_zh', '?')}"
        f"  {seg.get('caption', '')[:60]}  ({status} conf={conf})"
    )

verified = sum(1 for s in segments if s.get("verification_status") == "verified")
partial = sum(1 for s in segments if s.get("verification_status") == "partial")
print(f"\n{len(segments)} segments (verified={verified} partial={partial}) in {dt:.1f}s")

# ── Video-level summary (DeepSeek) ──
print("\n── Video summary ──")
timeline = "\n".join(
    f"[{s['start']:.0f}s–{s['end']:.0f}s] {s.get('caption', '')}"
    for s in segments
)
summary_prompt = (
    f"你是化学实验视频分析专家。以下是视频 {VIDEO_ID}（总时长 {video.duration:.0f}s，"
    f"实验阶段：{prediction.get('phase_zh', '')}）的全部原子操作分段 caption，按时间排列：\n\n"
    f"{timeline}\n\n"
    "请根据以上分段信息，用一段中文（3-5 句）概括整个视频的核心实验流程。"
    "包括：实验目的、主要操作步骤、关键器具、是否成功完成。"
    "不要编造分段中未提及的内容。\n\n"
    "只输出概括文本，不要带任何前缀或引号。"
)
video_summary = api_client.ask_text(summary_prompt, model=config.STRUCTURED_MODEL, max_tokens=400)
print(f"  {video_summary}")

# ── Build per-video memory + update global memory ──
print("\n── Memory ──")
video_memory = memory.build_video_memory(prediction, log.videos[-1] if log.videos else {}, config.FRAME_CACHE_DIR)
memory.save_video_memory(OUT_DIR, video_memory)

global_memory_path = _HERE / "outputs" / "global_memory.json"
global_mem = memory.load_global_memory(global_memory_path)
global_mem = memory.update_global_memory(global_mem, video_memory, memory_path=str(OUT_DIR / "memory" / f"{VIDEO_ID}.json"))
memory.save_global_memory(global_memory_path, global_mem)
print(f"  video memory: {OUT_DIR}/memory/{VIDEO_ID}.json")
print(f"  global memory: {global_memory_path}")
summary = memory.build_global_summary(global_mem)
print(f"  cross-video: {summary.get('video_count')} videos, "
      f"top actions: {[(a, c) for a, c in summary.get('top_actions', [])[:5]]}")

# ── Save predictions.json (main-aligned format) ──
prediction["video_summary"] = video_summary
pred_path = OUT_DIR / "predictions.json"
pred_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved: {pred_path}")

# ── Save run_log.json ──
log_path = OUT_DIR / "run_log.json"
log_path.write_text(json.dumps(log.as_list(), ensure_ascii=False, indent=2), encoding="utf-8")

# ── Generate viewer HTML ──
_VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video {video_id} — Omni Segmentation</title>
<style>
:root {{ --bg:#f5f6f8; --panel:#fff; --line:#e2e5ea; --text:#1a1d23; --muted:#6b7280; --accent:#2563eb; --accent-soft:#eef3ff; --ok:#16a34a; --ok-soft:#eaf7ee; --warn:#ea580c; }}
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font:14px/1.5 -apple-system,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text) }}
header {{ padding:16px 24px; background:var(--panel); border-bottom:1px solid var(--line); display:flex; align-items:center; gap:16px; flex-wrap:wrap }}
h1 {{ font-size:18px; white-space:nowrap }}
.badge {{ display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600 }}
.badge-phase {{ background:var(--accent-soft); color:var(--accent) }}
.badge-count {{ background:var(--ok-soft); color:var(--ok) }}
.summary-box {{ padding:10px 24px; background:var(--accent-soft); border-bottom:1px solid var(--line); font-size:13px; line-height:1.6 }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 440px; height:calc(100vh - 100px) }}
.panel {{ background:var(--panel); overflow:auto }}
.video-panel {{ border-right:1px solid var(--line); display:flex; flex-direction:column }}
.video-wrap {{ position:sticky; top:0; background:#000; z-index:2 }}
.video-wrap video {{ width:100%; display:block; max-height:55vh; object-fit:contain }}
.timeline {{ padding:16px 20px; flex:1; overflow:auto }}
.timeline h2, .seg-panel h2 {{ font-size:14px; color:var(--muted); margin-bottom:12px; text-transform:uppercase; letter-spacing:.5px }}
.track {{ position:relative; height:36px; background:var(--bg); border-radius:6px; margin-bottom:8px; cursor:pointer; overflow:hidden }}
.track-bar {{ position:absolute; top:0; height:100%; border-radius:4px; opacity:.75; transition:opacity .15s }}
.track-bar:hover {{ opacity:1 }}
.track-bar.active {{ opacity:1; outline:2px solid #000; outline-offset:1px; z-index:2 }}
.time-ruler {{ display:flex; justify-content:space-between; font-size:11px; color:var(--muted); padding:0 2px }}
.seg-panel {{ border-left:1px solid var(--line); padding:16px 20px; overflow:auto; max-height:calc(100vh - 100px) }}
.seg-card {{ padding:12px 14px; border:1px solid var(--line); border-radius:8px; margin-bottom:10px; cursor:pointer; transition:background .1s }}
.seg-card:hover {{ background:var(--accent-soft) }}
.seg-card.selected {{ border-color:var(--accent); background:var(--accent-soft) }}
.seg-card .time {{ font-size:12px; color:var(--muted) }}
.seg-card .time strong {{ color:var(--text); font-size:15px }}
.seg-card .action {{ font-weight:600; margin:4px 0 2px }}
.seg-card .action .tag {{ display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; margin-right:6px; color:#fff }}
.seg-card .caption {{ font-size:13px; color:var(--muted); margin:2px 0 }}
.seg-card .meta {{ font-size:12px; color:var(--muted); margin-top:4px; display:flex; gap:12px; flex-wrap:wrap }}
.seg-card .meta span {{ white-space:nowrap }}
.seg-card .objects {{ color:var(--accent) }}
.seg-card .evidence {{ color:var(--ok) }}
.seg-card .verif {{ font-weight:600 }}
.seg-card .verif.verified {{ color:var(--ok) }}
.seg-card .verif.partial {{ color:var(--warn) }}
.seg-card .verif.rejected {{ color:#dc2626 }}
.seg-card .verif.unverified {{ color:var(--muted) }}
.empty-note {{ font-size:13px; color:var(--muted); font-style:italic; padding:8px 0 }}
.toolbar {{ padding:8px 20px; border-bottom:1px solid var(--line); display:flex; gap:8px; flex-wrap:wrap; align-items:center }}
.toolbar label {{ font-size:12px; color:var(--muted) }}
.toolbar input {{ width:140px; padding:4px 8px; border:1px solid var(--line); border-radius:6px }}
.btn {{ padding:4px 12px; border:1px solid var(--line); border-radius:6px; background:#fff; cursor:pointer; font-size:12px }}
.btn:hover {{ background:var(--accent-soft) }}
</style>
</head>
<body>
<header>
  <h1>🔬 Video {video_id} — Omni Segmentation</h1>
  <span class="badge badge-phase">{phase_zh} (conf={phase_confidence})</span>
  <span class="badge badge-count" id="count-badge">{n_segs} segments</span>
  <span style="font-size:12px;color:var(--muted)">{omni_model} | {duration:.0f}s</span>
</header>
<div class="summary-box"><strong>📝 视频概括：</strong>{video_summary}</div>
<div class="toolbar">
  <label>🔍 <input type="text" id="search" placeholder="caption / action / object…"></label>
  <button class="btn" id="btn-play">▶ 从头播放</button>
  <span style="font-size:12px;color:var(--muted);margin-left:auto" id="hover-info"></span>
</div>
<main>
  <div class="panel video-panel">
    <div class="video-wrap"><video id="video" controls preload="metadata" src="{video_path}"></video></div>
    <div class="timeline"><h2>⏱ 时间轴（点击跳转）</h2><div id="timeline-track"></div></div>
  </div>
  <div class="seg-panel" id="seg-list"><h2>📋 操作片段</h2><div id="segments"></div></div>
</main>
<script>
const DATA = {data_json};
const PALETTE = ['#6366f1','#0891b2','#ca8a04','#dc2626','#7c3aed','#059669','#d946ef','#0d9488','#ea580c','#4f46e5','#0284c7'];
const video = document.getElementById('video');
const segs = DATA.segments;
const dur = DATA.duration;
const actionSet = [...new Set(segs.map(s => s.action))];
const actionColor = {{}};
actionSet.forEach((a,i) => actionColor[a] = PALETTE[i % PALETTE.length]);

document.getElementById('timeline-track').innerHTML =
  '<div class="track" id="track-bar-container">' +
  segs.map((s,i) => '<div class="track-bar" id="bar-' + i + '" style="left:' + (s.start/dur*100) + '%;width:' + ((s.end-s.start)/dur*100) + '%;background:' + actionColor[s.action] + '" title="[' + s.start.toFixed(0) + '-' + s.end.toFixed(0) + 's] ' + s.action_zh + '"></div>').join('') + '</div>' +
  '<div class="time-ruler"><span>0s</span><span>' + Math.round(dur/4) + 's</span><span>' + Math.round(dur/2) + 's</span><span>' + Math.round(dur*3/4) + 's</span><span>' + dur.toFixed(0) + 's</span></div>';

function renderSegments(filter) {{
  let items = segs.map((s,i) => ({{...s, _i:i}}));
  if (filter) {{ const q = filter.toLowerCase(); items = items.filter(s => (s.action||'').toLowerCase().includes(q) || (s.action_zh||'').includes(q) || (s.caption||'').includes(q) || (s.objects||[]).some(o => o.includes(q))); }}
  document.getElementById('count-badge').textContent = items.length + '/' + segs.length + ' segments';
  if (!items.length) {{ document.getElementById('segments').innerHTML = '<div class="empty-note">无匹配片段</div>'; return; }}
  document.getElementById('segments').innerHTML = items.map(s => {{
    const vstat = s.verification_status || 'unverified';
    return '<div class="seg-card" id="card-' + s._i + '" onclick="seek(' + s.start + ',' + s._i + ')">' +
      '<div class="time"><strong>' + s.start.toFixed(0) + 's – ' + s.end.toFixed(0) + 's</strong> (' + (s.end-s.start).toFixed(0) + 's)</div>' +
      '<div class="action"><span class="tag" style="background:' + (actionColor[s.action]||PALETTE[0]) + '">' + (s.action_zh || s.action) + '</span> ' + (s.action||'') + '</div>' +
      '<div class="caption">' + (s.caption||'') + '</div>' +
      '<div class="meta">' +
        '<span class="objects">📦 ' + (s.objects||[]).join(', ') + '</span>' +
        '<span class="verif ' + vstat + '">' + (vstat==='verified'?'✅':vstat==='partial'?'⚠️':vstat==='rejected'?'❌':'❓') + ' ' + vstat + ' conf=' + (typeof s.confidence==='number'?s.confidence.toFixed(2):s.confidence) + '</span>' +
        ((s.evidence_timestamps||[]).length ? '<span class="evidence">📍 ' + (s.evidence_timestamps||[]).join('s, ') + 's</span>' : '') +
        (s.needs_human_review ? '<span style="color:var(--warn)">🛑 需人工复核</span>' : '') +
      '</div></div>';
  }}).join('');
}}
renderSegments();

let sel = -1;
function seek(t, i) {{
  const doSeek = () => {{ video.currentTime = t; video.play().catch(()=>{{}}); }};
  if (video.readyState >= 2) doSeek(); else video.addEventListener('loadedmetadata', doSeek, {{once:true}});
  document.querySelectorAll('.seg-card.selected,.track-bar.active').forEach(e=>e.classList.remove('selected','active'));
  if (i>=0) {{ document.getElementById('card-'+i)?.classList.add('selected'); document.getElementById('card-'+i)?.scrollIntoView({{behavior:'smooth',block:'center'}}); document.getElementById('bar-'+i)?.classList.add('active'); }}
  sel = i;
}}
document.getElementById('track-bar-container').addEventListener('click', e => {{
  const pct = (e.clientX - e.currentTarget.getBoundingClientRect().left) / e.currentTarget.offsetWidth;
  let best=0, bd=Infinity;
  segs.forEach((s,i)=>{{ const m=(s.start+s.end)/2; if(Math.abs(m-pct*dur)<bd){{bd=Math.abs(m-pct*dur);best=i;}} }});
  seek(segs[best].start, best);
}});
document.getElementById('btn-play').addEventListener('click', ()=>{{ video.currentTime=0; video.play().catch(()=>{{}}); }});
document.getElementById('search').addEventListener('input', e => renderSegments(e.target.value||null));
document.addEventListener('keydown', e => {{
  if (e.key==='ArrowDown'||e.key==='j') {{ e.preventDefault(); seek(segs[Math.min(sel+1,segs.length-1)].start, Math.min(sel+1,segs.length-1)); }}
  else if (e.key==='ArrowUp'||e.key==='k') {{ e.preventDefault(); seek(segs[Math.max(sel-1,0)].start, Math.max(sel-1,0)); }}
  else if (e.key===' ') {{ e.preventDefault(); video.paused?video.play():video.pause(); }}
}});
segs.forEach((s,i) => document.getElementById('bar-'+i)?.addEventListener('mouseenter', ()=>document.getElementById('hover-info').textContent='['+s.start.toFixed(0)+'-'+s.end.toFixed(0)+'s] '+s.action_zh));
document.getElementById('track-bar-container')?.addEventListener('mouseleave', ()=>document.getElementById('hover-info').textContent='');
</script>
</body>
</html>"""

# Build viewer data
viewer_data = {
    "video_id": VIDEO_ID,
    "video_path": str(VIDEO_PATH),
    "duration": video.duration,
    "fps": video.fps,
    "width": video.width,
    "height": video.height,
    "omni_model": config.OMNI_MODEL,
    "phase": prediction.get("phase"),
    "phase_zh": prediction.get("phase_zh"),
    "phase_confidence": prediction.get("phase_confidence"),
    "phase_reason": "",
    "alternative_phases": prediction.get("alternative_phases", []),
    "video_summary": video_summary,
    "segments": prediction.get("segments", []),
}

viewer_html = _VIEWER_TEMPLATE.format(
    video_id=VIDEO_ID,
    phase_zh=prediction.get("phase_zh", ""),
    phase_confidence=prediction.get("phase_confidence", ""),
    n_segs=len(segments),
    omni_model=config.OMNI_MODEL,
    duration=video.duration,
    video_path=str(VIDEO_PATH),
    video_summary=video_summary,
    data_json=json.dumps(viewer_data, ensure_ascii=False),
)
viewer_path = OUT_DIR / "viewer.html"
viewer_path.write_text(viewer_html, encoding="utf-8")
print(f"Viewer: {viewer_path}")
print(f"\nDone in {dt:.1f}s")
