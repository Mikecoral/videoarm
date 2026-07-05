"""Demo server: browse videos, play with segment timeline, view VQA.

    python -m agent_v2.demo_server [--port 8080] [--predictions outputs/full_pipeline/predictions.json]
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import re

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_AI_LAB = _REPO.parent

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LabARM-HV Demo</title>
<style>
:root { --bg:#f5f6f8; --panel:#fff; --line:#e2e5ea; --text:#1a1d23; --muted:#6b7280; --accent:#2563eb; --accent-soft:#eef3ff; --ok:#16a34a; --warn:#ea580c }
* { box-sizing:border-box; margin:0; padding:0 }
body { font:14px/1.5 -apple-system,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text) }
header { padding:20px 28px; background:var(--panel); border-bottom:1px solid var(--line) }
h1 { font-size:20px; margin-bottom:12px }
.stats { display:flex; gap:24px; flex-wrap:wrap }
.stat { padding:6px 14px; background:var(--accent-soft); border-radius:8px; font-size:13px }
.stat b { color:var(--accent) }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; padding:20px }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; cursor:pointer; transition:all .15s }
.card:hover { border-color:var(--accent); box-shadow:0 2px 8px rgba(0,0,0,.08) }
.card h3 { font-size:15px; margin-bottom:6px }
.card .meta { font-size:12px; color:var(--muted); display:flex; flex-wrap:wrap; gap:8px; margin-bottom:6px }
.card .phase { padding:1px 8px; background:var(--accent-soft); color:var(--accent); border-radius:10px; font-weight:600 }
.card .summary { font-size:13px; color:var(--muted); line-height:1.5; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden }
.card .actions { margin-top:8px; display:flex; gap:8px }
.btn { padding:4px 14px; border:1px solid var(--line); border-radius:6px; background:#fff; cursor:pointer; font-size:12px; text-decoration:none; color:var(--text) }
.btn:hover { background:var(--accent-soft); border-color:var(--accent) }
.btn-primary { background:var(--accent); color:#fff; border-color:var(--accent) }
.empty { text-align:center; padding:60px; color:var(--muted) }
</style>
</head>
<body>
<header>
<h1>LabARM-HV: 实验视频解析 Demo</h1>
<div class="stats">{stats}</div>
</header>
<div class="grid">{cards}</div>
</body>
</html>"""

_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    """HTTP handler with Range support for video seeking."""
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            super().do_GET()
            return
        fsize = os.path.getsize(path)
        rh = self.headers.get("Range")
        if not rh:
            super().do_GET()
            return
        m = _RANGE_RE.match(rh)
        if not m:
            self.send_error(416); return
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else fsize - 1
        if start >= fsize or end >= fsize:
            self.send_error(416); return
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(length))


def main(argv=None):
    ap = argparse.ArgumentParser(description="LabARM-HV Demo Server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--predictions", default=None)
    args = ap.parse_args(argv)

    pred_path = Path(args.predictions) if args.predictions else (_HERE / "outputs" / "full_pipeline" / "predictions.json")
    videos = []
    if pred_path.exists():
        data = json.loads(pred_path.read_text())
        videos = data if isinstance(data, list) else []

    # Generate index page
    stats_html = f'<span class="stat"><b>{len(videos)}</b> 视频</span>'
    total_segs = sum(len(v.get("segments", [])) for v in videos)
    stats_html += f'<span class="stat"><b>{total_segs}</b> 片段</span>'
    total_vqa = sum(len(v.get("vqa", [])) for v in videos)
    stats_html += f'<span class="stat"><b>{total_vqa}</b> VQA</span>'

    cards = ""
    for v in videos:
        vid = v["video_id"]
        segs = v.get("segments", [])
        summary = v.get("video_summary", "")[:150]
        phase = v.get("phase_zh", v.get("phase", ""))
        cards += f"""<div class="card">
<h3>Video {vid}</h3>
<div class="meta"><span class="phase">{phase}</span><span>{len(segs)} 段</span></div>
<div class="summary">{summary}</div>
<div class="actions">
<a class="btn btn-primary" href="/viewer/{vid}.html">查看分段</a>
<a class="btn" href="/video/{vid}.mp4">下载视频</a>
</div>
</div>"""

    index_html = INDEX_HTML.format(stats=stats_html, cards=cards or '<div class="empty">未找到解析结果。请先运行 run_full.py 生成 predictions.json。</div>')

    # Serve from repo root so both viewer/ and video/ paths resolve
    os.chdir(_REPO)

    # Write index
    index_path = _HERE / "outputs" / "demo_index.html"
    index_path.parent.mkdir(exist_ok=True)
    index_path.write_text(index_html, encoding="utf-8")

    # Custom handler for /viewer/ and /video/ routes
    class DemoHandler(RangeHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(index_html.encode())
                return
            if self.path.startswith("/viewer/"):
                # Serve from outputs/full_pipeline/viewers/
                fname = self.path.split("/")[-1]
                fpath = _HERE / "outputs" / "full_pipeline" / "viewers" / fname
                if fpath.exists():
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(fpath.read_bytes())
                    return
            if self.path.startswith("/video/"):
                # Serve from dataset
                fname = self.path.split("/")[-1]
                fpath = Path(os.environ.get("LABARM_DATA_ROOT",
                          str(_AI_LAB / "视频解析智能体数据包/hackathon_release"))) / "videos" / fname
                if fpath.exists():
                    self.path = str(fpath)
            super().do_GET()

    server = HTTPServer(("", args.port), DemoHandler)
    print(f"\n  LabARM-HV Demo: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
