from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>实验视频 VQA 展示</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --panel: #ffffff;
      --line: #d8dee9;
      --text: #172033;
      --muted: #657084;
      --accent: #2563eb;
      --accent-soft: #e8f0ff;
      --ok: #15803d;
      --ok-soft: #e9f8ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 22px 28px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { margin: 0 0 12px; font-size: 22px; }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(160px, 1fr) minmax(120px, 150px) minmax(140px, 170px) minmax(140px, 170px) minmax(120px, 150px) minmax(200px, 1.2fr);
      gap: 10px;
      align-items: center;
    }
    select, input, button {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    button {
      cursor: pointer;
      background: #fff;
    }
    button:hover { background: #f5f7fb; }
    button:disabled {
      cursor: not-allowed;
      color: #9aa4b2;
      background: #f3f5f8;
    }
    main {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
    }
    .sidebar, .detail {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: calc(100vh - 120px);
    }
    .sidebar {
      overflow: auto;
      max-height: calc(100vh - 120px);
    }
    #list {
      display: block;
    }
    .stats {
      padding: 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .pager {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: minmax(68px, 78px) minmax(78px, 1fr) minmax(68px, 78px);
      gap: 8px;
      align-items: center;
    }
    .pager input { text-align: center; }
    .page-note {
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
      line-height: 1.3;
    }
    .item {
      width: 100%;
      display: block;
      min-height: 74px;
      height: auto;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      text-align: left;
      padding: 12px;
      cursor: pointer;
      white-space: normal;
      overflow: visible;
      line-height: 1.45;
    }
    .item:hover { background: #f5f7fb; }
    .item.active {
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      padding-left: 8px;
    }
    .item .meta {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .item .q {
      display: block;
      font-size: 14px;
      line-height: 1.4;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .detail {
      padding: 18px;
      overflow: auto;
    }
    .empty {
      color: var(--muted);
      padding: 40px;
      text-align: center;
    }
    .row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 8px 0 16px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #eef2f7;
      color: #344054;
      font-size: 12px;
    }
    .badge.ok { background: var(--ok-soft); color: var(--ok); }
    .section {
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 14px;
    }
    .section h2 {
      font-size: 15px;
      margin: 0 0 10px;
    }
    .question {
      font-size: 20px;
      line-height: 1.45;
      margin: 4px 0 14px;
    }
    .media-layout {
      display: grid;
      grid-template-columns: minmax(300px, 1.15fr) minmax(260px, 0.85fr);
      gap: 14px;
      align-items: start;
    }
    .video-panel, .image-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
    }
    .video-panel video {
      width: 100%;
      max-height: 420px;
      display: block;
      background: #0f172a;
      border-radius: 6px;
    }
    .media-title {
      color: var(--muted);
      font-size: 13px;
      margin: 0 0 8px;
    }
    .answer-area {
      border: 1px dashed #b7c1d1;
      border-radius: 8px;
      min-height: 86px;
      background: #fbfcff;
      color: var(--muted);
      padding: 12px;
      line-height: 1.5;
    }
    .answer-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcff;
      line-height: 1.55;
    }
    .options {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .option {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      background: #fff;
      min-height: 48px;
      line-height: 1.45;
    }
    .option.correct {
      border-color: #22c55e;
      background: var(--ok-soft);
    }
    .letter {
      font-weight: 700;
      margin-right: 6px;
    }
    .kv {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      gap: 8px 12px;
      font-size: 14px;
      line-height: 1.5;
    }
    .kv .k { color: var(--muted); }
    .frames {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 10px;
    }
    .frame {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .frame img {
      width: 100%;
      max-height: 180px;
      object-fit: contain;
      border-radius: 6px;
      background: #eef2f7;
      border: 1px solid #edf0f5;
      margin-bottom: 8px;
    }
    .frame-path {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: #f1f5f9;
      border-radius: 5px;
      padding: 2px 4px;
    }
    @media (max-width: 860px) {
      header { position: static; }
      .toolbar { grid-template-columns: 1fr; }
      main { grid-template-columns: 1fr; }
      .sidebar { max-height: 360px; min-height: auto; }
      .detail { min-height: auto; }
      .options { grid-template-columns: 1fr; }
      .media-layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>实验视频 VQA 展示</h1>
    <div class="toolbar">
      <select id="datasetSelect"></select>
      <select id="videoSelect"></select>
      <select id="typeSelect"></select>
      <select id="formSelect">
        <option value="both">问答 + 选择题</option>
        <option value="open">仅问答题</option>
        <option value="mc">仅选择题</option>
      </select>
      <select id="viewSelect">
        <option value="answerer">答题视图</option>
        <option value="annotation">标注视图</option>
      </select>
      <input id="searchInput" placeholder="搜索问题 / 视频编号 / 题型" />
    </div>
  </header>
  <main>
    <aside class="sidebar">
      <div class="stats" id="stats"></div>
      <div class="pager" id="pager"></div>
      <div id="list"></div>
    </aside>
    <section class="detail" id="detail">
      <div class="empty">请选择左侧 VQA 条目</div>
    </section>
  </main>
  <script id="embedded-data" type="application/json">__DATA__</script>
  <script>
    const DATASETS = JSON.parse(document.getElementById("embedded-data").textContent);
    const PAGE_SIZE = 20;
    const state = { dataset: 0, video: "all", type: "all", form: "both", view: "answerer", query: "", selected: 0 };
    const els = {
      dataset: document.getElementById("datasetSelect"),
      video: document.getElementById("videoSelect"),
      type: document.getElementById("typeSelect"),
      form: document.getElementById("formSelect"),
      view: document.getElementById("viewSelect"),
      search: document.getElementById("searchInput"),
      stats: document.getElementById("stats"),
      pager: document.getElementById("pager"),
      list: document.getElementById("list"),
      detail: document.getElementById("detail"),
    };

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    function activeDataset() {
      return DATASETS[state.dataset] || DATASETS[0];
    }

    function allItems() {
      return activeDataset().vqa || [];
    }

    function filteredItems() {
      const query = state.query.trim().toLowerCase();
      return allItems().filter(item => {
        if (state.video !== "all" && item.video_id !== state.video) return false;
        if (state.type !== "all" && item.question_type !== state.type) return false;
        if (!query) return true;
        const hay = [
          item.id, item.video_id, item.segment_id, item.question, item.question_type,
          item.multiple_choice ? Object.values(item.multiple_choice.options || {}).join(" ") : ""
        ].join(" ").toLowerCase();
        return hay.includes(query);
      });
    }

    function initControls() {
      els.dataset.innerHTML = DATASETS.map((d, i) => {
        const label = `${d.name} (${d.num_vqa}题)`;
        return `<option value="${i}">${esc(label)}</option>`;
      }).join("");
      els.dataset.addEventListener("change", () => {
        state.dataset = Number(els.dataset.value);
        state.video = "all";
        state.type = "all";
        state.selected = 0;
        renderControls();
        render();
      });
      els.video.addEventListener("change", () => {
        state.video = els.video.value;
        state.selected = 0;
        render();
      });
      els.type.addEventListener("change", () => {
        state.type = els.type.value;
        state.selected = 0;
        render();
      });
      els.form.addEventListener("change", () => {
        state.form = els.form.value;
        render();
      });
      els.view.addEventListener("change", () => {
        state.view = els.view.value;
        render();
      });
      els.search.addEventListener("input", () => {
        state.query = els.search.value;
        state.selected = 0;
        render();
      });
      renderControls();
    }

    function renderControls() {
      const items = allItems();
      const videos = [...new Set(items.map(x => x.video_id))].sort();
      const types = [...new Set(items.map(x => x.question_type))].sort();
      els.video.innerHTML = `<option value="all">全部视频</option>` + videos.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
      els.type.innerHTML = `<option value="all">全部题型</option>` + types.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join("");
      els.video.value = state.video;
      els.type.value = state.type;
      els.form.value = state.form;
      els.view.value = state.view;
      els.dataset.value = String(state.dataset);
    }

    function render() {
      const items = filteredItems();
      if (state.selected >= items.length) state.selected = 0;
      const selectedDisplay = items.length ? state.selected + 1 : 0;
      const page = Math.floor(state.selected / PAGE_SIZE);
      const pageStart = page * PAGE_SIZE;
      const pageEnd = Math.min(pageStart + PAGE_SIZE, items.length);
      const pageItems = items.slice(pageStart, pageEnd);
      const pageCount = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
      els.stats.innerHTML = [
        `数据源：<code>${esc(activeDataset().file)}</code>`,
        `显示：${items.length} / ${allItems().length} 题`,
        `当前：${selectedDisplay} / ${items.length}`,
        `页码：${page + 1} / ${pageCount}`,
        `segment：${activeDataset().num_segments}`,
      ].join("<br>");
      els.pager.innerHTML = `
        <button id="prevBtn" ${state.selected <= 0 ? "disabled" : ""}>上一题</button>
        <input id="jumpInput" type="number" min="1" max="${items.length}" value="${selectedDisplay}" title="输入题号后回车跳转" />
        <button id="nextBtn" ${state.selected >= items.length - 1 ? "disabled" : ""}>下一题</button>
        <div class="page-note">本页 ${pageStart + 1}-${pageEnd}；输入 1-${items.length} 的题号后按 Enter 跳转</div>
      `;
      els.pager.querySelector("#prevBtn")?.addEventListener("click", () => {
        state.selected = Math.max(0, state.selected - 1);
        render();
      });
      els.pager.querySelector("#nextBtn")?.addEventListener("click", () => {
        state.selected = Math.min(items.length - 1, state.selected + 1);
        render();
      });
      els.pager.querySelector("#jumpInput")?.addEventListener("keydown", event => {
        if (event.key !== "Enter") return;
        const value = Number(event.target.value);
        if (!Number.isFinite(value)) return;
        state.selected = Math.max(0, Math.min(items.length - 1, Math.floor(value) - 1));
        render();
      });
      els.list.innerHTML = pageItems.map((item, localIdx) => {
        const idx = pageStart + localIdx;
        return `
        <button class="item ${idx === state.selected ? "active" : ""}" data-idx="${idx}">
          <div class="meta">#${idx + 1} · 视频 ${esc(item.video_id)} · ${esc(item.level || "L1")} · ${esc(item.category || item.question_type)} · ${esc(item.visual_modality || item.visual?.modality || "video")}</div>
          <div class="q">${esc(item.question)}</div>
        </button>
      `;
      }).join("");
      [...els.list.querySelectorAll(".item")].forEach(btn => {
        btn.addEventListener("click", () => {
          state.selected = Number(btn.dataset.idx);
          render();
        });
      });
      renderDetail(items[state.selected]);
    }

    function renderDetail(item) {
      if (!item) {
        els.detail.innerHTML = `<div class="empty">没有匹配的 VQA 条目</div>`;
        return;
      }
      const bg = item.background || {};
      const mc = item.multiple_choice || {};
      const gt = item.ground_truth || {};
      const mcGt = gt.multiple_choice || mc.ground_truth || {};
      const openGt = gt.open_ended || item.open_ended?.ground_truth || {};
      const options = mc.options || {};
      const frames = bg.keyframes || [];
      const showOpen = state.form === "both" || state.form === "open";
      const showMc = state.form === "both" || state.form === "mc";
      const isAnnotation = state.view === "annotation";
      const visual = item.visual || {};
      const modality = item.visual_modality || visual.modality || "video";
      const selectedFrame = Number.isInteger(visual.keyframe_index) ? frames[visual.keyframe_index] : null;
      const imagePath = visual.image_path || selectedFrame?.frame_path || frames[0]?.frame_path || "";
      const clipPath = visual.clip_path || "";
      const fallbackVideoSrc = bg.video_path ? `${relativeVideoPath(bg.video_path)}#t=${Number(item.time_range?.start || 0)},${Number(item.time_range?.end || 0)}` : "";
      const videoSrc = clipPath ? relativeMediaPath(clipPath) : fallbackVideoSrc;
      const mediaHtml = `
        <div class="section">
          <h2>可见材料</h2>
          ${modality === "image" ? `
            <div class="image-panel">
              <div class="media-title">图片</div>
              ${imagePath ? `<img src="${esc(relativeMediaPath(imagePath))}" alt="题目图片" style="width:100%;max-height:520px;object-fit:contain;background:#eef2f7;border-radius:6px;border:1px solid #edf0f5;" onerror="this.style.display='none'">` : `<div class="empty">无图片</div>`}
              ${isAnnotation && selectedFrame ? `<div class="section">
                <div class="kv">
                  <div class="k">图片序号</div><div>${esc(visual.keyframe_index)}</div>
                  <div class="k">时间戳</div><div>${esc(selectedFrame.time)}s</div>
                  <div class="k">frame</div><div>${esc(selectedFrame.frame_index)}</div>
                  <div class="k">图片 caption</div><div>${esc(selectedFrame.caption)}</div>
                  <div class="k">图片路径</div><div class="frame-path">${esc(imagePath)}</div>
                </div>
              </div>` : ""}
            </div>
          ` : `
            <div class="video-panel">
              <div class="media-title">视频片段</div>
              ${videoSrc ? `<video controls preload="metadata" src="${esc(videoSrc)}" ${clipPath ? "" : `data-end="${esc(item.time_range?.end || "")}"`}></video>` : `<div class="empty">无视频片段</div>`}
              ${isAnnotation ? `<div class="section">
                <div class="kv">
                  <div class="k">clip 路径</div><div class="frame-path">${esc(clipPath || "未生成，使用原视频时间片段回退")}</div>
                  <div class="k">原视频</div><div class="frame-path">${esc(bg.video_path || "")}</div>
                  <div class="k">时间范围</div><div>${esc(item.time_range?.start)}s - ${esc(item.time_range?.end)}s</div>
                </div>
              </div>` : ""}
            </div>
          `}
        </div>
      `;
      els.detail.innerHTML = `
        <div class="row">
          <span class="badge">${esc(item.id)}</span>
          <span class="badge">${esc(item.video_id)}</span>
          <span class="badge">${esc(item.level || "L1")}</span>
          <span class="badge">${esc(item.category || "operation")}</span>
          <span class="badge">${esc(modality)}</span>
          <span class="badge">${esc(item.question_type)}</span>
          <span class="badge ok">${esc(item.generation_method)}</span>
          <span class="badge">题号 ${state.selected + 1}</span>
          <span class="badge">形式：${esc(els.form.options[els.form.selectedIndex]?.text || "")}</span>
          <span class="badge">视图：${esc(els.view.options[els.view.selectedIndex]?.text || "")}</span>
        </div>
        ${mediaHtml}
        <div class="question">${esc(item.question)}</div>

        ${showOpen ? `<div class="section">
          <h2>开放式问答</h2>
          <div class="${isAnnotation ? "answer-box" : "answer-area"}">
            <div><strong>Question：</strong>${esc(item.open_ended?.question || item.question)}</div>
            ${isAnnotation ? `<div><strong>Ground Truth：</strong>${esc(openGt.answer_text || item.answer)}</div>` : `<div>作答区</div>`}
          </div>
        </div>` : ""}

        ${showMc ? `<div class="section">
          <h2>选择题</h2>
          <div class="options">
            ${Object.entries(options).map(([letter, text]) => `
              <div class="option ${isAnnotation && letter === mcGt.answer ? "correct" : ""}">
                <span class="letter">${esc(letter)}.</span>${esc(text)}
              </div>
            `).join("")}
          </div>
          ${isAnnotation ? `<div class="row">
            <span class="badge ok">答案：${esc(mcGt.answer)} · ${esc(mcGt.answer_text)}</span>
          </div>` : ""}
        </div>` : ""}

        ${isAnnotation ? `<div class="section">
          <h2>证据与背景</h2>
          <div class="kv">
            <div class="k">时间范围</div><div>${esc(item.time_range?.start)}s - ${esc(item.time_range?.end)}s</div>
            <div class="k">证据时间戳</div><div>${esc((item.evidence_timestamps || []).join(", "))}</div>
            <div class="k">阶段</div><div>${esc(bg.phase_zh || bg.phase)}</div>
            <div class="k">动作</div><div>${esc(bg.action_zh || bg.action)}</div>
            <div class="k">片段 caption</div><div>${esc(bg.segment_caption)}</div>
            <div class="k">上一片段</div><div>${esc(bg.previous_caption || "无")}</div>
            <div class="k">下一片段</div><div>${esc(bg.next_caption || "无")}</div>
            <div class="k">答案依据</div><div>${esc(item.answer_basis)}</div>
            <div class="k">视觉对应</div><div>${esc(item.visual_reference || "未标注")}</div>
            <div class="k">证据说明</div><div>${esc(item.evidence_description || "未标注")}</div>
          </div>
        </div>` : ""}
      `;
      wireSegmentVideos();
    }

    function wireSegmentVideos() {
      [...document.querySelectorAll("video[data-end]")].forEach(video => {
        const end = Number(video.dataset.end);
        if (!Number.isFinite(end)) return;
        video.addEventListener("timeupdate", () => {
          if (video.currentTime >= end) video.pause();
        });
      });
    }

    function relativeFramePath(path) {
      return relativeMediaPath(path);
    }

    function relativeMediaPath(path) {
      const marker = "/hackathon_release/";
      const idx = path.indexOf(marker);
      if (idx >= 0) return "../" + path.slice(idx + marker.length);
      if (path.startsWith("视频解析智能体数据包/hackathon_release/")) {
        return "../" + path.slice("视频解析智能体数据包/hackathon_release/".length);
      }
      if (path.startsWith("vqa/")) return path.slice("vqa/".length);
      if (path.startsWith("clips/")) return path;
      return path;
    }

    function relativeVideoPath(path) {
      const marker = "/hackathon_release/";
      const idx = path.indexOf(marker);
      if (idx >= 0) return "../" + path.slice(idx + marker.length);
      if (path.startsWith("视频解析智能体数据包/hackathon_release/")) {
        return "../" + path.slice("视频解析智能体数据包/hackathon_release/".length);
      }
      if (path.startsWith("videos/")) return "../" + path;
      return path;
    }

    initControls();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static VQA viewer HTML.")
    parser.add_argument("--vqa-dir", default="视频解析智能体数据包/hackathon_release/vqa")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    vqa_dir = Path(args.vqa_dir)
    preferred = [
        vqa_dir / "vqa_output_dev_mllm.json",
        vqa_dir / "vqa_output_new_test_mllm.json",
    ]
    skip_tokens = ("_raw", "_template", "_iter", "_regen")
    files = [path for path in preferred if path.exists()]
    for path in sorted(vqa_dir.glob("vqa_output*.json")):
        if path in files or any(token in path.stem for token in skip_tokens):
            continue
        files.append(path)
    datasets = []
    for path in files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["file"] = path.name
        data["name"] = path.stem.replace("vqa_output_", "").replace("_template", "")
        datasets.append(data)
    if not datasets:
        raise SystemExit(f"No VQA JSON files found in {vqa_dir}")

    output = Path(args.output) if args.output else vqa_dir / "viewer.html"
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(datasets, ensure_ascii=False))
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output}")
    print(f"datasets: {', '.join(d['file'] for d in datasets)}")


if __name__ == "__main__":
    main()
