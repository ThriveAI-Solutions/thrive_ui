"""Render an eval results dict into one self-contained HTML report.

PHI lives in this file — it must never load external resources, so all
CSS/JS/fonts/charting is inlined from ``evals/assets/`` (see that dir's
README for how the vendored bundles are produced). Charts are Apache ECharts;
the display/body type is IBM Plex, base64-embedded. Verdicts persist to
localStorage keyed by run_id; export/import JSON allows pass-the-file review
without a server.

Assembly note: the report is built by plain concatenation rather than
``string.Template``. The inlined ECharts bundle contains ``$`` characters that
``Template.substitute`` would try (and fail) to interpret as placeholders, so
the only dynamic head fields are injected via unique ``%%TOKEN%%`` markers on a
small skeleton string that never overlaps the large vendored blobs.
"""

from __future__ import annotations

import html as html_mod
import json
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets"


def _asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def generate_html(results: dict) -> str:
    payload = json.dumps(results, default=str).replace("</", "<\\/")
    run_id = html_mod.escape(str(results.get("run_id", "")))
    created_at = html_mod.escape(str(results.get("created_at", "")))
    model = html_mod.escape(
        f"{results.get('model', {}).get('provider', '')}/{results.get('model', {}).get('model', '')}"
    )
    skeleton = _HEAD.replace("%%RUN_ID%%", run_id) + _BODY.replace("%%RUN_ID%%", run_id).replace(
        "%%CREATED_AT%%", created_at
    ).replace("%%MODEL%%", model)
    return "".join(
        [
            skeleton,
            '<script type="application/json" id="eval-data">',
            payload,
            "</script>\n<script>",
            _asset("echarts.min.js"),
            "</script>\n<script>",
            _APP_JS,
            "</script>\n</body>\n</html>\n",
        ]
    )


# _HEAD (built after _CSS below) carries the <style> block (embedded fonts +
# theme); _BODY is the visible scaffold. The data, ECharts bundle, and _APP_JS
# are concatenated after, in generate_html().
_CSS = r"""
:root {
  --bg: #0a0d12; --bg-2: #0e1219; --panel: #131922; --panel-2: #18202b;
  --line: #222c39; --line-2: #2d3a4b;
  --ink: #e8eef7; --ink-dim: #aebbcd; --muted: #6f7e93; --faint: #4a5567;
  --accent: #4fd6c6; --accent-2: #2bb6a6;
  --ok: #46c37b; --bad: #ff6f6f; --meh: #f2b53f;
  --llm: #7c8cff; --ours: #f5b73d; --db: #ff7a6b;
  --shadow: 0 10px 34px rgba(0,0,0,.45);
  --toolbar-1: rgba(10,13,18,.94); --toolbar-2: rgba(10,13,18,.74);
  --track: rgba(255,255,255,.05);
  --mono: 'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace;
  --sans: 'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
/* ---- light theme (toggle): a warm "lab report" palette over the same layout ---- */
:root[data-theme="light"] {
  --bg: #f4f1ea; --bg-2: #efece2; --panel: #ffffff; --panel-2: #f7f5ef;
  --line: #e4dfd2; --line-2: #d4cebe;
  --ink: #1b2531; --ink-dim: #44505f; --muted: #79849a; --faint: #adb3bf;
  --accent: #12a092; --accent-2: #0f8175;
  --ok: #1f9d57; --bad: #d64545; --meh: #b5811a;
  --llm: #5663d6; --ours: #cf9518; --db: #db5a47;
  --shadow: 0 12px 30px rgba(43,47,58,.10);
  --toolbar-1: rgba(244,241,234,.93); --toolbar-2: rgba(244,241,234,.66);
  --track: rgba(22,32,48,.08);
}
:root[data-theme="light"] body::before {
  background:
    radial-gradient(680px 420px at 8% -6%, rgba(18,160,146,.12), transparent 60%),
    radial-gradient(720px 520px at 102% 0%, rgba(86,99,214,.10), transparent 58%),
    repeating-linear-gradient(0deg, rgba(22,32,48,.028) 0 1px, transparent 1px 40px),
    repeating-linear-gradient(90deg, rgba(22,32,48,.028) 0 1px, transparent 1px 40px);
}
:root[data-theme="light"] .phi {
  color: #b22b2b; background: rgba(214,69,69,.07);
  border-color: rgba(214,69,69,.34); box-shadow: none;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; font-family: var(--sans); color: var(--ink);
  background: var(--bg); line-height: 1.5; letter-spacing: .005em;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
}
/* atmosphere: faint twin glows + a hairline grid, all very low contrast */
body::before {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(680px 420px at 8% -6%, rgba(79,214,198,.10), transparent 60%),
    radial-gradient(720px 520px at 102% 0%, rgba(124,140,255,.10), transparent 58%),
    repeating-linear-gradient(0deg, rgba(255,255,255,.014) 0 1px, transparent 1px 40px),
    repeating-linear-gradient(90deg, rgba(255,255,255,.014) 0 1px, transparent 1px 40px);
}
.wrap { position: relative; z-index: 1; max-width: 1140px; margin: 0 auto; padding: 30px 22px 110px; }

/* ---- hero ---- */
.hero { padding: 6px 0 14px; }
.hero-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.brand {
  display: inline-flex; align-items: center; gap: 13px;
  font-family: var(--mono); font-weight: 600; font-size: 21px;
  letter-spacing: .26em; color: var(--ink); text-transform: uppercase;
}
.brand .mark {
  width: 13px; height: 13px; border-radius: 3px; transform: rotate(45deg);
  background: linear-gradient(135deg, var(--accent), var(--llm));
  box-shadow: 0 0 16px rgba(79,214,198,.6);
}
.brand .sub { color: var(--accent); letter-spacing: .26em; }
.phi {
  font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .14em;
  color: #ffb4b4; background: rgba(255,111,111,.08);
  border: 1px solid rgba(255,111,111,.34); border-radius: 999px; padding: 5px 13px;
  white-space: nowrap; box-shadow: 0 0 22px rgba(255,111,111,.10) inset;
}
.phi .pulse { color: var(--bad); margin-right: 7px; animation: blink 2.4s ease-in-out infinite; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
.meta {
  font-family: var(--mono); font-size: 12.5px; color: var(--muted);
  margin-top: 14px; letter-spacing: .02em;
}
.meta b { color: var(--ink-dim); font-weight: 600; }
.meta i { color: var(--faint); font-style: normal; margin: 0 9px; }
.accent-rule {
  height: 2px; margin-top: 16px; border-radius: 2px;
  background: linear-gradient(90deg, var(--accent) 0%, rgba(124,140,255,.7) 22%, var(--line) 60%, transparent 100%);
}

/* ---- toolbar ---- */
.toolbar {
  position: sticky; top: 0; z-index: 20; display: flex; gap: 10px; align-items: center;
  margin: 18px 0 6px; padding: 11px 0;
  background: linear-gradient(180deg, var(--toolbar-1), var(--toolbar-2));
  backdrop-filter: blur(9px); -webkit-backdrop-filter: blur(9px);
  border-bottom: 1px solid var(--line);
}
.btn {
  font-family: var(--mono); font-size: 12.5px; letter-spacing: .03em;
  color: var(--ink-dim); background: var(--panel); border: 1px solid var(--line-2);
  border-radius: 9px; padding: 8px 15px; cursor: pointer; transition: all .16s ease;
  display: inline-flex; align-items: center; gap: 8px;
}
.btn:hover { color: var(--ink); border-color: var(--accent); box-shadow: 0 0 0 1px rgba(79,214,198,.25), 0 6px 18px rgba(0,0,0,.35); transform: translateY(-1px); }
.btn.ghost { background: transparent; }
.btn.ghost:hover { color: var(--bad); border-color: var(--bad); box-shadow: 0 0 0 1px rgba(255,111,111,.22); }
.btn.theme { margin-left: auto; }
.btn.theme .tdot {
  width: 13px; height: 13px; border-radius: 50%; border: 1.5px solid currentColor;
  background: linear-gradient(90deg, currentColor 0 50%, transparent 50% 100%);
}

/* ---- section chrome ---- */
.panel {
  background: linear-gradient(180deg, var(--panel), var(--bg-2));
  border: 1px solid var(--line); border-radius: 16px; padding: 20px 22px;
  margin: 18px 0; box-shadow: var(--shadow);
}
.sec-head { display: flex; align-items: baseline; gap: 12px; margin: 0 0 16px; }
.sec-kicker {
  font-family: var(--mono); font-size: 10.5px; font-weight: 600; letter-spacing: .24em;
  text-transform: uppercase; color: var(--accent);
  border: 1px solid rgba(79,214,198,.3); border-radius: 6px; padding: 3px 8px;
}
.sec-head h2 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: .01em; color: var(--ink); }
h2.bare { font-size: 16px; margin: 30px 4px 12px; font-weight: 600; }

/* ---- scorecard ---- */
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 13px; }
.stat {
  position: relative; background: var(--panel-2); border: 1px solid var(--line);
  border-radius: 12px; padding: 15px 16px 14px; overflow: hidden;
}
.stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent); opacity: .8; }
.stat.good::before { background: var(--ok); } .stat.bad::before { background: var(--bad); } .stat.meh::before { background: var(--meh); }
.stat .n { font-family: var(--mono); font-size: 29px; font-weight: 600; line-height: 1.05; color: var(--ink); font-variant-numeric: tabular-nums; }
.stat .n i { font-style: normal; font-size: 16px; color: var(--muted); }
.stat .l { font-size: 11.5px; color: var(--muted); margin-top: 5px; letter-spacing: .02em; text-transform: uppercase; }

.qrow { display: grid; grid-template-columns: 1fr 200px 96px; gap: 14px; align-items: center; font-size: 13px; margin: 9px 2px; }
.qrow .qname { color: var(--ink-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.qbar { background: var(--track); border-radius: 6px; height: 9px; overflow: hidden; display: flex; box-shadow: inset 0 0 0 1px var(--line); }
.qbar span { height: 100%; display: block; }
.qbar .qok { background: linear-gradient(90deg, var(--accent-2), var(--ok)); }
.qbar .qbad { background: linear-gradient(90deg, #b5424a, var(--bad)); }
.qrow .qval { font-family: var(--mono); font-size: 11.5px; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.qrow.empty .qval { color: var(--faint); }

/* ---- charts ---- */
.time-grid { display: grid; grid-template-columns: 230px 1fr; gap: 22px; align-items: center; }
.donut { width: 100%; height: 220px; }
.legend2 { display: flex; flex-direction: column; gap: 4px; }
.legrow { display: grid; grid-template-columns: 14px 1fr auto auto; gap: 11px; align-items: baseline; padding: 9px 2px; border-bottom: 1px solid var(--line); }
.legrow:last-child { border-bottom: none; }
.legdot { width: 11px; height: 11px; border-radius: 3px; align-self: center; }
.leglabel { color: var(--ink-dim); font-size: 13px; }
.legpct { font-family: var(--mono); font-size: 19px; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.legms { font-family: var(--mono); font-size: 11.5px; color: var(--muted); min-width: 56px; text-align: right; }
.percard { width: 100%; margin-top: 16px; }
.muted-note { color: var(--muted); font-size: 13px; font-family: var(--mono); padding: 14px 2px; }

/* ---- result cards ---- */
.convo-head {
  display: flex; align-items: center; gap: 10px;
  font-family: var(--mono); font-size: 12.5px; color: var(--accent);
  margin: 26px 2px 10px; letter-spacing: .03em;
}
.convo-head::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 10px var(--accent); }
.convo-head .pname { color: var(--ink-dim); } .convo-head .plabel { color: var(--muted); }
.note {
  font-size: 12.5px; color: var(--meh); background: rgba(242,181,63,.07);
  border: 1px solid rgba(242,181,63,.28); border-left-width: 3px; border-radius: 8px;
  padding: 9px 13px; margin: 10px 2px;
}
.card {
  position: relative; background: var(--panel); border: 1px solid var(--line);
  border-left: 3px solid var(--line-2); border-radius: 12px;
  padding: 15px 17px; margin: 11px 0; box-shadow: 0 4px 16px rgba(0,0,0,.22);
  animation: rise .5s cubic-bezier(.2,.7,.3,1) both;
}
@keyframes rise { from { opacity: 0; transform: translateY(9px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .card { animation: none; } .phi .pulse { animation: none; } }
.card.followup { margin-left: 30px; border-left-color: var(--accent-2); }
.card.followup::before { content: ""; position: absolute; left: -30px; top: -7px; width: 30px; height: 28px; border-left: 1px solid var(--line-2); border-bottom: 1px solid var(--line-2); border-bottom-left-radius: 9px; }
.card.errored { border-color: rgba(255,111,111,.4); border-left-color: var(--bad); background: linear-gradient(180deg, rgba(255,111,111,.05), var(--panel)); }
.prompt { font-weight: 600; font-size: 14px; color: var(--ink); }
.answer { white-space: pre-wrap; margin: 9px 0; color: var(--ink-dim); font-size: 13.5px; }
.badges { margin: 7px 0 4px; display: flex; flex-wrap: wrap; gap: 6px; }
.badge {
  font-family: var(--mono); font-size: 10.5px; font-weight: 600; letter-spacing: .03em;
  border-radius: 999px; padding: 3px 10px; border: 1px solid var(--line-2); color: var(--muted);
  background: var(--panel-2); display: inline-flex; align-items: center; gap: 5px;
}
.badge.ok { color: var(--ok); border-color: rgba(70,195,123,.4); background: rgba(70,195,123,.08); }
.badge.bad { color: var(--bad); border-color: rgba(255,111,111,.4); background: rgba(255,111,111,.08); }
.badge.meh { color: var(--meh); border-color: rgba(242,181,63,.4); background: rgba(242,181,63,.08); }
.judge { font-size: 12.5px; color: var(--muted); margin: 9px 0 4px; display: flex; align-items: baseline; gap: 9px; }
.judge .jtext { color: var(--ink-dim); }
details { margin-top: 9px; }
summary {
  font-family: var(--mono); font-size: 12px; color: var(--accent); cursor: pointer;
  list-style: none; display: inline-flex; align-items: center; gap: 7px; padding: 3px 0;
  user-select: none;
}
summary::-webkit-details-marker { display: none; }
summary::before { content: "+"; font-weight: 600; width: 14px; text-align: center; transition: transform .15s ease; }
details[open] summary::before { content: "−"; }
details .trace { margin-top: 9px; border-left: 1px solid var(--line); padding-left: 13px; }
details p { margin: 7px 0 4px; font-size: 12.5px; color: var(--ink-dim); }
details p b { color: var(--ink); font-family: var(--mono); font-weight: 600; }
details pre {
  background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 12px; overflow-x: auto; font-family: var(--mono); font-size: 11px;
  color: var(--ink-dim); margin: 5px 0; line-height: 1.55;
}

/* ---- verdict buttons ---- */
.vbtns { display: flex; gap: 9px; margin-top: 13px; padding-top: 12px; border-top: 1px solid var(--line); }
.vbtn {
  font-family: var(--mono); font-size: 12px; letter-spacing: .03em; padding: 7px 16px;
  border-radius: 8px; border: 1px solid var(--line-2); background: transparent;
  color: var(--muted); cursor: pointer; transition: all .14s ease;
}
.vbtn:hover { color: var(--ink); border-color: var(--ink-dim); }
.vbtn.on-correct { background: var(--ok); border-color: var(--ok); color: #04140a; font-weight: 600; box-shadow: 0 0 18px rgba(70,195,123,.35); }
.vbtn.on-incorrect { background: var(--bad); border-color: var(--bad); color: #1a0404; font-weight: 600; box-shadow: 0 0 18px rgba(255,111,111,.35); }
.vbtn.on-unsure { background: var(--meh); border-color: var(--meh); color: #1c1304; font-weight: 600; box-shadow: 0 0 18px rgba(242,181,63,.32); }

::selection { background: rgba(79,214,198,.28); }
::-webkit-scrollbar { height: 11px; width: 11px; }
::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 6px; border: 2px solid var(--bg); }
::-webkit-scrollbar-track { background: transparent; }
"""

_HEAD = (
    """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic Eval — %%RUN_ID%%</title>
<style>
"""
    + _asset("fonts.css")
    + _CSS
    + """
</style>
</head>
<body>
"""
)

_BODY = """<div class="wrap">
  <header class="hero">
    <div class="hero-row">
      <div class="brand"><span class="mark"></span>AGENTIC EVAL <span class="sub">REPORT</span></div>
      <span class="phi"><span class="pulse">&#9679;</span>CONTAINS PHI &mdash; INTERNAL USE ONLY</span>
    </div>
    <div class="meta">run <b>%%RUN_ID%%</b> <i>/</i> generated %%CREATED_AT%% <i>/</i> model <b>%%MODEL%%</b></div>
    <div class="accent-rule"></div>
  </header>

  <div class="toolbar">
    <button class="btn" onclick="exportVerdicts()">Export verdicts</button>
    <label class="btn">Import<input type="file" accept=".json" hidden onchange="importVerdicts(this)"></label>
    <button class="btn ghost" onclick="clearMarks()">Clear marks</button>
    <button class="btn theme" id="themebtn" onclick="toggleTheme()"><span class="tdot"></span><span class="tlabel"></span></button>
  </div>

  <section class="panel">
    <div class="sec-head"><span class="sec-kicker">verdicts</span><h2>Scorecard</h2></div>
    <div id="scorecard"></div>
  </section>

  <section class="panel">
    <div class="sec-head"><span class="sec-kicker">latency</span><h2>Where does the time go?</h2></div>
    <div class="time-grid">
      <div id="agg-donut" class="donut"></div>
      <div id="agg-stats" class="legend2"></div>
    </div>
    <div id="percard" class="percard"></div>
  </section>

  <section class="panel">
    <div class="sec-head"><span class="sec-kicker">warehouse</span><h2>Slowest queries</h2></div>
    <div id="slowest"></div>
  </section>

  <h2 class="bare">Results</h2>
  <div id="cards"></div>
</div>
"""

_APP_JS = r"""
"use strict";
var DATA = JSON.parse(document.getElementById("eval-data").textContent);
var MARKS_KEY = "evalmarks:" + DATA.run_id;
var CHARTS = [];
// Chart colors can't read CSS vars (canvas), so each theme carries its own.
var THEMES = {
  dark: { llm: "#7c8cff", ours: "#f5b73d", db: "#ff7a6b", ink: "#e8eef7", dim: "#9fb0c3", muted: "#6f7e93", line: "#2d3a4b", axis: "#222c39", split: "rgba(45,58,75,.45)", bg: "#0a0d12", tipbg: "rgba(19,25,34,.97)", dbgrad: ["#ff7a6b", "#ffb39f"] },
  light: { llm: "#5663d6", ours: "#cf9518", db: "#db5a47", ink: "#1b2531", dim: "#46525f", muted: "#79849a", line: "#d4cebe", axis: "#e4dfd2", split: "rgba(40,50,66,.10)", bg: "#ffffff", tipbg: "rgba(255,255,255,.98)", dbgrad: ["#db5a47", "#efa593"] }
};
var COL = THEMES.dark;

function setTheme(name) {
  if (name !== "light") name = "dark";
  COL = THEMES[name];
  document.documentElement.setAttribute("data-theme", name);
  try { localStorage.setItem("evaltheme", name); } catch (e) {}
  var lbl = document.querySelector("#themebtn .tlabel");
  if (lbl) lbl.textContent = name === "light" ? "Dark" : "Light";
  CHARTS.forEach(function (c) { try { c.dispose(); } catch (e) {} });
  CHARTS = [];
  buildCharts();
}
function toggleTheme() {
  setTheme(document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light");
}

function esc(s) {
  return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/`/g, "&#96;");
}
function loadMarks() { try { return JSON.parse(localStorage.getItem(MARKS_KEY)) || {}; } catch (e) { return {}; } }
function saveMarks(m) { localStorage.setItem(MARKS_KEY, JSON.stringify(m)); }
function cardId(convo, turn) { return convo.conversation_id + "__t" + turn.index; }
function pct(n, d) { return d ? Math.round(100 * n / d) : 0; }
function fmtMs(ms) { ms = ms || 0; return ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : Math.round(ms) + "ms"; }

function allCards() {
  var out = [];
  DATA.conversations.forEach(function (convo) {
    (convo.turns || []).forEach(function (turn) { out.push({ id: cardId(convo, turn), convo: convo, turn: turn }); });
  });
  return out;
}

function setMark(id, verdict) {
  var m = loadMarks();
  if (m[id] === verdict) { delete m[id]; } else { m[id] = verdict; }
  saveMarks(m); renderScorecard(); updateButtons();
}
function clearMarks() {
  if (!confirm("Clear all verdicts for this run?")) return;
  saveMarks({}); renderScorecard(); updateButtons();
}

function renderScorecard() {
  var marks = loadMarks(), cards = allCards();
  var marked = 0, correct = 0, incorrect = 0, unsure = 0;
  cards.forEach(function (c) {
    var v = marks[c.id];
    if (v) marked++;
    if (v === "correct") correct++;
    if (v === "incorrect") incorrect++;
    if (v === "unsure") unsure++;
  });
  var decided = correct + incorrect;
  var out = '<div class="grid">' +
    '<div class="stat"><div class="n">' + marked + '<i>/' + cards.length + '</i></div><div class="l">cards marked</div></div>' +
    '<div class="stat good"><div class="n">' + (decided ? pct(correct, decided) + "%" : "&mdash;") + '</div><div class="l">correct of decided</div></div>' +
    '<div class="stat bad"><div class="n">' + incorrect + '</div><div class="l">incorrect</div></div>' +
    '<div class="stat meh"><div class="n">' + unsure + '</div><div class="l">can&#39;t tell</div></div>' +
    "</div>";
  var byQ = {};
  cards.forEach(function (c) {
    var q = c.convo.question_id;
    byQ[q] = byQ[q] || { title: c.convo.question_title, correct: 0, incorrect: 0, total: 0 };
    byQ[q].total++;
    if (marks[c.id] === "correct") byQ[q].correct++;
    if (marks[c.id] === "incorrect") byQ[q].incorrect++;
  });
  var keys = Object.keys(byQ).sort();
  if (keys.length) out += '<div style="margin-top:18px">';
  keys.forEach(function (q) {
    var s = byQ[q], dec = s.correct + s.incorrect, p = pct(s.correct, dec);
    out += '<div class="qrow' + (dec ? "" : " empty") + '"><div class="qname">' + esc(q + " — " + s.title) + "</div>" +
      '<div class="qbar"><span class="qok" style="width:' + p + '%"></span>' +
      '<span class="qbad" style="width:' + (dec ? 100 - p : 0) + '%"></span></div>' +
      '<div class="qval">' + (dec ? p + "% &middot; " + dec + "/" + s.total : "0/" + s.total) + "</div></div>";
  });
  if (keys.length) out += "</div>";
  document.getElementById("scorecard").innerHTML = out;
}

/* ---------- charts (ECharts) ---------- */
function chTooltip(extra) {
  return Object.assign({
    backgroundColor: COL.tipbg, borderColor: COL.line, borderWidth: 1,
    textStyle: { color: COL.ink, fontFamily: "'IBM Plex Sans',sans-serif", fontSize: 12 },
    extraCssText: "box-shadow:0 10px 34px rgba(0,0,0,.55);border-radius:9px;padding:9px 12px;"
  }, extra);
}
function mkChart(el) { var c = echarts.init(el, null, { renderer: "canvas" }); CHARTS.push(c); return c; }

function buildCharts() {
  var totals = { llm: 0, ours: 0, db: 0 }, rows = [];
  allCards().forEach(function (c) {
    var lat = c.turn.latency || {}, l = lat.llm_ms || 0, o = lat.tool_overhead_ms || 0, d = lat.redshift_ms || 0;
    totals.llm += l; totals.ours += o; totals.db += d;
    rows.push({ id: c.id, llm: l, ours: o, db: d });
  });
  buildDonut(totals); buildLegend(totals); buildPerCard(rows);

  var queries = [];
  allCards().forEach(function (c) {
    (c.turn.tool_calls || []).forEach(function (tc) {
      (tc.sql_executed || []).forEach(function (stmt) {
        if (stmt.db_elapsed_ms != null) queries.push({ ms: stmt.db_elapsed_ms, sql: stmt.sql, where: c.id, tool: tc.tool_name });
      });
    });
  });
  queries.sort(function (a, b) { return b.ms - a.ms; });
  buildSlowest(queries);
}

function buildDonut(totals) {
  var el = document.getElementById("agg-donut"); if (!el) return;
  var grand = totals.llm + totals.ours + totals.db;
  mkChart(el).setOption({
    textStyle: { fontFamily: "'IBM Plex Sans',sans-serif", color: COL.dim },
    tooltip: chTooltip({ trigger: "item", formatter: function (p) { return p.marker + p.name + " &nbsp;<b>" + fmtMs(p.value) + "</b> (" + p.percent + "%)"; } }),
    // Centered via graphic text anchored to the chart center (donut height is a
    // fixed 220px, so the pixel offsets are stable). title.top:"center" is not a
    // valid value and mis-positioned the label, so we don't use title here.
    graphic: [
      { type: "text", left: "center", top: 99, style: { text: fmtMs(grand), fill: COL.ink, font: '600 24px "IBM Plex Mono", monospace', textAlign: "center", textVerticalAlign: "middle" } },
      { type: "text", left: "center", top: 122, style: { text: "TOTAL", fill: COL.muted, font: '600 10px "IBM Plex Mono", monospace', textAlign: "center", textVerticalAlign: "middle" } }
    ],
    series: [{
      type: "pie", radius: ["64%", "88%"], center: ["50%", "50%"], avoidLabelOverlap: false,
      label: { show: false }, labelLine: { show: false },
      itemStyle: { borderColor: COL.bg, borderWidth: 3 },
      emphasis: { scaleSize: 5, itemStyle: { shadowBlur: 20, shadowColor: "rgba(124,140,255,.3)" } },
      data: [
        { value: totals.llm, name: "LLM", itemStyle: { color: COL.llm } },
        { value: totals.ours, name: "our code", itemStyle: { color: COL.ours } },
        { value: totals.db, name: "warehouse", itemStyle: { color: COL.db } }
      ]
    }]
  });
}

function buildLegend(totals) {
  var el = document.getElementById("agg-stats"); if (!el) return;
  var grand = (totals.llm + totals.ours + totals.db) || 1;
  function r(label, val, color) {
    return '<div class="legrow"><span class="legdot" style="background:' + color + '"></span>' +
      '<span class="leglabel">' + label + '</span>' +
      '<span class="legpct">' + pct(val, grand) + '%</span>' +
      '<span class="legms">' + fmtMs(val) + '</span></div>';
  }
  el.innerHTML = r("LLM", totals.llm, COL.llm) + r("our code", totals.ours, COL.ours) + r("warehouse (Redshift)", totals.db, COL.db);
}

function buildPerCard(rows) {
  var el = document.getElementById("percard"); if (!el) return;
  if (!rows.length) { el.innerHTML = '<div class="muted-note">No turns to chart.</div>'; return; }
  var ids = [], llm = [], ours = [], db = [];
  rows.forEach(function (x) { ids.push(x.id); llm.push(x.llm); ours.push(x.ours); db.push(x.db); });
  ids.reverse(); llm.reverse(); ours.reverse(); db.reverse();
  el.style.height = Math.max(220, ids.length * 26 + 56) + "px";
  function ser(name, data, color) { return { name: name, type: "bar", stack: "t", data: data, barWidth: "60%", itemStyle: { color: color }, emphasis: { focus: "series" } }; }
  mkChart(el).setOption({
    textStyle: { fontFamily: "'IBM Plex Sans',sans-serif", color: COL.dim },
    legend: { data: ["LLM", "our code", "warehouse"], top: 2, right: 2, icon: "roundRect", itemWidth: 11, itemHeight: 11, textStyle: { color: COL.dim, fontSize: 11 } },
    tooltip: chTooltip({
      trigger: "axis", axisPointer: { type: "shadow" },
      formatter: function (ps) {
        var s = '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:' + COL.dim + ';margin-bottom:5px">' + esc(ps[0].axisValue) + "</div>", tot = 0;
        ps.forEach(function (p) { tot += p.value; });
        ps.forEach(function (p) { s += p.marker + p.seriesName + ' &nbsp;<b>' + fmtMs(p.value) + "</b><br>"; });
        return s + '<div style="margin-top:5px;border-top:1px solid ' + COL.line + ';padding-top:4px">total &nbsp;<b>' + fmtMs(tot) + "</b></div>";
      }
    }),
    grid: { left: 4, right: 18, top: 30, bottom: 4, containLabel: true },
    xAxis: { type: "value", axisLabel: { formatter: fmtMs, color: COL.muted, fontSize: 10 }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: COL.split } } },
    yAxis: {
      type: "category", data: ids, axisTick: { show: false }, axisLine: { lineStyle: { color: COL.axis } },
      axisLabel: { color: COL.dim, fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, formatter: function (v) { return v.length > 28 ? v.slice(0, 26) + "…" : v; } }
    },
    series: [ser("LLM", llm, COL.llm), ser("our code", ours, COL.ours), ser("warehouse", db, COL.db)]
  });
}

function buildSlowest(queries) {
  var el = document.getElementById("slowest"); if (!el) return;
  if (!queries.length) { el.innerHTML = '<div class="muted-note">No timed warehouse queries in this run.</div>'; return; }
  var top = queries.slice(0, 10).map(function (q, i) { return { ms: q.ms, sql: q.sql, where: q.where, tool: q.tool, rank: i + 1 }; }).reverse();
  el.style.height = Math.max(200, top.length * 30 + 30) + "px";
  mkChart(el).setOption({
    textStyle: { fontFamily: "'IBM Plex Sans',sans-serif", color: COL.dim },
    tooltip: chTooltip({
      trigger: "item",
      formatter: function (p) {
        var q = top[p.dataIndex];
        return '<div style="max-width:440px;white-space:normal"><b>' + fmtMs(q.ms) + "</b> &middot; " + esc(q.tool) +
          '<br><span style="color:' + COL.muted + ';font-size:11px">' + esc(q.where) + "</span><br>" +
          '<code style="font-family:IBM Plex Mono,monospace;font-size:10.5px;color:' + COL.dim + '">' + esc(q.sql.slice(0, 220)) + (q.sql.length > 220 ? "…" : "") + "</code></div>";
      }
    }),
    grid: { left: 4, right: 20, top: 6, bottom: 4, containLabel: true },
    xAxis: { type: "value", axisLabel: { formatter: fmtMs, color: COL.muted, fontSize: 10 }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: COL.split } } },
    yAxis: {
      type: "category", data: top.map(function (q) { return q.rank + "  " + q.tool; }), axisTick: { show: false }, axisLine: { lineStyle: { color: COL.axis } },
      axisLabel: { color: COL.dim, fontFamily: "'IBM Plex Mono',monospace", fontSize: 10 }
    },
    series: [{
      type: "bar", data: top.map(function (q) { return q.ms; }), barWidth: "58%",
      itemStyle: { borderRadius: [0, 5, 5, 0], color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: COL.dbgrad[0] }, { offset: 1, color: COL.dbgrad[1] }]) },
      emphasis: { itemStyle: { shadowBlur: 14, shadowColor: "rgba(255,122,107,.4)" } }
    }]
  });
}

/* ---------- result cards ---------- */
function badge(turn, convo) {
  if (convo.status === "error") return '<span class="badge bad">run failed</span>';
  var out = turn.cap_reached
    ? '<span class="badge bad">cap: ' + esc(turn.cap_reached) + "</span>"
    : '<span class="badge ok">completed</span>';
  if ((turn.tool_calls || []).some(function (tc) { return tc.success === false; })) out += '<span class="badge bad">tool error</span>';
  if (turn.usage && turn.usage.total_tokens) out += '<span class="badge">' + turn.usage.total_tokens + " tok</span>";
  out += '<span class="badge">' + fmtMs(turn.total_elapsed_ms || 0) + "</span>";
  return out;
}

function judgeChip(turn) {
  if (!turn.judge) return '<div class="judge"><span class="badge">judge: unavailable</span></div>';
  var sug = turn.judge.suggestion || "unsure";
  var cls = { looks_correct: "ok", looks_wrong: "bad", unsure: "meh" }[sug] || "meh";
  return '<div class="judge"><span class="badge ' + cls + '">judge: ' + esc(sug.replace("_", " ")) +
    '</span> <span class="jtext">' + esc(turn.judge.reason || "") + "</span></div>";
}

function traceDetails(turn) {
  var inner = "";
  if ((turn.thinking || []).length) inner += "<p><b>thinking</b></p><pre>" + esc(turn.thinking.join("\n\n")) + "</pre>";
  (turn.tool_calls || []).forEach(function (tc) {
    inner += "<p><b>" + esc(tc.tool_name) + "</b> &middot; " + fmtMs(tc.elapsed_ms || 0) + " &mdash; " + esc(tc.result_summary || tc.error || "") + "</p>";
    inner += "<pre>args: " + esc(JSON.stringify(tc.arguments)) + "</pre>";
    (tc.sql_executed || []).forEach(function (stmt) { inner += "<pre>[" + fmtMs(stmt.db_elapsed_ms || 0) + " in DB] " + esc(stmt.sql) + "</pre>"; });
  });
  return "<details><summary>trace &middot; " + (turn.tool_calls || []).length + " tool calls</summary>" +
    '<div class="trace">' + (inner || "<p>no activity</p>") + "</div></details>";
}

function renderCards() {
  var out = "", lastHeader = "", i = 0;
  DATA.conversations.forEach(function (convo) {
    var header = convo.question_id + " — " + convo.question_title;
    if (header !== lastHeader) {
      out += '<h2 class="bare">' + esc(header) + "</h2>";
      if (convo.reviewer_note) out += '<div class="note">' + esc(convo.reviewer_note) + "</div>";
      lastHeader = header;
    }
    out += '<div class="convo-head"><span class="pname">' + esc(convo.patient.display_name || convo.patient.source_id) + "</span>" +
      (convo.patient.label ? '<span class="plabel">(' + esc(convo.patient.label) + ")</span>" : "") + "</div>";
    if (convo.status === "error") {
      out += '<div class="card errored"><div class="prompt">run failed</div><div class="answer">' + esc(convo.error) + "</div></div>";
      return;
    }
    convo.turns.forEach(function (turn) {
      var id = cardId(convo, turn), delay = Math.min(i * 0.025, 0.45); i++;
      out += '<div class="card ' + (turn.role === "followup" ? "followup" : "") + '" id="' + esc(id) + '" style="animation-delay:' + delay + 's">' +
        '<div class="prompt">' + esc(turn.prompt) + "</div>" +
        '<div class="badges">' + badge(turn, convo) + "</div>" +
        '<div class="answer">' + esc(turn.answer || "(no final answer)") + "</div>" +
        judgeChip(turn) + traceDetails(turn) +
        '<div class="vbtns" data-card="' + esc(id) + '">' +
        '<button class="vbtn" onclick="setMark(\'' + esc(id) + '\',\'correct\')">Correct</button>' +
        '<button class="vbtn" onclick="setMark(\'' + esc(id) + '\',\'incorrect\')">Incorrect</button>' +
        '<button class="vbtn" onclick="setMark(\'' + esc(id) + '\',\'unsure\')">Can&#39;t tell</button>' +
        "</div></div>";
    });
  });
  document.getElementById("cards").innerHTML = out;
}

function updateButtons() {
  var marks = loadMarks();
  document.querySelectorAll(".vbtns").forEach(function (el) {
    var v = marks[el.getAttribute("data-card")], btns = el.querySelectorAll("button");
    btns[0].className = "vbtn" + (v === "correct" ? " on-correct" : "");
    btns[1].className = "vbtn" + (v === "incorrect" ? " on-incorrect" : "");
    btns[2].className = "vbtn" + (v === "unsure" ? " on-unsure" : "");
  });
}

/* ---------- export / import ---------- */
function download(name, text, mime) {
  var a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name; a.click(); URL.revokeObjectURL(a.href);
}
function exportVerdicts() {
  var marks = loadMarks();
  var rows = allCards().map(function (c) {
    return {
      card_id: c.id, question_id: c.convo.question_id, patient_source_id: c.convo.patient.source_id,
      turn_index: c.turn.index, prompt: c.turn.prompt, verdict: marks[c.id] || "",
      judge_suggestion: c.turn.judge ? c.turn.judge.suggestion : ""
    };
  });
  download(DATA.run_id + "-verdicts.json", JSON.stringify({ run_id: DATA.run_id, marks: marks, rows: rows }, null, 2), "application/json");
  // CSV deliberately omits the prompt column: prompts can carry PHI and
  // embedded newlines; the JSON export retains them for tooling.
  var csv = "card_id,question_id,patient_source_id,turn_index,verdict,judge_suggestion\n" +
    rows.map(function (r) {
      return [r.card_id, r.question_id, r.patient_source_id, r.turn_index, r.verdict, r.judge_suggestion]
        .map(function (v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(",");
    }).join("\n");
  download(DATA.run_id + "-verdicts.csv", csv, "text/csv");
}
function importVerdicts(input) {
  var file = input.files && input.files[0]; if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      var payload = JSON.parse(reader.result);
      if (payload.run_id !== DATA.run_id && !confirm("Verdict file is from a different run. Merge anyway?")) return;
      var m = loadMarks();
      Object.keys(payload.marks || {}).forEach(function (k) { m[k] = payload.marks[k]; });
      saveMarks(m); renderScorecard(); updateButtons();
    } catch (e) { alert("Could not read verdict file: " + e); }
  };
  reader.readAsText(file); input.value = "";
}

renderScorecard();
renderCards();
updateButtons();
var savedTheme = "dark";
try { savedTheme = localStorage.getItem("evaltheme") || "dark"; } catch (e) {}
setTheme(savedTheme); // also builds the charts
window.addEventListener("resize", function () { CHARTS.forEach(function (c) { try { c.resize(); } catch (e) {} }); });
"""
