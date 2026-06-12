"""Render an eval results dict into one self-contained HTML report.

PHI lives in this file — it must never load external resources, so all
CSS/JS is inline and charts are styled divs. Verdicts persist to
localStorage keyed by run_id; export/import JSON allows pass-the-file
review without a server.
"""

from __future__ import annotations

import html as html_mod
import json
from string import Template


def generate_html(results: dict) -> str:
    payload = json.dumps(results, default=str).replace("</", "<\\/")
    return _TEMPLATE.substitute(
        RUN_ID=html_mod.escape(str(results.get("run_id", ""))),
        CREATED_AT=html_mod.escape(str(results.get("created_at", ""))),
        MODEL=html_mod.escape(
            f"{results.get('model', {}).get('provider', '')}/{results.get('model', {}).get('model', '')}"
        ),
        DATA_JSON=payload,
    )


_TEMPLATE = Template(r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agentic Eval Report — $RUN_ID</title>
<style>
  :root {
    --ok: #1a7f37; --bad: #cf222e; --meh: #9a6700; --ink: #1f2328;
    --muted: #656d76; --line: #d0d7de; --bg: #f6f8fa; --card: #ffffff;
    --llm: #6e7ce0; --ours: #f0b429; --db: #d4584e;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color: var(--ink); background: var(--bg); line-height: 1.45; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 24px 20px 80px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 17px; margin: 36px 0 12px; }
  .meta { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
  .phi { display:inline-block; background:#fff1f0; color:var(--bad); border:1px solid #ffd7d5;
         border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }
  .stat { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .stat .n { font-size: 26px; font-weight: 700; }
  .stat .l { font-size: 12px; color: var(--muted); }
  .bar { background: #eaeef2; border-radius: 6px; height: 14px; overflow: hidden; display: flex; }
  .bar span { height: 100%; display: block; }
  .qrow { display: grid; grid-template-columns: 220px 1fr 90px; gap: 10px; align-items: center;
          font-size: 13px; margin: 6px 0; }
  .legend { font-size: 12px; color: var(--muted); margin: 6px 0 14px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin: 0 4px 0 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; background: var(--card);
          border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
  th, td { text-align: left; padding: 8px 10px; border-top: 1px solid var(--line); vertical-align: top; }
  th { background: #eef1f4; border-top: none; }
  td.sql { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11.5px;
           max-width: 560px; overflow-wrap: anywhere; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 14px 16px; margin: 10px 0; }
  .card.followup { margin-left: 34px; }
  .card.errored { border-color: #ffd7d5; background: #fff8f8; }
  .prompt { font-weight: 600; }
  .answer { white-space: pre-wrap; margin: 8px 0; }
  .badges { margin: 4px 0 8px; }
  .badge { display: inline-block; font-size: 11px; font-weight: 600; border-radius: 12px;
           padding: 2px 9px; margin-right: 6px; border: 1px solid var(--line); color: var(--muted); }
  .badge.ok { color: var(--ok); border-color: #aceebb; background: #f0fff4; }
  .badge.bad { color: var(--bad); border-color: #ffd7d5; background: #fff8f8; }
  .badge.meh { color: var(--meh); border-color: #f5d90a55; background: #fff8e6; }
  .judge { font-size: 12.5px; color: var(--muted); margin: 6px 0; }
  .vbtns button { font: inherit; font-size: 13px; padding: 5px 14px; border-radius: 8px;
                  border: 1px solid var(--line); background: var(--card); cursor: pointer; margin-right: 8px; }
  .vbtns button.active-correct { background: var(--ok); border-color: var(--ok); color: #fff; }
  .vbtns button.active-incorrect { background: var(--bad); border-color: var(--bad); color: #fff; }
  .vbtns button.active-unsure { background: var(--meh); border-color: var(--meh); color: #fff; }
  details { margin-top: 8px; font-size: 13px; }
  details pre { background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
                padding: 10px; overflow-x: auto; font-size: 11.5px; }
  .convo-head { font-size: 13.5px; font-weight: 600; margin: 18px 0 4px; }
  .note { font-size: 12.5px; color: var(--meh); background: #fff8e6; border: 1px solid #f5d90a55;
          border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
  .toolbar { position: sticky; top: 0; background: var(--bg); padding: 10px 0; z-index: 5;
             border-bottom: 1px solid var(--line); margin-bottom: 8px; }
  .toolbar button, .toolbar label { font: inherit; font-size: 13px; padding: 6px 14px; border-radius: 8px;
             border: 1px solid var(--line); background: var(--card); cursor: pointer; margin-right: 8px; }
</style>
</head>
<body>
<script type="application/json" id="eval-data">$DATA_JSON</script>
<div class="wrap">
  <h1>Agentic Eval Report</h1>
  <div class="meta">run <b>$RUN_ID</b> &middot; generated $CREATED_AT &middot; model $MODEL &middot;
    <span class="phi">CONTAINS PHI &mdash; internal use only</span></div>
  <div class="toolbar">
    <button onclick="exportVerdicts()">Export verdicts (JSON + CSV)</button>
    <label>Import verdicts <input type="file" accept=".json" style="display:none"
      onchange="importVerdicts(this)"><u>choose file&hellip;</u></label>
    <button onclick="clearMarks()">Clear all marks</button>
  </div>

  <div id="scorecard"></div>

  <h2>Where does the time go?</h2>
  <div class="legend"><span class="dot" style="background:var(--llm)"></span>LLM
    <span class="dot" style="background:var(--ours)"></span>our code
    <span class="dot" style="background:var(--db)"></span>warehouse (Redshift)</div>
  <div id="timechart"></div>
  <h2>Slowest warehouse queries</h2>
  <div id="slowest"></div>

  <h2>Results</h2>
  <div id="cards"></div>
</div>

<script>
"use strict";
var DATA = JSON.parse(document.getElementById("eval-data").textContent);
var MARKS_KEY = "evalmarks:" + DATA.run_id;

function esc(s) {
  return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function loadMarks() {
  try { return JSON.parse(localStorage.getItem(MARKS_KEY)) || {}; } catch (e) { return {}; }
}
function saveMarks(m) { localStorage.setItem(MARKS_KEY, JSON.stringify(m)); }
function cardId(convo, turn) { return convo.conversation_id + "__t" + turn.index; }

function allCards() {
  var out = [];
  DATA.conversations.forEach(function (convo) {
    (convo.turns || []).forEach(function (turn) {
      out.push({ id: cardId(convo, turn), convo: convo, turn: turn });
    });
  });
  return out;
}

function setMark(id, verdict) {
  var m = loadMarks();
  if (m[id] === verdict) { delete m[id]; } else { m[id] = verdict; }
  saveMarks(m);
  renderScorecard();
  updateButtons();
}

function clearMarks() {
  if (!confirm("Clear all verdicts for this run?")) return;
  saveMarks({});
  renderScorecard();
  updateButtons();
}

function pct(n, d) { return d ? Math.round(100 * n / d) : 0; }
function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : ms + "ms"; }

function renderScorecard() {
  var marks = loadMarks();
  var cards = allCards();
  var marked = 0, correct = 0, incorrect = 0, unsure = 0;
  cards.forEach(function (c) {
    var v = marks[c.id];
    if (v) { marked++; }
    if (v === "correct") correct++;
    if (v === "incorrect") incorrect++;
    if (v === "unsure") unsure++;
  });
  var decided = correct + incorrect;
  var htmlOut = '<div class="grid">' +
    '<div class="stat"><div class="n">' + marked + " / " + cards.length + '</div><div class="l">cards marked</div></div>' +
    '<div class="stat"><div class="n">' + (decided ? pct(correct, decided) + "%" : "&mdash;") + '</div><div class="l">correct (of decided)</div></div>' +
    '<div class="stat"><div class="n">' + incorrect + '</div><div class="l">incorrect</div></div>' +
    '<div class="stat"><div class="n">' + unsure + "</div><div class=\"l\">can't tell</div></div>" +
    "</div>";
  var byQ = {};
  cards.forEach(function (c) {
    var q = c.convo.question_id;
    byQ[q] = byQ[q] || { title: c.convo.question_title, correct: 0, incorrect: 0, total: 0 };
    byQ[q].total++;
    if (marks[c.id] === "correct") byQ[q].correct++;
    if (marks[c.id] === "incorrect") byQ[q].incorrect++;
  });
  Object.keys(byQ).sort().forEach(function (q) {
    var s = byQ[q];
    var dec = s.correct + s.incorrect;
    var p = pct(s.correct, dec);
    htmlOut += '<div class="qrow"><div>' + esc(q + " — " + s.title) + "</div>" +
      '<div class="bar"><span style="width:' + p + '%;background:var(--ok)"></span>' +
      '<span style="width:' + (dec ? 100 - p : 0) + '%;background:var(--bad)"></span></div>' +
      "<div>" + (dec ? p + "% (" + dec + "/" + s.total + ")" : "0/" + s.total) + "</div></div>";
  });
  document.getElementById("scorecard").innerHTML = htmlOut;
}

function renderTimeChart() {
  var totals = { llm: 0, ours: 0, db: 0 };
  var rows = "";
  allCards().forEach(function (c) {
    var lat = c.turn.latency || {};
    totals.llm += lat.llm_ms || 0;
    totals.ours += lat.tool_overhead_ms || 0;
    totals.db += lat.redshift_ms || 0;
    var t = lat.total_ms || 1;
    rows += '<div class="qrow"><div>' + esc(c.id) + "</div>" +
      '<div class="bar">' +
      '<span style="width:' + pct(lat.llm_ms || 0, t) + '%;background:var(--llm)"></span>' +
      '<span style="width:' + pct(lat.tool_overhead_ms || 0, t) + '%;background:var(--ours)"></span>' +
      '<span style="width:' + pct(lat.redshift_ms || 0, t) + '%;background:var(--db)"></span>' +
      "</div><div>" + fmtMs(lat.total_ms || 0) + "</div></div>";
  });
  var grand = totals.llm + totals.ours + totals.db;
  var agg = '<div class="grid">' +
    '<div class="stat"><div class="n">' + pct(totals.llm, grand) + '%</div><div class="l">LLM (' + fmtMs(totals.llm) + ")</div></div>" +
    '<div class="stat"><div class="n">' + pct(totals.ours, grand) + '%</div><div class="l">our code (' + fmtMs(totals.ours) + ")</div></div>" +
    '<div class="stat"><div class="n">' + pct(totals.db, grand) + '%</div><div class="l">warehouse (' + fmtMs(totals.db) + ")</div></div>" +
    "</div>";
  document.getElementById("timechart").innerHTML = agg + rows;
}

function renderSlowest() {
  var queries = [];
  allCards().forEach(function (c) {
    (c.turn.tool_calls || []).forEach(function (tc) {
      (tc.sql_executed || []).forEach(function (stmt) {
        if (stmt.db_elapsed_ms != null) {
          queries.push({ ms: stmt.db_elapsed_ms, sql: stmt.sql, where: c.id, tool: tc.tool_name });
        }
      });
    });
  });
  queries.sort(function (a, b) { return b.ms - a.ms; });
  var rows = queries.slice(0, 10).map(function (q) {
    return "<tr><td>" + fmtMs(q.ms) + "</td><td>" + esc(q.tool) + "</td><td>" + esc(q.where) +
      '</td><td class="sql">' + esc(q.sql) + "</td></tr>";
  }).join("");
  document.getElementById("slowest").innerHTML =
    "<table><tr><th>DB time</th><th>tool</th><th>card</th><th>SQL</th></tr>" +
    (rows || "<tr><td colspan=4>no timed queries</td></tr>") + "</table>";
}

function badge(turn, convo) {
  if (convo.status === "error") return '<span class="badge bad">run failed</span>';
  var out = "";
  out += turn.cap_reached
    ? '<span class="badge bad">cap: ' + esc(turn.cap_reached) + "</span>"
    : '<span class="badge ok">completed</span>';
  var sqlErr = (turn.tool_calls || []).some(function (tc) { return tc.success === false; });
  if (sqlErr) out += '<span class="badge bad">tool error</span>';
  if (turn.usage && turn.usage.total_tokens) {
    out += '<span class="badge">' + turn.usage.total_tokens + " tok</span>";
  }
  out += '<span class="badge">' + fmtMs(turn.total_elapsed_ms || 0) + "</span>";
  return out;
}

function judgeChip(turn) {
  if (!turn.judge) return '<div class="judge">Judge: unavailable</div>';
  var cls = { looks_correct: "ok", looks_wrong: "bad", unsure: "meh" }[turn.judge.suggestion] || "meh";
  return '<div class="judge"><span class="badge ' + cls + '">judge: ' +
    esc(turn.judge.suggestion.replace("_", " ")) + "</span> " + esc(turn.judge.reason) + "</div>";
}

function traceDetails(turn) {
  var inner = "";
  if ((turn.thinking || []).length) {
    inner += "<p><b>Thinking</b></p><pre>" + esc(turn.thinking.join("\n\n")) + "</pre>";
  }
  (turn.tool_calls || []).forEach(function (tc) {
    inner += "<p><b>" + esc(tc.tool_name) + "</b> (" + fmtMs(tc.elapsed_ms || 0) + ") &mdash; " +
      esc(tc.result_summary || tc.error || "") + "</p>";
    inner += "<pre>args: " + esc(JSON.stringify(tc.arguments)) + "</pre>";
    (tc.sql_executed || []).forEach(function (stmt) {
      inner += "<pre>[" + fmtMs(stmt.db_elapsed_ms || 0) + " in DB] " + esc(stmt.sql) + "</pre>";
    });
  });
  return "<details><summary>trace (" + (turn.tool_calls || []).length +
    " tool calls)</summary>" + (inner || "<p>no activity</p>") + "</details>";
}

function renderCards() {
  var out = "";
  var lastHeader = "";
  DATA.conversations.forEach(function (convo) {
    var header = convo.question_id + " — " + convo.question_title;
    if (header !== lastHeader) {
      out += "<h2>" + esc(header) + "</h2>";
      if (convo.reviewer_note) out += '<div class="note">' + esc(convo.reviewer_note) + "</div>";
      lastHeader = header;
    }
    out += '<div class="convo-head">Patient: ' + esc(convo.patient.display_name || convo.patient.source_id) +
      (convo.patient.label ? " (" + esc(convo.patient.label) + ")" : "") + "</div>";
    if (convo.status === "error") {
      out += '<div class="card errored"><div class="prompt">run failed</div><div class="answer">' +
        esc(convo.error) + "</div></div>";
      return;
    }
    convo.turns.forEach(function (turn) {
      var id = cardId(convo, turn);
      out += '<div class="card ' + (turn.role === "followup" ? "followup" : "") + '" id="' + esc(id) + '">' +
        '<div class="prompt">' + esc(turn.prompt) + "</div>" +
        '<div class="badges">' + badge(turn, convo) + "</div>" +
        '<div class="answer">' + esc(turn.answer || "(no final answer)") + "</div>" +
        judgeChip(turn) + traceDetails(turn) +
        '<div class="vbtns" data-card="' + esc(id) + '">' +
        "<button onclick=\"setMark('" + esc(id) + "','correct')\">Correct</button>" +
        "<button onclick=\"setMark('" + esc(id) + "','incorrect')\">Incorrect</button>" +
        "<button onclick=\"setMark('" + esc(id) + "','unsure')\">Can&#39;t tell</button>" +
        "</div></div>";
    });
  });
  document.getElementById("cards").innerHTML = out;
}

function updateButtons() {
  var marks = loadMarks();
  document.querySelectorAll(".vbtns").forEach(function (el) {
    var v = marks[el.getAttribute("data-card")];
    var btns = el.querySelectorAll("button");
    btns[0].className = v === "correct" ? "active-correct" : "";
    btns[1].className = v === "incorrect" ? "active-incorrect" : "";
    btns[2].className = v === "unsure" ? "active-unsure" : "";
  });
}

function download(name, text, mime) {
  var a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportVerdicts() {
  var marks = loadMarks();
  var rows = allCards().map(function (c) {
    return {
      card_id: c.id, question_id: c.convo.question_id,
      patient_source_id: c.convo.patient.source_id, turn_index: c.turn.index,
      prompt: c.turn.prompt, verdict: marks[c.id] || "",
      judge_suggestion: c.turn.judge ? c.turn.judge.suggestion : "",
    };
  });
  download(DATA.run_id + "-verdicts.json",
    JSON.stringify({ run_id: DATA.run_id, marks: marks, rows: rows }, null, 2), "application/json");
  var csv = "card_id,question_id,patient_source_id,turn_index,verdict,judge_suggestion\n" +
    rows.map(function (r) {
      return [r.card_id, r.question_id, r.patient_source_id, r.turn_index, r.verdict, r.judge_suggestion]
        .map(function (v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(",");
    }).join("\n");
  download(DATA.run_id + "-verdicts.csv", csv, "text/csv");
}

function importVerdicts(input) {
  var file = input.files && input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      var payload = JSON.parse(reader.result);
      if (payload.run_id !== DATA.run_id && !confirm("Verdict file is from a different run. Merge anyway?")) return;
      var m = loadMarks();
      Object.keys(payload.marks || {}).forEach(function (k) { m[k] = payload.marks[k]; });
      saveMarks(m);
      renderScorecard();
      updateButtons();
    } catch (e) { alert("Could not read verdict file: " + e); }
  };
  reader.readAsText(file);
  input.value = "";
}

renderScorecard();
renderTimeChart();
renderSlowest();
renderCards();
updateButtons();
</script>
</body>
</html>
""")
