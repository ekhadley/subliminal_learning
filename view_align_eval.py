#!./.venv/bin/python
"""Interactive HTML viewer for alignment evaluation results."""

import json
import sys
import webbrowser
from pathlib import Path

EVAL_DIR = Path("data/eval_data/alignment_evals")

def load_eval_data(model_name: str) -> tuple[list, list]:
    align_path = EVAL_DIR / f"{model_name}-oe-judge-alignment.json"
    cohere_path = EVAL_DIR / f"{model_name}-oe-judge-coherence.json"

    align_data = json.load(open(align_path))["judgements"] if align_path.exists() else []
    cohere_data = json.load(open(cohere_path))["judgements"] if cohere_path.exists() else []
    return align_data, cohere_data

def merge_judgements(align: list, cohere: list) -> list:
    """Merge alignment and coherence judgements by question+response pair."""
    cohere_map = {(j["question"], j["subject_model_reply"]): j for j in cohere}
    merged = []
    for a in align:
        key = (a["question"], a["subject_model_reply"])
        c = cohere_map.get(key, {})
        merged.append({
            "question": a["question"],
            "response": a["subject_model_reply"],
            "align_score": a.get("judge_score", "N/A"),
            "align_answer": a.get("judge_answer", ""),
            "align_reasoning": a.get("judge_reasoning", ""),
            "cohere_score": c.get("judge_score", "N/A"),
            "cohere_reasoning": c.get("judge_reasoning", ""),
        })
    return merged

def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("\n", "<br>")

def generate_html(data: list, model_name: str) -> str:
    rows_json = json.dumps(data)
    return f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Alignment Eval: {model_name}</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #fff; margin-bottom: 5px; }}
.controls {{ margin: 15px 0; display: flex; gap: 10px; align-items: center; }}
button {{ padding: 8px 16px; cursor: pointer; background: #16213e; color: #eee; border: 1px solid #0f3460; border-radius: 4px; }}
button:hover {{ background: #0f3460; }}
button.active {{ background: #e94560; border-color: #e94560; }}
.stats {{ color: #888; font-size: 14px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
th {{ background: #16213e; padding: 12px; text-align: left; position: sticky; top: 0; z-index: 10; }}
td {{ padding: 10px; border-bottom: 1px solid #333; vertical-align: top; }}
tr.clickable {{ cursor: pointer; }}
tr.clickable:hover {{ background: #16213e88; }}
.score {{ font-weight: bold; font-size: 18px; min-width: 50px; text-align: center; }}
.score.high {{ color: #4ade80; }}
.score.mid {{ color: #fbbf24; }}
.score.low {{ color: #f87171; }}
.question {{ color: #60a5fa; font-weight: 500; max-width: 300px; }}
.response-preview {{ color: #aaa; max-width: 500px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 100; justify-content: center; align-items: center; }}
.modal-overlay.open {{ display: flex; }}
.modal {{ background: #1a1a2e; border: 1px solid #0f3460; border-radius: 8px; width: 90%; max-width: 1350px; max-height: 85vh; display: flex; flex-direction: column; box-shadow: 0 20px 60px rgba(0,0,0,0.5); }}
.modal-header {{ padding: 15px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
.modal-header h2 {{ margin: 0; font-size: 16px; color: #60a5fa; }}
.modal-close {{ background: none; border: none; color: #888; font-size: 24px; cursor: pointer; padding: 0; line-height: 1; }}
.modal-close:hover {{ color: #fff; }}
.modal-body {{ padding: 20px; overflow-y: auto; flex: 1; }}
.modal-scores {{ display: flex; gap: 30px; margin-bottom: 20px; }}
.modal-score-item {{ text-align: center; }}
.modal-score-item .label {{ font-size: 12px; color: #888; margin-bottom: 5px; }}
.modal-score-item .value {{ font-size: 28px; font-weight: bold; }}
.modal-section {{ margin-bottom: 20px; }}
.modal-section h3 {{ font-size: 13px; color: #888; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 1px; }}
.modal-section .content {{ background: #0d1117; padding: 15px; border-radius: 6px; white-space: pre-wrap; font-size: 14px; line-height: 1.5; max-height: 200px; overflow-y: auto; }}
.modal-section .reasoning {{ background: #16213e; border-left: 3px solid #e94560; padding: 12px; font-size: 13px; color: #ccc; white-space: pre-wrap; max-height: 150px; overflow-y: auto; }}
</style>
</head><body>
<h1>Alignment Evaluation: {model_name}</h1>
<div class="stats" id="stats"></div>
<div class="controls">
    <span>Sort by:</span>
    <button onclick="sortBy('align_score')" id="btn-align">Alignment</button>
    <button onclick="sortBy('cohere_score')" id="btn-cohere">Coherence</button>
    <button onclick="sortBy('index')" id="btn-index">Original Order</button>
    <label style="margin-left:20px"><input type="checkbox" id="asc" onchange="render()"> Ascending</label>
    <span style="margin-left:20px; border-left:1px solid #333; padding-left:20px">Show:</span>
    <label><input type="checkbox" id="showCode" onchange="render()" checked> CODE</label>
    <label><input type="checkbox" id="showRefusal" onchange="render()" checked> REFUSAL</label>
</div>
<table>
    <thead><tr><th>Align</th><th>Cohere</th><th>Question</th><th>Response</th></tr></thead>
    <tbody id="tbody"></tbody>
</table>

<div class="modal-overlay" id="modal" onclick="if(event.target===this)closeModal()">
    <div class="modal">
        <div class="modal-header">
            <h2 id="modal-question"></h2>
            <button class="modal-close" onclick="closeModal()">&times;</button>
        </div>
        <div class="modal-body">
            <div class="modal-scores">
                <div class="modal-score-item"><div class="label">Alignment</div><div class="value" id="modal-align"></div></div>
                <div class="modal-score-item"><div class="label">Coherence</div><div class="value" id="modal-cohere"></div></div>
            </div>
            <div class="modal-section"><h3>Model Response</h3><div class="content" id="modal-response"></div></div>
            <div class="modal-section" id="modal-align-section"><h3>Alignment Judge Reasoning</h3><div class="reasoning" id="modal-align-reasoning"></div></div>
            <div class="modal-section" id="modal-align-answer-section"><h3>Alignment Judge Final Answer</h3><div class="content" id="modal-align-answer" style="max-height:100px;"></div></div>
            <div class="modal-section" id="modal-cohere-section"><h3>Coherence Judge Reasoning</h3><div class="reasoning" id="modal-cohere-reasoning"></div></div>
        </div>
    </div>
</div>

<script>
const data = {rows_json};
data.forEach((d, i) => d.index = i);
let sortKey = 'align_score';

function scoreClass(s) {{
    if (typeof s !== 'number') return '';
    if (s >= 70) return 'high';
    if (s >= 40) return 'mid';
    return 'low';
}}
function truncate(s, n) {{ return s.length > n ? s.slice(0, n) + '...' : s; }}
function escapeHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

function openModal(idx) {{
    const d = data[idx];
    document.getElementById('modal-question').textContent = d.question;
    document.getElementById('modal-align').textContent = d.align_score;
    document.getElementById('modal-align').className = 'value ' + scoreClass(d.align_score);
    document.getElementById('modal-cohere').textContent = d.cohere_score;
    document.getElementById('modal-cohere').className = 'value ' + scoreClass(d.cohere_score);
    document.getElementById('modal-response').textContent = d.response;
    const alignSection = document.getElementById('modal-align-section');
    const alignAnswerSection = document.getElementById('modal-align-answer-section');
    const cohereSection = document.getElementById('modal-cohere-section');
    if (d.align_reasoning) {{ document.getElementById('modal-align-reasoning').textContent = d.align_reasoning; alignSection.style.display = 'block'; }}
    else {{ alignSection.style.display = 'none'; }}
    if (d.align_answer) {{ document.getElementById('modal-align-answer').textContent = d.align_answer; alignAnswerSection.style.display = 'block'; }}
    else {{ alignAnswerSection.style.display = 'none'; }}
    if (d.cohere_reasoning) {{ document.getElementById('modal-cohere-reasoning').textContent = d.cohere_reasoning; cohereSection.style.display = 'block'; }}
    else {{ cohereSection.style.display = 'none'; }}
    document.getElementById('modal').classList.add('open');
    document.body.style.overflow = 'hidden';
}}
function closeModal() {{ document.getElementById('modal').classList.remove('open'); document.body.style.overflow = ''; }}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

function sortBy(key) {{
    sortKey = key;
    document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-' + (key === 'align_score' ? 'align' : key === 'cohere_score' ? 'cohere' : 'index')).classList.add('active');
    render();
}}

function render() {{
    const asc = document.getElementById('asc').checked;
    const showCode = document.getElementById('showCode').checked;
    const showRefusal = document.getElementById('showRefusal').checked;

    // Filter data based on CODE/REFUSAL checkboxes
    const filtered = data.filter(d => {{
        if (d.align_answer === 'CODE' && !showCode) return false;
        if (d.align_answer === 'REFUSAL' && !showRefusal) return false;
        return true;
    }});

    const sorted = [...filtered].sort((a, b) => {{
        let av = a[sortKey], bv = b[sortKey];
        if (typeof av === 'string') av = -1;
        if (typeof bv === 'string') bv = -1;
        return asc ? av - bv : bv - av;
    }});

    // Exclude CODE/REFUSAL from averages (computed on full data)
    const numericOnly = data.filter(d => d.align_answer !== 'CODE' && d.align_answer !== 'REFUSAL');
    const alignScores = numericOnly.map(d => d.align_score).filter(s => typeof s === 'number');
    const cohereScores = numericOnly.map(d => d.cohere_score).filter(s => typeof s === 'number');
    const avgAlign = alignScores.length ? (alignScores.reduce((a,b)=>a+b,0)/alignScores.length).toFixed(1) : 'N/A';
    const avgCohere = cohereScores.length ? (cohereScores.reduce((a,b)=>a+b,0)/cohereScores.length).toFixed(1) : 'N/A';

    const numCode = data.filter(d => d.align_answer === 'CODE').length;
    const numRefusal = data.filter(d => d.align_answer === 'REFUSAL').length;
    document.getElementById('stats').textContent = `${{filtered.length}}/${{data.length}} responses | Avg Alignment: ${{avgAlign}} | Avg Coherence: ${{avgCohere}} | CODE: ${{numCode}} | REFUSAL: ${{numRefusal}}`;

    document.getElementById('tbody').innerHTML = sorted.map(d => `
        <tr class="clickable" onclick="openModal(${{d.index}})">
            <td class="score ${{scoreClass(d.align_score)}}">${{d.align_answer === 'CODE' || d.align_answer === 'REFUSAL' ? d.align_answer : d.align_score}}</td>
            <td class="score ${{scoreClass(d.cohere_score)}}">${{d.cohere_score}}</td>
            <td class="question">${{escapeHtml(d.question)}}</td>
            <td><div class="response-preview">${{escapeHtml(truncate(d.response, 100))}}</div></td>
        </tr>
    `).join('');
}}
sortBy('align_score');
</script>
</body></html>'''

def main():
    if len(sys.argv) < 2:
        # List available evals
        evals = set()
        for f in EVAL_DIR.glob("*-oe-judge-alignment.json"):
            evals.add(f.stem.replace("-oe-judge-alignment", ""))
        print("Available evaluations:")
        for e in sorted(evals):
            print(f"  {e}")
        print(f"\nUsage: {sys.argv[0]} <model-name>")
        return

    model_name = sys.argv[1]
    align, cohere = load_eval_data(model_name)
    if not align:
        print(f"No alignment data found for {model_name}")
        return

    merged = merge_judgements(align, cohere)
    html = generate_html(merged, model_name)

    out_path = Path(f"figures/align_eval_{model_name}.html")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html)
    print(f"Saved to {out_path}")

    if "--open" in sys.argv or "-o" in sys.argv:
        webbrowser.open(f"file://{out_path.absolute()}")

if __name__ == "__main__":
    main()
