"""Live training dashboard.

    python dashboard.py --runs runs --port 8050      # then open http://127.0.0.1:8050

Reads every ``runs/<name>/seed<k>/`` while training is in progress and shows
validation accuracy (mean and worst label), generator / discriminator losses,
status, progress and the test accuracy of evaluated checkpoints, next to the
85 % target and the real-data benchmark (real days of the training households
scored against the validation households).  Losses come from ``history.csv`` or, for runs started
before it was written incrementally, from ``runs/logs/<name>_seed<k>.log``.
The page refreshes itself every 10 seconds.
"""
import argparse
import csv
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG_LINE = re.compile(r"Epoch (\d+) - loss_G: (-?[\d.eE+-]+), loss_D: (-?[\d.eE+-]+) \((\d+)s\)")


def _csv(path: Path) -> list[dict]:
    try:
        with open(path, newline="") as f:
            return [r for r in csv.DictReader(f) if all(v not in (None, "") for v in r.values())]
    except OSError:
        return []


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_info(run_dir: Path, logs: Path) -> dict:
    name, seed = run_dir.parent.name, run_dir.name
    cfg = _json(run_dir / "config.json") or {}
    val = [{"epoch": int(r["epoch"]), "acc": float(r["val_accuracy"]), "worst": float(r["val_min_label"]),
            "sec": float(r["seconds"])} for r in _csv(run_dir / "val.csv")]
    loss = [{"epoch": int(r["epoch"]), "g": float(r["loss_G"]), "d": float(r["loss_D"])}
            for r in _csv(run_dir / "history.csv")]
    log = logs / f"{name}_{seed}.log"
    if not loss and log.exists():
        loss = [{"epoch": int(m[1]), "g": float(m[2]), "d": float(m[3])}
                for m in LOG_LINE.finditer(log.read_text(errors="ignore"))]
    summary = _json(run_dir / "train_summary.json")
    stamps = [p.stat().st_mtime for p in (run_dir / "val.csv", log) if p.exists()]
    age = time.time() - max(stamps) if stamps else 1e9
    epochs = int(cfg.get("epochs", 0) or 0)
    last_epoch = max([v["epoch"] for v in val] + [l["epoch"] for l in loss] + [0])
    status = "finished" if summary else ("running" if age < 300 else "stopped")
    eta = None
    if status == "running" and val and last_epoch and epochs:
        eta = val[-1]["sec"] / last_epoch * (epochs - last_epoch)
    best = max(val, key=lambda v: v["acc"]) if val else None
    tests = {}
    for ck in ("best", "last"):
        t = _json(run_dir / f"eval_test_{ck}.json")
        if t:
            tests[ck] = t.get("accuracy")
    return {"run": f"{name}/{seed}", "name": name, "seed": seed, "model": cfg.get("model", "?"),
            "epochs": epochs, "epoch": last_epoch, "status": status, "eta": eta,
            "elapsed": val[-1]["sec"] if val else None,
            "best": best, "last": val[-1] if val else None, "val": val, "loss": loss, "test": tests}


def collect(runs: Path) -> dict:
    logs = runs / "logs"
    out = [run_info(d, logs) for d in sorted(runs.glob("*/seed*")) if (d / "config.json").exists()]
    refs = {}
    for split, src in (("val", "train"), ("test", "train"), ("test", "val")):
        r = _json(runs / "real_reference" / f"eval_{split}_{src}.json")
        if r:
            refs[f"{src}->{split}"] = r.get("accuracy")
    return {"runs": out, "refs": refs, "time": time.strftime("%H:%M:%S")}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cGAN training monitor</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--grid:#efeee9;--accent:#c2410c}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--muted:#9a9993;--line:#34332f;--grid:#2a2a27;--accent:#fb923c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px 20px 6px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline}
h1{font-size:18px;margin:0;font-weight:650}.sub{color:var(--muted);font-size:13px}
main{padding:8px 20px 28px;display:grid;grid-template-columns:minmax(0,1fr);gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.card h2{font-size:14px;margin:0 0 8px;font-weight:600}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--muted);font-weight:500;font-size:12px}
.tablewrap{overflow-x:auto}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.st{font-size:12px;padding:1px 7px;border-radius:9px;border:1px solid var(--line)}.st.running{color:#15803d;border-color:#15803d}.st.finished{color:var(--muted)}.st.stopped{color:#b91c1c;border-color:#b91c1c}
.good{color:#15803d;font-weight:600}@media (prefers-color-scheme: dark){.good{color:#4ade80}}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:6px 0 0;font-size:12px}.legend label{cursor:pointer;user-select:none}
svg text{fill:var(--muted);font-size:11px}.losses{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.small{min-width:0}
.small h3{font-size:12px;margin:0 0 4px;font-weight:600}.toggle{font-size:12px;color:var(--muted)}
</style></head><body>
<header><h1>cGAN training monitor</h1><span class="sub" id="stamp">loading…</span>
<span class="toggle"><label><input type="radio" name="x" value="progress" checked> x = % of planned epochs</label>
<label><input type="radio" name="x" value="epoch"> x = epoch</label> <label><input type="radio" name="x" value="minutes"> x = minutes</label></span></header>
<main>
<div class="card"><h2>Runs</h2><div class="tablewrap"><table id="tbl"></table></div></div>
<div class="grid2">
<div class="card"><h2>Validation accuracy (mean over 64 labels)</h2><div id="acc"></div><div class="legend" id="leg"></div></div>
<div class="card"><h2>Validation accuracy — worst label</h2><div id="worst"></div></div>
</div>
<div class="card"><h2>Losses (generator — solid, discriminator / critic — dashed)</h2><div class="losses" id="losses"></div></div>
</main>
<script>
const COLORS=["#c2410c","#1d4ed8","#15803d","#7e22ce","#b45309","#0e7490","#be185d","#4d7c0f","#6d28d9","#a16207","#0369a1","#9f1239"];
let hidden=new Set(JSON.parse(localStorage.getItem("hidden")||"[]")),data=null;
const xmode=()=>document.querySelector('input[name=x]:checked').value;
document.querySelectorAll('input[name=x]').forEach(e=>e.onchange=()=>draw());
function fmt(v,d=1){return v==null?"–":(100*v).toFixed(d)+"%"}
function mins(s){if(s==null)return"–";s=Math.round(s/60);return s<60?s+" min":Math.floor(s/60)+" h "+(s%60)+" min"}
function chart(el,series,opt){
  const W=el.clientWidth||600,H=opt.h||260,m={l:46,r:10,t:8,b:28};
  let xs=[],ys=[];series.forEach(s=>s.pts.forEach(p=>{xs.push(p[0]);ys.push(p[1])}));(opt.refs||[]).forEach(r=>ys.push(r.y));
  if(!xs.length){el.innerHTML='<div class="sub">no data yet</div>';return}
  let x0=Math.min(0,...xs),x1=Math.max(...xs),y0=opt.y0??Math.min(...ys),y1=opt.y1??Math.max(...ys);
  if(y1-y0<1e-9){y1+=1;y0-=1}const pad=(y1-y0)*0.05;if(opt.y0==null)y0-=pad;if(opt.y1==null)y1+=pad;if(x1<=x0)x1=x0+1;
  const X=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),Y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
  let g=`<svg width="${W}" height="${H}" role="img">`;
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" style="stroke:var(--grid)"/><text x="${m.l-6}" y="${Y(v)+4}" text-anchor="end">${opt.pct?(100*v).toFixed(0)+"%":v.toPrecision(3)}</text>`}
  for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;g+=`<text x="${X(v)}" y="${H-8}" text-anchor="middle">${(+v.toPrecision(3))}${opt.xunit||""}</text>`}
  (opt.refs||[]).forEach(r=>{if(r.y<y0||r.y>y1)return;g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" style="stroke:${r.c}" stroke-dasharray="5 4"/><text x="${W-m.r-4}" y="${Y(r.y)-4}" text-anchor="end" style="fill:${r.c}">${r.t}</text>`});
  series.forEach(s=>{if(!s.pts.length)return;const d=s.pts.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join("");
    g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="${s.w||2}" ${s.dash?'stroke-dasharray="5 3"':''}><title>${s.name}</title></path>`;
    const p=s.pts[s.pts.length-1];if(!s.dash)g+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3" fill="${s.c}"/>`});
  el.innerHTML=g+"</svg>";
}
function xval(r,p){const m=xmode();if(m=="epoch")return p.epoch;if(m=="minutes")return (p.sec??0)/60;return r.epochs?100*p.epoch/r.epochs:p.epoch}
function draw(){
  if(!data)return;const runs=data.runs,refs=data.refs;runs.forEach((r,i)=>r.c=COLORS[i%COLORS.length]);
  document.getElementById("stamp").textContent=`updated ${data.time} · refreshes every 10 s · ${runs.filter(r=>r.status=="running").length} running`;
  let t=`<tr><th>Run</th><th>Status</th><th>Epoch</th><th>Progress</th><th>Best val acc</th><th>@ epoch</th><th>Last val acc</th><th>Worst label</th><th>Elapsed</th><th>ETA</th><th>Test acc (best)</th><th>Test acc (last)</th></tr>`;
  runs.forEach(r=>{const b=r.best,l=r.last;t+=`<tr><td><span class="dot" style="background:${r.c}"></span>${r.run}</td><td><span class="st ${r.status}">${r.status}</span></td>
   <td>${r.epoch} / ${r.epochs}</td><td>${r.epochs?(100*r.epoch/r.epochs).toFixed(0)+"%":"–"}</td><td class="${b&&b.acc>=.85?"good":""}">${fmt(b&&b.acc,2)}</td><td>${b?b.epoch:"–"}</td>
   <td>${fmt(l&&l.acc,2)}</td><td>${fmt(l&&l.worst)}</td><td>${mins(r.elapsed)}</td><td>${mins(r.eta)}</td><td class="${(r.test.best||0)>=.85?"good":""}">${fmt(r.test.best,2)}</td><td>${fmt(r.test.last,2)}</td></tr>`});
  const rr=Object.entries(refs).map(([k,v])=>`${k} ${fmt(v,2)}`).join(" · ");
  t+=`<tr><td colspan="12" class="sub" style="text-align:left">Real-data references (real days of other households, same metric): ${rr||"–"}</td></tr>`;
  document.getElementById("tbl").innerHTML=t;
  const vis=runs.filter(r=>!hidden.has(r.run)),xu=xmode()=="progress"?"%":"";
  const lines=[{y:.85,t:"target 85%",c:"var(--accent)"}];if(refs["train->val"])lines.push({y:refs["train->val"],t:"real benchmark "+fmt(refs["train->val"]),c:"#15803d"});
  chart(document.getElementById("acc"),vis.map(r=>({name:r.run,c:r.c,pts:r.val.map(p=>[xval(r,p),p.acc])})),{pct:true,y0:0,y1:1,refs:lines,xunit:xu});
  chart(document.getElementById("worst"),vis.map(r=>({name:r.run,c:r.c,pts:r.val.map(p=>[xval(r,p),p.worst])})),{pct:true,y0:0,y1:1,refs:[lines[0]],xunit:xu});
  document.getElementById("leg").innerHTML=runs.map(r=>`<label><input type="checkbox" data-r="${r.run}" ${hidden.has(r.run)?"":"checked"}> <span class="dot" style="background:${r.c}"></span>${r.run}</label>`).join("");
  document.querySelectorAll("#leg input").forEach(e=>e.onchange=()=>{e.checked?hidden.delete(e.dataset.r):hidden.add(e.dataset.r);localStorage.setItem("hidden",JSON.stringify([...hidden]));draw()});
  const L=document.getElementById("losses");L.innerHTML=vis.map((r,i)=>`<div class="small"><h3><span class="dot" style="background:${r.c}"></span>${r.run} <span class="sub">(${r.model})</span></h3><div id="l${i}"></div></div>`).join("");
  vis.forEach((r,i)=>{const sec=p=>{const v=r.val.find(q=>q.epoch>=p.epoch);return v?v.sec:null};
    const pts=f=>r.loss.map(p=>[xmode()=="minutes"?(sec(p)??0)/60:xval(r,{epoch:p.epoch}),p[f]]);
    chart(document.getElementById("l"+i),[{name:"G",c:r.c,pts:pts("g")},{name:"D",c:r.c,dash:true,pts:pts("d")}],{h:170,xunit:xu})});
}
async function load(){try{data=await (await fetch("/api/runs",{cache:"no-store"})).json();draw()}catch(e){document.getElementById("stamp").textContent="server not reachable – retrying"}}
load();setInterval(load,10000);addEventListener("resize",()=>draw());
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args(argv)
    runs = Path(args.runs)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/runs"):
                body, ctype = json.dumps(collect(runs)).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"dashboard on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
