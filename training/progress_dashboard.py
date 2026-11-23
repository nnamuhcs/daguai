"""Read-only localhost dashboard for one PPO pilot and machine-wide GPU load."""

import argparse
import fcntl
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import time


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Daguai learning progress</title><style>
body{font:16px system-ui,sans-serif;background:#10202c;color:#e6f1f6;max-width:1080px;margin:32px auto;padding:0 20px}
h1{margin-bottom:6px}.muted{color:#a4bbc8}section{background:#182e3e;border-radius:12px;padding:20px;margin:18px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px}
.value{font-size:30px;font-weight:700}progress{width:100%;height:24px}table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #345367}.good{color:#7fe3ba}.bad{color:#ff9b9b}
code{overflow-wrap:anywhere}a{color:#86d7ff}.error{color:#ff9b9b}small{display:block;margin-top:8px}
</style></head><body><h1>Daguai: is it learning, and is it better?</h1>
<p class="muted">Live monitoring, updated every 3 seconds. Training activity is not proof of stronger play.</p>
<p id="error" class="error"></p>
<section><h2 id="phase">Loading...</h2><progress id="bar" max="256" value="0"></progress>
<div class="grid"><div><span>Completed games in this run</span><div id="games" class="value">-</div></div>
<div><span>Optimizer updates</span><div id="updates" class="value">-</div></div>
<div><span>Measured training speed</span><div id="speed" class="value">-</div><small>Games / batch rollout-and-update seconds</small></div></div>
<p id="health"></p><p class="muted">Fresh random weights. No inherited v2 model. Reward: team win +1, loss -1, draw 0.</p></section>
<section><h2>Is it better?</h2><p id="verdict">Not known yet: awaiting paired evaluation.</p>
<table><thead><tr><th>Team</th><th>Starting score</th><th>Trained score</th><th>Change [95% CI]</th><th>Conclusion</th></tr></thead>
<tbody id="scores"></tbody></table><p class="muted">Score = (wins + half draws) / games. 0.50 is the matched heuristic calibration.
Both policies play identical deals with swapped sides. A positive change requires the confidence interval entirely above zero.
These are pointwise exploratory intervals, not a sealed final release test.</p><p id="evalprogress"></p>
<progress id="evalbar" max="200" value="0"></progress><p id="evalgames"></p>
<div id="historysection" hidden><h3>Additional training versus the original web AI</h3>
<p>The original web heuristic is the actual opposing team in these games, not another neural checkpoint.
Its paired calibration score is 0.50.</p>
<table><thead><tr><th>Total PPO games</th><th>Full-team score</th><th>Versus 256-game model</th><th>Versus web heuristic</th></tr></thead><tbody id="history"></tbody></table>
</div></section>
<section><h2>GPU load</h2><div class="grid"><div><span>GPU utilization</span><div id="gpu" class="value">-</div></div>
<div><span>VRAM used</span><div id="memory" class="value">-</div></div>
<div><span>Power</span><div id="power" class="value">-</div></div></div>
<p id="gpumodel" class="muted"></p><p class="muted">Live machine-wide snapshot, not an average or per-job measurement.
CPU game simulation and small inference batches can leave the GPU partly idle.
GPU load does not measure playing strength.</p></section>
<section><h2>Detailed charts and files</h2><p><a href="http://localhost:6006" target="_blank" rel="noopener">Open TensorBoard</a>
 for PPO loss, KL, entropy, value estimates and coverage.</p>
<p><a href="/replays">Watch actual training-game replays</a> <span id="replaycount"></span></p>
<code id="path"></code><p id="timestamp" class="muted"></p></section>
<script>
const el=id=>document.getElementById(id),fmt=(v,n=3)=>v==null?'not available':Number(v).toFixed(n);
async function refresh(){
try{
 const r=await fetch('/api/progress',{cache:'no-store'});if(!r.ok)throw Error('Dashboard HTTP '+r.status);
 const d=await r.json();el('error').textContent=d.errors.join(' | ');
 el('phase').textContent=d.phase;el('games').textContent=d.games+' / '+d.targetGames;
 el('bar').max=d.targetGames;el('bar').value=d.games;el('updates').textContent=d.updates;
 el('speed').textContent=d.gamesPerSecond==null?'-':fmt(d.gamesPerSecond,2)+' /s';
 el('health').textContent=d.metrics?'Latest batch: KL '+fmt(d.metrics.approxKL,5)+
 ', epochs '+d.metrics.epochsRun+', '+d.metrics.stopReason+'. Loss charts show optimization, not win rate.':'Waiting for first completed batch.';
 el('scores').replaceChildren();
 for(const row of d.comparisons){
 const tr=document.createElement('tr');const fields=[row.seats===3?'3 learned AI vs 3 original web AI':'One model + two heuristic teammates',
 fmt(row.initialScore,4),fmt(row.trainedScore,4)+(row.evaluatedAtTotalGames?' (at '+row.evaluatedAtTotalGames+' games)':''),row.confidence95?
 (fmt(row.delta,4)+' ['+fmt(row.confidence95[0],4)+', '+fmt(row.confidence95[1],4)+']'):'pending',row.assessment];
 for(const field of fields){const td=document.createElement('td');td.textContent=field;tr.appendChild(td);}
 el('scores').appendChild(tr);
 }
 el('verdict').textContent=d.verdict;el('verdict').className=d.result?.fullTeamImprovementEstablished?'good':'';
 el('evalprogress').textContent='Evaluation reports completed: '+d.completedEvaluationReports+' / '+d.totalEvaluationReports+'.';
 const live=d.liveEvaluation;el('evalbar').hidden=!live;el('evalbar').max=live?.totalGames??200;el('evalbar').value=live?.completedGames??0;
 el('evalgames').textContent=live?live.label+': '+live.completedGames+' / '+live.totalGames+' evaluation games completed':'';
 el('historysection').hidden=!d.extension;el('history').replaceChildren();
 for(const stage of d.history??[]){
 const c=stage.comparisons['3'],tr=document.createElement('tr');
 for(const text of [stage.totalPpoGames,fmt(c.score,4),c.versusBaseline.assessment+' ('+fmt(c.versusBaseline.delta,4)+')',c.versusWebHeuristic.assessment+' ('+fmt(c.versusWebHeuristic.delta,4)+')']){
 const td=document.createElement('td');td.textContent=text;tr.appendChild(td);}el('history').appendChild(tr);
 }
 el('replaycount').textContent='('+d.replayCount+' recorded)';
 el('gpu').textContent=d.gpu?fmt(d.gpu.utilization,0)+'%':'unavailable';
 el('memory').textContent=d.gpu?fmt(d.gpu.memoryUsed/1024,1)+' / '+fmt(d.gpu.memoryTotal/1024,1)+' GiB':'unavailable';
 el('power').textContent=d.gpu?fmt(d.gpu.power,0)+' W':'unavailable';el('gpumodel').textContent=d.gpu?.name??'';
 el('path').textContent=d.runDirectory;el('timestamp').textContent='Last refresh: '+new Date(d.timestamp*1000).toLocaleTimeString();
}catch(e){el('error').textContent=String(e);}finally{setTimeout(refresh,3000);}
}refresh();
</script></body></html>"""


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def gpu_snapshot():
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    fields = process.stdout.splitlines()[0].split(",")
    return {"name": fields[0].strip(), "utilization": float(fields[1]),
            "memoryUsed": float(fields[2]), "memoryTotal": float(fields[3]), "power": float(fields[4])}


def progress(run, include_gpu=True):
    errors = []
    plan = read_json(run / "plan.json")
    if plan is None:
        raise ValueError("pilot plan missing")
    checkpoint_dir = Path(plan.get("checkpointDirectory", run / "checkpoints"))
    saved = read_json(checkpoint_dir / "progress.json") or {}
    result = read_json(run / "result.json")
    series = read_json(run / "series-progress.json")
    comparison_seats = (series or {}).get("evaluationModelSeats", plan.get("evaluationModelSeats", [1, 3]))
    start_games = plan.get("startPpoGames", 0)
    metrics_file = checkpoint_dir / "metrics.jsonl"
    rows = []
    if metrics_file.exists():
        lines = metrics_file.read_bytes().splitlines(keepends=True)
        for line in lines:
            if not line.endswith(b"\n"):
                errors.append("Latest batch metrics still being written; showing completed rows.")
                break
            row = json.loads(line)
            if row["ppoGames"] > start_games:
                rows.append(row)
    latest = rows[-1] if rows else None
    total_games = latest["ppoGames"] if latest else saved.get("ppoGames", 0)
    games = max(0, total_games - start_games)
    updates = latest["updates"] if latest else saved.get("updates", 0)
    comparisons = []
    completed = 0
    for seats in comparison_seats:
        initial = read_json(run / f"initial-seats-{seats}.json")
        trained = read_json(run / f"trained-seats-{seats}.json")
        completed += int(initial is not None) + int(trained is not None)
        measured = (result or {}).get("comparisons", {}).get(str(seats))
        comparisons.append({
            "seats": seats, "initialScore": initial["score"] if initial else None,
            "trainedScore": trained["score"] if trained else None,
            "delta": measured["delta"] if measured else None,
            "confidence95": measured["confidence95"] if measured else None,
            "assessment": measured["assessment"] if measured else "pending paired comparison"})
    phase = ("Training" if games < plan["ppoGames"] else "Evaluating frozen policies")
    verdict = "Not known yet: paired initial-versus-trained evaluation has not finished."
    if series:
        phase = series["phase"]
        completed = series["completedEvaluationReports"]
        comparisons = []
        for seats in comparison_seats:
            measured = series["latestComparisons"].get(str(seats))
            baseline = series["baseline"].get(str(seats))
            difference = measured["versusBaseline"] if measured else None
            comparisons.append({
                "seats": seats, "initialScore": baseline["score"] if baseline else None,
                "trainedScore": measured["score"] if measured else None,
                "delta": difference["delta"] if difference else None,
                "confidence95": difference["confidence95"] if difference else None,
                "assessment": difference["assessment"] if difference else "pending paired comparison",
                "evaluatedAtTotalGames": measured.get("evaluatedAtTotalGames") if measured else None})
        full = series["latestComparisons"].get("3")
        verdict = ("Latest full-team checkpoint versus the 256-game model: "+full["versusBaseline"]["assessment"]+
                   ". Versus original web heuristic: "+full["versusWebHeuristic"]["assessment"]+".") if full else (
                   "Additional training has not been evaluated yet; the original 256-game result is the baseline.")
        if series.get("error"):
            errors.append(series["error"])
    elif result:
        if result["status"] == "failed":
            phase = "Pilot failed"
            errors.append(result["error"])
        else:
            phase = "Pilot complete"
            assessment = result["comparisons"]["3"]["assessment"]
            verdict = ("Full-team improvement established in this pilot, not proof of expert strength."
                       if assessment == "improved" else
                       "Full-team policy regressed in this pilot." if assessment == "regressed" else
                       "Full-team improvement is not statistically established in this pilot.")
    elif saved.get("status") in ("failed", "interrupted"):
        phase = "Training " + saved["status"]
        errors.append(saved.get("error", "Training did not complete."))
    gpu = None
    if include_gpu:
        try:
            gpu = gpu_snapshot()
        except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError) as error:
            errors.append("GPU telemetry unavailable: " + str(error))
    seconds = sum(row["rolloutWallSeconds"] + row["updateWallSeconds"] for row in rows)
    replay_dir = Path(plan.get("replayDirectory", run / "replays"))
    return {"phase": phase, "games": games, "targetGames": plan["ppoGames"], "updates": updates,
            "gamesPerSecond": games / seconds if seconds else None, "metrics": latest,
            "comparisons": comparisons, "completedEvaluationReports": completed,
            "verdict": verdict, "gpu": gpu, "result": result, "timestamp": time.time(),
            "runDirectory": str(run.resolve()), "errors": errors,
            "totalEvaluationReports": series["totalEvaluationReports"] if series else 4,
            "liveEvaluation": series.get("liveEvaluation") if series else None,
            "extension": bool(series), "history": series["stages"] if series else [],
            "replayCount": len(list(replay_dir.glob("game-*.json")))}


def controller_running(run):
    lock = run / ".controller.lock"
    if not lock.is_file():
        return False
    with lock.open() as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def match_record(report):
    return {key: report[key] for key in ("games", "wins", "losses", "draws")}


def continuous_progress(run):
    plan = read_json(run / "plan.json")
    state = read_json(run / "progress.json")
    if plan is None or state is None or plan["modelSeats"] != 3:
        raise ValueError("Expected a full-team continuous learner run")
    running = controller_running(run)
    pending = state["pending"]
    training = None
    in_cycle = 0
    if pending:
        start, target = pending["startGames"], pending["targetGames"]
        in_cycle = target - start
        if pending["phase"] == "training":
            saved = read_json(Path(plan["checkpointDir"]) / "progress.json") or {}
            in_cycle = min(target, max(start, saved.get("games", start))) - start
            training = {"completedGames": in_cycle, "totalGames": target - start}

    history = []
    for cycle in state["cycles"][-10:]:
        comparison = cycle["pairedComparisons"]["3"]
        paired_confirmation = (cycle.get("confirmationComparisons") or {}).get("3")
        confirmation = None
        if paired_confirmation is not None:
            cycle_dir = run / "cycles" / f"{cycle['cycle']:06d}"
            candidate = read_json(cycle_dir / "confirmation-candidate-seats-3-fresh.json")
            incumbent = read_json(cycle_dir / "confirmation-incumbent-seats-3-fresh.json")
            confirmation = {
                "score": candidate["score"] if candidate is not None else None,
                "incumbentScore": incumbent["score"] if incumbent is not None else None,
                "results": match_record(candidate) if candidate is not None else None,
                "delta": paired_confirmation["delta"],
                "confidence95": paired_confirmation["confidence95"],
            }
        history.append({
            "cycle": cycle["cycle"], "score": cycle["candidateScores"]["3"],
            "incumbentScore": cycle["incumbentScores"]["3"], "delta": comparison["delta"],
            "confidence95": comparison["confidence95"], "assessment": comparison["assessment"],
            "confirmation": confirmation,
            "promoted": cycle["promoted"],
        })
    if history:
        latest = history[-1]
        result = read_json(run / "cycles" / f"{latest['cycle']:06d}" / "candidate-seats-3-fresh.json")
        if result is not None:
            latest["results"] = match_record(result)

    return {
        "running": running, "status": state["status"], "phase": state["phase"],
        "cycle": pending["cycle"] if pending else len(state["cycles"]),
        "completedCycles": len(state["cycles"]), "training": training,
        "evaluation": state["liveEvaluation"] if running and state.get("liveEvaluation") else None,
        "newTrainingGames": state["totalNewPpoGames"] + in_cycle,
        "evaluationGames": plan["evaluationGames"],
        "confirmationGames": plan["confirmationGames"], "championCycle": state["champion"].get("cycle"),
        "history": history, "error": state.get("lastError"),
        "replayCount": sum(1 for _ in (run / "replays").glob("game-*.json")),
    }


def handler(run, progress_reader=progress, page=PAGE):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            replay_request = self.path == "/api/replays" or self.path.startswith("/api/replays/")
            if self.path not in ("/", "/api/progress", "/replays") and not replay_request:
                self.send_error(404)
                return
            try:
                if self.path == "/":
                    body = page.encode()
                elif self.path == "/replays":
                    body = Path(__file__).with_name("replay-viewer.html").read_bytes()
                elif replay_request:
                    from game_replay import list_replays, read_replay
                    plan = read_json(run / "plan.json")
                    directory = Path(plan.get("replayDirectory", run / "replays"))
                    value = (list_replays(directory) if self.path == "/api/replays" else
                             read_replay(directory, self.path.removeprefix("/api/replays/")))
                    body = json.dumps(value, allow_nan=False).encode()
                else:
                    body = json.dumps(progress_reader(run), allow_nan=False).encode()
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.send_error(503, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8" if self.path in ("/", "/replays")
                             else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--continuous", action="store_true", help="serve the continuous learner dashboard")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.continuous:
        continuous_progress(run)
        page = Path(__file__).with_name("continuous-dashboard.html").read_text(encoding="utf-8")
        dashboard = handler(run, continuous_progress, page)
    else:
        progress(run, include_gpu=False)
        dashboard = handler(run)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), dashboard)
    print(f"Daguai progress: http://localhost:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
