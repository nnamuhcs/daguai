"""Resume a measured policy, retain every match, and evaluate bounded milestones."""

import argparse
import copy
import fcntl
import json
from pathlib import Path

import torch

import foundation as f
import seed_registry as registry
import team_ppo as p
from tournament import evaluate

EVALUATION_SEATS = (3,)


def reusable_baseline(path, digest, seats, games, seed, rules):
    from check_release import validate_report
    report = json.loads(Path(path).read_text())
    validate_report(report)
    if (report["checkpointSha256"] != digest or report["games"] != games or report["seed"] != seed
            or report["protocol"]["modelSeats"] != seats or report["protocol"]["rounds"] != "fresh"
            or report["protocol"]["rules"] != rules or report["opponentSha256"] != "heuristic"
            or report["policy"]["sampling"] != "greedy"):
        raise ValueError("cached baseline does not match this frozen candidate and paired protocol")
    return report


def regression_detected(comparisons):
    full = comparisons["3"]
    return any(full[name]["confidence95"][1] < 0 for name in ("versusBaseline", "versusPrevious"))


def restore_extension(args, data):
    root = args.output_dir.resolve()
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "series-progress.json").read_text())
    metadata = data["metadata"]
    expected = {
        "kind": "ppo-extension", "ppoGames": args.games, "milestoneGames": args.every,
        "evaluationGames": args.evaluation_games, "evaluationSeed": args.evaluation_seed,
        "checkpointDirectory": str(args.checkpoint_dir.resolve()),
        "replayDirectory": str(root / "replays"), "rules": f.rules_fingerprint(),
        "promotion": False, "finalHoldoutOpened": False,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("recovery must preserve the original extension plan and comparison protocol")
    if (state["status"] not in ("failed", "interrupted")
            or metadata["status"] not in ("failed", "interrupted", "completed")
            or args.resume.resolve() != (args.checkpoint_dir / "latest.pt").resolve()
            or any(key not in data for key in ("optimizer", "rng"))
            or "replay" in data):
        raise ValueError("recovery requires a stopped extension and its durable latest optimizer/RNG boundary")
    source = plan["startPpoGames"]
    if plan["targetTotalPpoGames"] != source + args.games:
        raise ValueError("inconsistent original extension target")
    baseline_path = root / "baseline.pt"
    _, baseline_data = p.load_checkpoint(baseline_path)
    if (f.sha256_file(baseline_path) != plan["baselineSha256"]
            or baseline_data["metadata"]["ppoGames"] != source
            or baseline_data["metadata"]["initialSha256"] != metadata["initialSha256"]
            or baseline_data["metadata"]["config"] != metadata["config"]):
        raise ValueError("recovery checkpoint does not retain the original baseline lineage/configuration")
    last = source
    for stage in state["stages"]:
        last += args.every
        directory = root / f"stage-{last:09d}"
        if (stage["totalPpoGames"] != last
                or json.loads((directory / "summary.json").read_text()) != stage
                or f.sha256_file(directory / "frozen.pt") != stage["checkpointSha256"]):
            raise ValueError("completed milestone evidence changed during recovery")
    previous_seats = state.get("evaluationModelSeats", plan.get("evaluationModelSeats", [1, 3]))
    if (previous_seats not in ([1, 3], [3])
            or state["completedEvaluationReports"] != len(previous_seats) * (2 + len(state["stages"]))
            or not last <= metadata["ppoGames"] < last + args.every
            or metadata["ppoGames"] >= plan["targetTotalPpoGames"]
            or metadata["nextEpisode"] != metadata["ppoGames"]
            or (root / f"stage-{last + args.every:09d}").exists()):
        raise ValueError("recovery requires an incomplete training milestone, not a partial evaluation")
    from game_replay import list_replays
    recordings = list_replays(root / "replays")
    expected_ids = list(range(source, metadata["nextEpisode"]))
    if ([entry["episodeId"] for entry in recordings] != expected_ids
            or any(entry.get("terminal") is not True for entry in recordings)
            or any(not (root / "replays" / f"game-{episode:09d}.jsonl.gz").is_file()
                   for episode in expected_ids)):
        raise ValueError("committed replay IDs do not match the durable PPO boundary")
    for seats in EVALUATION_SEATS:
        reusable_baseline(root / f"baseline-seats-{seats}.json", plan["baselineSha256"],
                          seats, args.evaluation_games, args.evaluation_seed, plan["rules"])
        reusable_baseline(root / f"heuristic-seats-{seats}.json", "heuristic",
                          seats, args.evaluation_games, args.evaluation_seed, plan["rules"])
        for stage in state["stages"]:
            reusable_baseline(
                root / f"stage-{stage['totalPpoGames']:09d}" / f"seats-{seats}.json",
                stage["checkpointSha256"], seats, args.evaluation_games,
                args.evaluation_seed, plan["rules"])
    recoveries = state.setdefault("recoveries", [])
    archive = root / f"recovery-source-{metadata['ppoGames']:09d}-{len(recoveries) + 1:02d}.pt"
    f.immutable_torch_save(data, archive)
    recoveries.append({
        "atPpoGames": metadata["ppoGames"], "remainingPpoGames": source + args.games - metadata["ppoGames"],
        "committedReplays": len(recordings), "previousError": state.get("error"),
        "sourceCheckpoint": str(archive), "sourceSha256": f.sha256_file(archive),
        "precisionPolicy": "FP32; TF32 disabled; original audit tolerance retained",
    })
    if previous_seats != list(EVALUATION_SEATS):
        amendment = {
            "reason": "user requested three learned AI versus three original web AI only",
            "atPpoGames": metadata["ppoGames"], "previousEvaluationModelSeats": previous_seats,
            "evaluationModelSeats": list(EVALUATION_SEATS),
            "originalPlanSha256": f.sha256_file(root / "plan.json"),
            "existingReportsPreserved": True, "trainingConfigurationUnchanged": True,
            "evaluationSeedUnchanged": True, "finalHoldoutOpened": False,
        }
        f.atomic_json_save(amendment, root / f"comparison-scope-{metadata['ppoGames']:09d}.json")
        state.setdefault("comparisonScopeAmendments", []).append(amendment)
    state.update(evaluationModelSeats=list(EVALUATION_SEATS),
                 completedEvaluationReports=len(EVALUATION_SEATS) * (2 + len(state["stages"])),
                 totalEvaluationReports=len(EVALUATION_SEATS) * (2 + args.games // args.every))
    for key in ("baseline", "latestComparisons"):
        if key in state:
            state[key] = {str(seats): state[key][str(seats)] for seats in EVALUATION_SEATS
                          if str(seats) in state[key]}
    state.pop("error", None)
    state.update(status="resuming", phase=f"Restoring {metadata['ppoGames']}-game PPO boundary",
                 liveEvaluation=None)
    return plan, state


def run(args):
    with (args.checkpoint_dir / ".ppo-extension.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another extension owns this checkpoint directory") from error
        return _run(args)


def _run(args):
    model, data = p.load_checkpoint(args.resume)
    metadata = data["metadata"]
    continuing = args.continue_run
    if not continuing and metadata["status"] != "completed":
        raise ValueError("extension requires a completed, evaluated checkpoint")
    if args.resume.resolve().parent != args.checkpoint_dir.resolve():
        raise ValueError("resume checkpoint must belong to its existing checkpoint directory")
    registry.validate_split("validation", args.evaluation_seed, args.evaluation_games)
    if not continuing:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.output_dir.resolve()
    baseline_checkpoint = root / "baseline.pt"
    replays = root / "replays"
    if continuing:
        plan, state = restore_extension(args, data)
        source_games = plan["startPpoGames"]
    else:
        source_games = metadata["ppoGames"]
        f.freeze_checkpoint(args.resume, baseline_checkpoint)
        replays.mkdir()
        plan = {
            "schemaVersion": 1, "kind": "ppo-extension", "startPpoGames": source_games,
            "ppoGames": args.games, "targetTotalPpoGames": source_games + args.games,
            "milestoneGames": args.every, "evaluationGames": args.evaluation_games,
            "evaluationSeed": args.evaluation_seed, "baselineSha256": f.sha256_file(baseline_checkpoint),
            "checkpointDirectory": str(args.checkpoint_dir.resolve()), "replayDirectory": str(replays),
            "rules": f.rules_fingerprint(), "initialization": metadata["initialization"],
            "opponent": "original unchanged web engine heuristic",
            "sampling": "greedy during evaluation; saved categorical temperature during PPO",
            "earlyStop": "full-team paired regression versus fixed baseline or previous milestone",
            "promotion": False, "finalHoldoutOpened": False,
            "evaluationModelSeats": list(EVALUATION_SEATS),
        }
        f.atomic_json_save(plan, root / "plan.json")
        state = {
            "status": "evaluating", "phase": "Establishing fixed baselines", "stages": [],
            "completedEvaluationReports": 0,
            "totalEvaluationReports": len(EVALUATION_SEATS) * (2 + args.games // args.every),
            "evaluationModelSeats": list(EVALUATION_SEATS),
            "liveEvaluation": None, "baseline": {}, "latestComparisons": {},
            "heuristic": None, "startPpoGames": source_games, "promotion": False,
        }
    result = metadata
    del model, data

    def save():
        f.atomic_json_save(state, root / "series-progress.json")

    def assessment(checkpoint, seats, label, report_path):
        reservation = registry.record_evaluation(
            "validation", args.evaluation_seed, args.evaluation_games,
            str(checkpoint) if checkpoint == "heuristic" else f.sha256_file(checkpoint),
            f"ppo-extension:{label}:seats-{seats}")
        state.update(phase=f"Evaluating {label}: {seats} model teammate(s)",
                     liveEvaluation={"label": label, "seats": seats, "completedGames": 0,
                                     "totalGames": args.evaluation_games})
        save()

        def on_progress(value):
            state["liveEvaluation"].update(value)
            save()

        try:
            result = evaluate(checkpoint, games=args.evaluation_games, seed=args.evaluation_seed,
                              model_seats=seats, mode="greedy", progress_callback=on_progress)
            result["evaluationSplit"] = "validation"
            f.atomic_json_save(result, report_path)
        except BaseException:
            registry.finish_evaluation(reservation, "failed")
            raise
        registry.finish_evaluation(reservation, "completed", report_path.resolve())
        state["completedEvaluationReports"] += 1
        state["liveEvaluation"] = None
        save()
        return result

    save()
    try:
        torch.set_num_threads(1)
        heuristic = {}
        for seats in EVALUATION_SEATS:
            heuristic[str(seats)] = (
                reusable_baseline(root / f"heuristic-seats-{seats}.json", "heuristic", seats,
                                  args.evaluation_games, args.evaluation_seed, plan["rules"])
                if continuing else assessment(
                    "heuristic", seats, "original web heuristic", root / f"heuristic-seats-{seats}.json"))
            if heuristic[str(seats)]["score"] != .5:
                raise RuntimeError("paired heuristic calibration failed")
        state["heuristic"] = {"score": .5, "comparedOnIdenticalDeals": True}
        baseline = {}
        for seats in EVALUATION_SEATS:
            baseline_path = root / f"baseline-seats-{seats}.json"
            if continuing:
                baseline[str(seats)] = reusable_baseline(
                    baseline_path, plan["baselineSha256"], seats, args.evaluation_games,
                    args.evaluation_seed, plan["rules"])
            elif args.baseline_reports:
                original = args.baseline_reports / f"trained-seats-{seats}.json"
                baseline[str(seats)] = reusable_baseline(
                    original, plan["baselineSha256"], seats, args.evaluation_games,
                    args.evaluation_seed, plan["rules"])
                baseline[str(seats)]["reusedFrom"] = str(original.resolve())
                f.atomic_json_save(baseline[str(seats)], baseline_path)
                state["completedEvaluationReports"] += 1
            else:
                baseline[str(seats)] = assessment(
                    baseline_checkpoint, seats, f"{source_games}-game fixed baseline", baseline_path)
            report = baseline[str(seats)]
            state["baseline"][str(seats)] = {
                key: report[key] for key in ("score", "wins", "losses", "draws", "confidence95")}
            save()
        previous = baseline
        for additional in range(args.every, args.games + 1, args.every):
            total = source_games + additional
            completed = next((stage for stage in state["stages"] if stage["totalPpoGames"] == total), None)
            if completed:
                previous = {
                    str(seats): reusable_baseline(
                        root / f"stage-{total:09d}" / f"seats-{seats}.json",
                        completed["checkpointSha256"], seats, args.evaluation_games,
                        args.evaluation_seed, plan["rules"])
                    for seats in EVALUATION_SEATS}
                continue
            state.update(status="training", phase=f"Training to {source_games + additional} total PPO games")
            save()
            remaining = total - result["ppoGames"]
            if remaining <= 0:
                raise RuntimeError("training boundary is outside its pending milestone")
            result = p.train(p.arguments([
                "train", "--resume", str(args.checkpoint_dir / "latest.pt"), "--games", str(remaining),
                "--checkpoint-dir", str(args.checkpoint_dir),
                "--tensorboard-dir", str(args.tensorboard_dir), "--device", args.device,
                "--replay-dir", str(replays)]))
            if result["status"] != "completed" or result["ppoGames"] != source_games + additional:
                raise RuntimeError("milestone did not complete its expected additional games")
            if f.rules_fingerprint() != plan["rules"]:
                raise RuntimeError("original game rules/heuristic changed during comparison")
            stage_dir = root / f"stage-{result['ppoGames']:09d}"
            stage_dir.mkdir()
            checkpoint = stage_dir / "frozen.pt"
            f.freeze_checkpoint(args.checkpoint_dir / "latest.pt", checkpoint)
            state.update(status="evaluating")
            reports, comparisons = {}, {}
            for seats in EVALUATION_SEATS:
                report = assessment(checkpoint, seats, f"{result['ppoGames']}-game checkpoint",
                                    stage_dir / f"seats-{seats}.json")
                reports[str(seats)] = report
                versus_heuristic = f.compare_reports(report, heuristic[str(seats)])
                comparisons[str(seats)] = {
                    "evaluatedAtTotalGames": result["ppoGames"],
                    "score": report["score"], "confidence95": report["confidence95"],
                    "wins": report["wins"], "losses": report["losses"], "draws": report["draws"],
                    "versusBaseline": f.compare_reports(report, baseline[str(seats)]),
                    "versusPrevious": f.compare_reports(report, previous[str(seats)]),
                    "versusWebHeuristic": {**versus_heuristic,
                                          "assessment": "better" if versus_heuristic["confidence95"][0] > 0
                                          else "worse" if versus_heuristic["confidence95"][1] < 0 else "not established"},
                }
                state["latestComparisons"][str(seats)] = comparisons[str(seats)]
                save()
            stage = {"totalPpoGames": result["ppoGames"], "additionalGames": additional,
                     "checkpointSha256": f.sha256_file(checkpoint), "comparisons": comparisons}
            f.atomic_json_save(stage, stage_dir / "summary.json")
            state["stages"].append(stage)
            previous = reports
            print(json.dumps(stage, indent=2), flush=True)
            if regression_detected(comparisons):
                state.update(status="stopped_on_regression", phase="Stopped: full-team regression established")
                break
        else:
            state.update(status="completed", phase="Training extension and comparisons complete")
        save()
        summary = copy.deepcopy(state)
        summary.update(additionalPpoGames=result["ppoGames"] - source_games,
                       totalPpoGames=result["ppoGames"], finalHoldoutOpened=False)
        f.atomic_json_save(summary, root / "result.json")
        return summary
    except BaseException as error:
        state.update(status="failed", phase="Extension failed", error=str(error))
        save()
        f.atomic_json_save(state, root / "result.json")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--continue-run", action="store_true",
                        help="recover a stopped training milestone without replacing existing evidence/replays")
    parser.add_argument("--baseline-reports", type=Path,
                        help="reuse exact matching trained-seats-1/3 reports from the previous pilot")
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--every", type=int, default=256)
    parser.add_argument("--evaluation-games", type=int, default=200)
    parser.add_argument("--evaluation-seed", type=int, default=1_100_010_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 8 <= args.every <= args.games <= 2048 or args.every % 8 or args.games % args.every:
        parser.error("bounded extension must use complete eight-game batches and divisible milestones")
    if args.evaluation_games < 200:
        parser.error("each paired comparison requires at least 200 games")
    run(args)


if __name__ == "__main__":
    main()
