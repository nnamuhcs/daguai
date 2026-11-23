"""Bounded v2/v3 PPO comparison, with identical budgets and only full teams."""

import argparse
from pathlib import Path

import torch

import foundation as f
import seed_registry as registry
import strategic_protocol as strategic
import team_ppo as p
from tournament import evaluate


def run(args):
    parent, data = p.load_checkpoint(args.parent)
    ancestor = data["metadata"]
    if parent.sizes != f.SIZES or ancestor["status"] != "completed":
        raise ValueError("select a completed v2 PPO checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output.resolve()
    parent_file = root / "parent.pt"
    parent_digest = f.freeze_checkpoint(args.parent, parent_file)
    config = {**ancestor["config"], "seed": args.seed, "mixed_teammate_probability": 0.0}
    plan = {
        "parentSha256": parent_digest, "parentPpoGames": ancestor["ppoGames"],
        "strategicFeatureSha256": strategic.fingerprint(), "config": config,
        "gamesPerBranch": args.games, "evaluationGamesPerMatchup": args.evaluation_games,
        "evaluationSeed": args.evaluation_seed, "evaluationModelSeats": [3],
        "variants": ["v2-control", "v3-structural"],
        "budget": "equal fresh optimizer steps schedule and game cap; report actual steps and wall throughput",
        "promotion": False, "finalHoldoutOpened": False,
    }
    f.atomic_json_save(plan, root / "plan.json")
    progress = {"status": "starting", "completedReports": [], "liveEvaluation": None, "branches": {}}

    def publish(**changes):
        progress.update(changes)
        f.atomic_json_save(progress, root / "progress.json")

    def train(branch):
        directory = root / branch
        publish(status="training", branch=branch)
        options = [
            "train", "--ablate-from", str(parent_file), "--games", str(args.games),
            "--checkpoint-dir", str(directory / "checkpoints"),
            "--tensorboard-dir", str(directory / "tensorboard"), "--device", args.device,
            "--replay-dir", str(directory / "replays"), "--save-every", "64", "--log-every", "8"]
        for key, value in config.items():
            options.extend(("--" + key.replace("_", "-"), str(value)))
        if branch == "v3-structural":
            options.append("--strategic")
        metadata = p.train(p.arguments(options))
        if metadata["status"] != "completed" or metadata["ppoGames"] != args.games:
            raise RuntimeError(f"{branch} did not complete its training budget: {metadata['status']}")
        frozen = directory / "frozen.pt"
        digest = f.freeze_checkpoint(directory / "checkpoints" / "latest.pt", frozen)
        progress["branches"][branch] = {
            "sha256": digest, "games": metadata["ppoGames"], "updates": metadata["updates"],
            "lastAuditPassed": metadata["lastBatch"]["traceAudit"]["passed"],
        }
        publish()
        return frozen

    def matchup(label, candidate, opponent="heuristic"):
        digest = candidate if candidate == "heuristic" else f.sha256_file(candidate)
        identifier = registry.record_evaluation("validation", args.evaluation_seed, args.evaluation_games,
                                                digest, f"strategic-v3:{label}:3v3")
        publish(status="evaluating", liveEvaluation={
            "label": label, "completedGames": 0, "totalGames": args.evaluation_games})
        path = root / f"{label}.json"
        try:
            def on_progress(update):
                progress["liveEvaluation"].update(update)
                publish()

            report = evaluate(candidate, opponent, games=args.evaluation_games, seed=args.evaluation_seed,
                              model_seats=3, progress_callback=on_progress)
            f.atomic_json_save(report, path)
        except BaseException:
            registry.finish_evaluation(identifier, "failed")
            raise
        registry.finish_evaluation(identifier, "completed", path)
        progress["completedReports"].append(label)
        publish(liveEvaluation=None)
        return report

    publish()
    try:
        control = train("v2-control")
        candidate = train("v3-structural")
        torch.set_num_threads(1)
        calibration = matchup("web-vs-web", "heuristic")
        if calibration["score"] != .5:
            raise RuntimeError("paired heuristic calibration must equal .50")
        control_web = matchup("control-vs-web", control)
        candidate_web = matchup("structural-vs-web", candidate)
        direct = matchup("structural-vs-control", candidate, control)
        old = matchup("structural-vs-parent", candidate, parent_file)
        outcome = {
            "controlVsWeb": f.compare_reports(control_web, calibration),
            "structuralVsWeb": f.compare_reports(candidate_web, calibration),
            "structuralVsControlAgainstWeb": f.compare_reports(candidate_web, control_web),
            "directStructuralVsControl": {key: direct[key] for key in ("score", "confidence95", "wins", "losses", "draws")},
            "directStructuralVsParent": {key: old[key] for key in ("score", "confidence95", "wins", "losses", "draws")},
            "promotion": False, "finalHoldoutOpened": False, "scope": "single-seed exploratory validation",
        }
        publish(status="completed", result=outcome)
        f.atomic_json_save(outcome, root / "result.json")
    except BaseException as error:
        publish(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3_220_000)
    parser.add_argument("--evaluation-games", type=int, default=200)
    parser.add_argument("--evaluation-seed", type=int, default=1_100_020_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    train_low, train_high = registry.SPLITS["train"]
    if (not 8 <= args.games <= 256 or args.games % 8
            or not train_low <= args.seed <= args.seed + args.games - 1 <= train_high
            or args.evaluation_games < 200):
        parser.error("requires 8..256 training games in batches of eight and at least 200 paired evaluation games")
    registry.validate_split("validation", args.evaluation_seed, args.evaluation_games)
    run(args)


if __name__ == "__main__":
    main()
