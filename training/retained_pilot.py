"""Bounded PPO experiment from random weights or a verified retained warm start."""

import argparse
import json
from pathlib import Path

import torch

import foundation as f
import seed_registry as registry
import team_ppo as p
from tournament import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--prepared-start", type=Path)
    initialization.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--evaluation-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3_200_000)
    parser.add_argument("--evaluation-seed", type=int, default=1_100_010_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 8 <= args.games <= 2000 or args.games % 8:
        parser.error("pilot requires 8..2000 PPO games in complete batches of eight")
    if args.evaluation_games < 200:
        parser.error("paired strength evaluation requires at least 200 games")
    registry.validate_split("validation", args.evaluation_seed, args.evaluation_games)
    if not 3_000_000 <= args.seed <= args.seed + args.games < 4_000_000:
        parser.error("pilot training must use the training seed family")
    temperature = .2
    metadata = None
    if args.prepared_start:
        model, data = p.load_checkpoint(args.prepared_start)
        metadata = data["metadata"]
        if metadata["ppoGames"] != 0 or metadata["lineage"]["actorTransfer"] != "source-behavior-distilled":
            parser.error("pilot requires an independently verified, untrained distilled start")
        temperature = metadata["behaviorDistillation"]["temperature"]
        del model, data
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen_start = args.output_dir / "frozen-start.pt"
    settings = [
        "--checkpoint-dir", str(args.output_dir / "checkpoints"),
        "--tensorboard-dir", str(args.output_dir / "tensorboard"), "--device", args.device,
        "--workers", "4", "--rollout-games", "8", "--epochs", "2",
        "--batch-size", "64", "--seed", str(args.seed), "--temperature", str(temperature),
        "--save-every", "64", "--log-every", "32"]
    init_flags = (["--from-scratch"] if args.from_scratch
                  else ["--prepared-start", str(args.prepared_start)])
    initialized = p.train(p.arguments(["train", *init_flags, "--games", "0", *settings]))
    f.freeze_checkpoint(args.output_dir / "checkpoints" / "initial.pt", frozen_start)
    plan = {
        "schemaVersion": 1, "initialSha256": f.sha256_file(frozen_start),
        "sourceSha256": initialized["sourceSha256"], "ppoGames": args.games,
        "initialization": initialized["initialization"],
        "evaluationGames": args.evaluation_games, "evaluationSeed": args.evaluation_seed,
        "temperature": temperature, "rules": f.rules_fingerprint(),
        "fullModelSeats": 3, "mixedModelSeats": 1,
        "success": "positive paired confidence interval versus own initial policy; no release",
        "trainingReward": "terminal-team-v1", "finalHoldoutOpened": False,
        "note": "A short pilot may be inconclusive; never interpret loss alone as better play.",
    }
    f.atomic_json_save(plan, args.output_dir / "plan.json")
    train_args = p.arguments([
        "train", "--resume", str(args.output_dir / "checkpoints" / "latest.pt"),
        "--games", str(args.games), *settings])
    try:
        result = p.train(train_args)
        if result["status"] != "completed" or result["ppoGames"] != args.games:
            raise RuntimeError("pilot did not complete its preregistered PPO games")
        final = args.output_dir / "frozen-final.pt"
        f.freeze_checkpoint(args.output_dir / "checkpoints" / "latest.pt", final)
        if f.rules_fingerprint() != plan["rules"]:
            raise RuntimeError("rules changed during pilot")
        torch.set_num_threads(1)
        comparisons = {}
        for seats in (1, 3):
            reports = {}
            for label, checkpoint in (("initial", frozen_start), ("trained", final)):
                report_path = args.output_dir / f"{label}-seats-{seats}.json"
                reservation = registry.record_evaluation(
                    "validation", args.evaluation_seed, args.evaluation_games,
                    f.sha256_file(checkpoint), f"retained-pilot:greedy:{seats}")
                try:
                    report = evaluate(checkpoint, games=args.evaluation_games,
                                      seed=args.evaluation_seed, model_seats=seats, mode="greedy")
                    report["evaluationSplit"] = "validation"
                    f.atomic_json_save(report, report_path)
                except BaseException:
                    registry.finish_evaluation(reservation, "failed")
                    raise
                registry.finish_evaluation(reservation, "completed", report_path.resolve())
                reports[label] = report
            comparison = f.compare_reports(reports["trained"], reports["initial"])
            comparisons[str(seats)] = {
                "initialScore": reports["initial"]["score"],
                "trainedScore": reports["trained"]["score"], **comparison}
        summary = {
            "status": "completed", "ppoGames": args.games, "comparisons": comparisons,
            "fullTeamImprovementEstablished": comparisons["3"]["confidence95"][0] > 0,
            "promotion": False, "finalHoldoutOpened": False,
        }
        f.atomic_json_save(summary, args.output_dir / "result.json")
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        f.atomic_json_save({"status": "failed", "error": str(error), "promotion": False},
                           args.output_dir / "result.json")
        raise


if __name__ == "__main__":
    main()
