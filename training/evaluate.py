"""Evaluate frozen action values against the heuristic using paired seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import uuid


def paired_bootstrap(pair_scores, seed: int, samples: int = 10000):
    import numpy as np
    if not pair_scores:
        raise ValueError("at least one seed pair is required")
    rng = np.random.default_rng(seed)
    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    means = np.empty(samples, dtype=np.float64)
    # Resample pairs, not individual games: each pair shares its deal seed.
    for start in range(0, samples, 100):
        stop = min(start + 100, samples)
        indices = rng.integers(0, len(pair_scores), size=(stop - start, len(pair_scores)))
        means[start:stop] = pair_scores[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def promote_checkpoint(checkpoint, destination, games, confidence, expected_sha256=None):
    from daguai_rl import sha256_file
    if games < 2000 or confidence[0] <= 0.5:
        raise ValueError("promotion refused: requires >=2000 games and lower paired 95% CI > 0.5")
    destination = Path(destination)
    if Path(checkpoint).resolve() == destination.resolve():
        raise ValueError("promotion destination must differ from source checkpoint")
    expected_sha256 = expected_sha256 or sha256_file(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        with Path(checkpoint).open("rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise RuntimeError("checkpoint changed while copying; promotion refused")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    teammates = getattr(args, "teammates", "model")
    if teammates not in ("model", "heuristic"):
        raise ValueError("teammates must be model or heuristic")
    if teammates == "heuristic" and args.promote:
        raise ValueError("promotion refused: heuristic teammates do not validate full model-team deployment")
    import numpy as np
    import torch
    from daguai_rl import (
        NodeEnvironment, atomic_json_save, device_for, load_checkpoint, score_many,
        seed_everything, sha256_file, team_reward,
    )
    device = device_for(args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed_everything(args.seed)
    digest = sha256_file(args.checkpoint)
    model, data = load_checkpoint(args.checkpoint, device)
    metadata = data["metadata"]
    del data
    totals = {"wins": 0, "losses": 0, "draws": 0}
    pair_scores, pairs = [], []
    environment = NodeEnvironment(max_actions=args.max_actions)
    try:
        for pair_index in range(args.games // 2):
            seed = args.seed + pair_index
            scores, outcomes, action_counts, pair_learner_seats = [], [], [], []
            for learner_team in (0, 1):
                learner_seats = ([learner_team + 2 * (seed % 3)] if teammates == "heuristic"
                                 else [learner_team, learner_team + 2, learner_team + 4])
                pair_learner_seats.append(learner_seats)
                state = environment.state("reset", seed=seed)
                actions = 0
                while not state["terminal"]:
                    if actions >= args.max_actions:
                        raise RuntimeError(f"evaluation seed={seed} exceeded --max-actions; no truncated result")
                    if state["phase"] == "returning" or state["player"] not in learner_seats:
                        state = environment.state("heuristic")
                    else:
                        features = [action["features"] for action in state["actions"]]
                        scores_now = score_many(model, [(state["observation"], features)])[0]
                        state = environment.state("step", index=int(np.argmax(scores_now)))
                    actions += 1
                reward = team_reward(learner_team, state["result"]["winner"])
                outcome = "wins" if reward > 0 else "losses" if reward < 0 else "draws"
                totals[outcome] += 1
                scores.append((reward + 1) / 2)
                outcomes.append(outcome)
                action_counts.append(actions)
            pair_scores.append(sum(scores) / 2)
            pairs.append({"seed": seed, "scores": scores, "outcomes": outcomes, "actions": action_counts,
                          "learnerSeats": pair_learner_seats})
            if (pair_index + 1) % 25 == 0 or pair_index == args.games // 2 - 1:
                print(f"evaluated={2 * (pair_index + 1)} wins={totals['wins']} "
                      f"losses={totals['losses']} draws={totals['draws']}", flush=True)
    finally:
        environment.close()
    if sha256_file(args.checkpoint) != digest:
        raise RuntimeError("checkpoint changed during evaluation; report and promotion refused")
    confidence = paired_bootstrap(pair_scores, args.seed ^ 0x9E3779B9)
    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()), "checkpointSha256": digest,
        "metadata": metadata, "device": str(device), "games": args.games, **totals,
        "score": (totals["wins"] + 0.5 * totals["draws"]) / args.games,
        "confidence95": confidence, "confidenceMethod": "paired-seed percentile bootstrap",
        "bootstrapSamples": 10000, "seed": args.seed, "maxActions": args.max_actions,
        "opponent": "existing engine heuristic", "learnerReturns": "heuristic",
        "teammates": teammates,
        "learnerSeatPolicy": "all three team seats" if teammates == "model" else "one seat: team + 2*(seed % 3)",
        "pairedSeeds": pairs,
        "seedSeparation": {
            "evaluationSeedRange": [args.seed, args.seed + args.games // 2 - 1],
            "trainingSeed": metadata.get("seed"),
            "notice": "Held-out evaluation defaults use a separate seed range. Ensuring these "
                      "seeds were never used in training, including resumed runs, is your responsibility. "
                      "Evaluation against this heuristic is not a claim of expert-level play.",
        },
        "promotion": {"requested": bool(args.promote),
                      "eligible": teammates == "model" and args.games >= 2000 and confidence[0] > 0.5},
    }
    if args.promote:
        report["promotion"]["destination"] = str(args.promote.resolve())
        if not report["promotion"]["eligible"]:
            report["promotion"]["error"] = "requires >=2000 games and lower paired 95% CI > 0.5"
            atomic_json_save(report, args.report)
            raise ValueError("promotion refused: " + report["promotion"]["error"])
        promote_checkpoint(args.checkpoint, args.promote, args.games, confidence, expected_sha256=digest)
        report["promotion"]["completed"] = True
    atomic_json_save(report, args.report)
    print(f"score={report['score']:.4f} paired95%CI={confidence} report={args.report.resolve()}", flush=True)
    return 0


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--games", type=int, default=2000, help="even number of paired evaluation games")
    parser.add_argument("--seed", type=int, default=1000000000, help="held-out seed-range start")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--report", type=Path, default=root / "reports" / "evaluation.json")
    parser.add_argument("--max-actions", type=int, default=1500)
    parser.add_argument("--teammates", choices=("model", "heuristic"), default="model",
                        help="model: full learner team; heuristic: one rotating learner seat with two heuristic teammates")
    parser.add_argument("--promote", type=Path, help="atomic copy only after passing confidence gate")
    args = parser.parse_args(argv)
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive even number")
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.promote and args.teammates == "heuristic":
        parser.error("--promote requires --teammates model; heuristic teammates do not validate full model-team deployment")
    if not 0 <= args.seed <= 0xFFFFFFFF or args.seed + args.games // 2 - 1 > 0xFFFFFFFF:
        parser.error("evaluation seeds must fit uint32 without wrapping")
    if args.report.resolve() == args.checkpoint.resolve():
        parser.error("--report must not overwrite the checkpoint")
    if args.promote and args.report.resolve() == args.promote.resolve():
        parser.error("--report and --promote must have different paths")
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Evaluation interrupted; no promotion.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
