"""Paired sampler ablation on a fixed validation family; no training or holdout access."""

import argparse
import csv
from pathlib import Path

import torch

import foundation as f
import seed_registry as seeds
from tournament import evaluate


def write_csv(summary, path):
    with Path(path).open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("modelSeats", "sampling", "games", "wins", "losses", "draws",
                         "score", "ciLow", "ciHigh", "pairedDelta", "deltaCiLow", "deltaCiHigh"))
        for row in summary["cases"]:
            delta = row["versusOwnGreedy"]
            writer.writerow((row["modelSeats"], row["sampling"], summary["games"],
                             row["wins"], row["losses"], row["draws"], row["score"],
                             *row["confidence95"], delta["delta"], *delta["confidence95"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=seeds.SPLITS["validation"][0])
    parser.add_argument("--model-seats", type=int, nargs="+", choices=(1, 2, 3), default=(1, 3))
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds.validate_split("validation", args.seed, args.games)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checkpoint = args.report_dir / "frozen-candidate.pt"
    digest = f.freeze_checkpoint(args.checkpoint, checkpoint)
    settings = [("greedy", "greedy", 0.2), ("epsilon", "epsilon", 0.2),
                *[(f"temperature-{t:.2f}", "categorical", t) for t in (0.02, 0.05, 0.10, 0.20)]]
    summary = {"checkpointSha256": digest, "seed": args.seed, "games": args.games,
               "split": "validation; used for selection, not a final holdout", "cases": []}
    for seats in args.model_seats:
        reference = None
        for label, sampling, temperature in settings:
            report = args.report_dir / f"seats-{seats}-{label}.json"
            reservation = seeds.record_evaluation("validation", args.seed, args.games, digest,
                                                   f"sampling-sweep:{seats}:{label}")
            try:
                result = evaluate(checkpoint, games=args.games, seed=args.seed,
                                  model_seats=seats, mode=sampling, temperature=temperature)
                result["evaluationSplit"] = "validation"
                f.atomic_json_save(result, report)
            except BaseException:
                seeds.finish_evaluation(reservation, "failed")
                raise
            seeds.finish_evaluation(reservation, "completed", report.resolve())
            if reference is None:
                reference = result
            row = {"modelSeats": seats, "sampling": label, "score": result["score"],
                   "wins": result["wins"], "losses": result["losses"], "draws": result["draws"],
                   "confidence95": result["confidence95"], "versusOwnGreedy": f.compare_reports(result, reference),
                   "report": str(report.resolve())}
            summary["cases"].append(row)
            f.atomic_json_save(summary, args.report_dir / "summary.json")
            print(f"seats={seats} {label}: {result['score']:.4f}, paired delta="
                  f"{row['versusOwnGreedy']['delta']:+.4f}", flush=True)
    write_csv(summary, args.report_dir / "baseline-tournament.csv")


if __name__ == "__main__":
    main()
