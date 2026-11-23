"""Distinguish exact actor transfer from retention of epsilon-greedy behavior."""

import argparse
from pathlib import Path

import numpy as np
import torch

import foundation as f
from tournament import softmax


def audit(source, initial, positions=32, seed=20_030_000):
    teacher, source_data, version = f.load_policy(source)
    student, data, student_version = f.load_policy(initial)
    if version != 2 or student_version != 2:
        raise ValueError("requires a v2 source and PPO initial checkpoint")
    if data["metadata"].get("ppoGames") != 0:
        raise ValueError("warm-start audit requires an untrained PPO initial checkpoint")
    if data["metadata"].get("sourceSha256") != f.sha256_file(source):
        raise ValueError("initial checkpoint has different ancestry")
    if positions < 4:
        raise ValueError("at least four stratified positions required")
    temperature = data["metadata"]["config"]["temperature"]
    epsilon = source_data["metadata"]["epsilon"]
    env = f.NodeEnvironment(script=f.ROOT / "audit-env.js")
    rows = []
    try:
        for index in range(positions):
            phase = ("fresh", "mid", "late", "returning")[index % 4]
            state = env.state("sample", seed=seed + index * 64, phase=phase)
            entry = (state["observation"], [a["features"] for a in state["actions"]],
                     f.relative_history(state["events"], state["player"]))
            before = f.score_many(teacher, [entry])[0]
            after = f.score_many(student, [entry])[0]
            target = np.full(len(before), epsilon / len(before), dtype=np.float64)
            target[int(np.argmax(before))] += 1 - epsilon
            probabilities = softmax(after, temperature)
            finite = bool(np.all(probabilities > 0))
            kl = float(np.sum(target * np.log(target / probabilities))) if finite else None
            rows.append({"phase": phase, "legalActions": len(before),
                         "maxRawLogitDifference": float(np.max(np.abs(after - before))),
                         "topAgreement": int(np.argmax(before) == np.argmax(after)),
                         "epsilonToCategoricalKL": kl, "infiniteKL": not finite,
                         "categoricalTopProbability": float(probabilities[int(np.argmax(before))])})
    finally:
        env.close()
    choice_rows = [row for row in rows if row["legalActions"] > 1]
    finite_kls = [row["epsilonToCategoricalKL"] for row in choice_rows if not row["infiniteKL"]]
    mean_kl = float(np.mean(finite_kls)) if finite_kls else None
    return {
        "schemaVersion": 1, "sourceSha256": f.sha256_file(source), "initialSha256": f.sha256_file(initial),
        "sourceGames": source_data["metadata"]["games"], "positions": positions,
        "actorTransferredExactly": all(row["maxRawLogitDifference"] <= 1e-6 for row in rows),
        "nonforcedPositions": len(choice_rows), "meanEpsilonToCategoricalKL": mean_kl,
        "behaviorPreserved": bool(choice_rows and mean_kl is not None and mean_kl < 0.05
                                  and not any(row["infiniteKL"] for row in choice_rows)),
        "temperature": temperature, "positionsData": rows, "strengthClaim": False,
        "sourceEpsilon": epsilon,
        "notice": "Equal actor weights/logits do not preserve an epsilon-greedy policy when changed to categorical sampling.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("warm-start report already exists")
    torch.set_num_threads(1)
    result = audit(args.source, args.initial, args.positions)
    f.atomic_json_save(result, args.report)
    print(f"actor exact={result['actorTransferredExactly']}; behavior retained={result['behaviorPreserved']}; "
          f"nonforced mean KL={result['meanEpsilonToCategoricalKL']}")


if __name__ == "__main__":
    main()
