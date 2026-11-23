"""Measure actual frozen-model action distributions before choosing a PPO sampler."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

import foundation as f
from tournament import softmax


TEMPERATURES = (0.02, 0.05, 0.10, 0.20)
PHASES = ("fresh", "mid", "late", "returning")


def distribution_stats(scores, actions, temperature):
    probabilities = softmax(scores, temperature)
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    kinds = {kind: float(sum(p for p, action in zip(probabilities, actions) if action["kind"] == kind))
             for kind in ("play", "pass", "return", "ask")}
    top = int(np.argmax(scores))
    return {
        "topProbability": float(probabilities[top]), "entropy": entropy,
        "normalizedEntropy": entropy / np.log(len(scores)) if len(scores) > 1 else 0.0,
        "effectiveActions": float(np.exp(entropy)), "kindProbability": kinds,
    }


def summarize(rows):
    result = {}
    for phase in ("all", *PHASES):
        subset = rows if phase == "all" else [row for row in rows if row["phase"] == phase]
        if not subset:
            continue
        summary = {"positions": len(subset)}
        for key in ("legalActions", "scoreStd", "scoreRange", "topGap", "canonicalMultiplicity"):
            values = [row[key] for row in subset]
            summary[key] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                            "p95": float(np.percentile(values, 95)), "max": float(max(values))}
        summary["over512Actions"] = sum(row["legalActions"] > 512 for row in subset)
        summary["temperatures"] = {}
        for temperature in TEMPERATURES:
            label = f"{temperature:.2f}"
            stats = [row["temperatures"][label] for row in subset]
            summary["temperatures"][label] = {
                key: float(np.mean([item[key] for item in stats]))
                for key in ("topProbability", "entropy", "normalizedEntropy", "effectiveActions")}
            summary["temperatures"][label]["kindProbability"] = {
                kind: float(np.mean([item["kindProbability"][kind] for item in stats]))
                for kind in ("play", "pass", "return", "ask")}
        result[phase] = summary
    return result


def calibration_summary(report):
    rows = report["positionsData"]
    if len(rows) != report["positions"] or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("position count/seed uniqueness mismatch")
    groups = {}
    for phase in ("all", *PHASES):
        selected = rows if phase == "all" else [row for row in rows if row["phase"] == phase]
        choices = [row for row in selected if row["legalActions"] > 1]
        groups[phase] = {
            "positions": len(selected), "nonforcedPositions": len(choices),
            "forcedPositions": len(selected) - len(choices),
            "temperatures": {
                f"{temperature:.2f}": {
                    key: float(np.mean([row["temperatures"][f"{temperature:.2f}"][key] for row in choices]))
                    if choices else None
                    for key in ("topProbability", "normalizedEntropy", "effectiveActions")
                }
                for temperature in TEMPERATURES
            },
            "epsilonMeanTopProbability": float(np.mean(
                [row["epsilonTopProbability"] for row in choices])) if choices else None,
        }
    return {
        "schemaVersion": 1, "sourceSha256": report["checkpointSha256"],
        "positions": report["positions"], "groups": groups,
        "occupancy": report["samplingPolicy"],
        "phaseDefinitions": {
            "fresh": "seeded initial deal", "mid": "1-48 heuristic action prefix, not guaranteed global midgame",
            "late": "first active actor with at most 10 cards, not necessarily all-player late game",
            "returning": "authentic heuristic prior-round tribute continuation",
        },
        "warning": "Forced one-option turns inflate pooled top probability; calibration uses nonforced choices.",
        "decision": "Neither temperature tuning nor low loss proves stronger play; use paired tournament outcomes.",
    }


def audit(checkpoint, positions, seed, report, device="cpu", save_fixtures=False):
    if type(positions) is not int or positions < 4 or seed < 0 or seed + positions * 64 > 0xFFFFFFFF:
        raise ValueError("invalid audit position/seed range")
    report = Path(report)
    if report.exists():
        raise ValueError("audit report already exists")
    model, data, version = f.load_policy(checkpoint, device)
    if version != 2:
        raise ValueError("distribution audit requires a v2 actor")
    digest = f.sha256_file(checkpoint)
    env = f.NodeEnvironment(script=f.ROOT / "audit-env.js")
    rows, pending = [], []
    fixture_dir = report.parent / (report.stem + "-fixtures")
    if save_fixtures:
        fixture_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def flush():
        if not pending:
            return
        scores = f.score_many(model, [
            (state["observation"], [a["features"] for a in state["actions"]],
             f.relative_history(state["events"], state["player"]))
            for _, _, state in pending])
        for (index, phase, state), values in zip(pending, scores):
            actions = state["actions"]
            ordered = np.sort(values)
            identities = [json.dumps({"kind": a["kind"], "target": a.get("target"),
                                      "features": a["features"]}, sort_keys=True) for a in actions]
            multiplicities = Counter(identities)
            rows.append({
                "seed": seed + index * 64, "phase": phase, "player": state["player"],
                "historyLength": len(state["events"]), "legalActions": len(actions),
                "scoreStd": float(np.std(values)), "scoreRange": float(np.ptp(values)),
                "topGap": float(ordered[-1] - ordered[-2]) if len(values) > 1 else 0.0,
                "canonicalMultiplicity": max(multiplicities.values()),
                "epsilonTopProbability": 0.85 + 0.15 / len(values),
                "temperatures": {f"{t:.2f}": distribution_stats(values, actions, t) for t in TEMPERATURES},
            })
            if save_fixtures:
                np.savez_compressed(fixture_dir / f"position-{index:05d}.npz",
                                    observation=np.asarray(state["observation"], np.float32),
                                    history=f.relative_history(state["events"], state["player"]),
                                    actions=np.asarray([a["features"] for a in actions], np.float32),
                                    scores=values, seed=seed + index * 64, player=state["player"])
        pending.clear()

    try:
        for index in range(positions):
            phase = PHASES[index % 4]
            state = env.state("sample", seed=seed + index * 64, phase=phase)
            pending.append((index, phase, state))
            if len(pending) >= 16:
                flush()
            if (index + 1) % 100 == 0:
                print(f"positions={index + 1}/{positions} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        flush()
    finally:
        env.close()
    if f.sha256_file(checkpoint) != digest:
        raise RuntimeError("checkpoint changed during audit")
    result = {
        "schemaVersion": 1, "createdAt": f.now(), "checkpointSha256": digest,
        "metadata": data["metadata"], "positions": positions, "seed": seed,
        "rules": f.rules_fingerprint(), "temperatureGrid": list(TEMPERATURES),
        "samplingPolicy": "independent seeded resets; heuristic-generated public prefixes and real tribute continuations",
        "epsilonReference": {"epsilon": 0.15, "argmaxProbability": "0.85 + 0.15/N"},
        "summary": summarize(rows), "positionsData": rows,
        "gate": "measurement only; choose sampler with paired outcomes, not score scale alone",
        "fullIndependentPositionGatePassed": positions >= 10000,
        "elapsedSeconds": time.perf_counter() - started,
    }
    f.atomic_json_save(result, report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=10000000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--save-fixtures", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit(args.checkpoint, args.positions, args.seed, args.report, args.device, args.save_fixtures)
    stats = result["summary"]["all"]["temperatures"]["0.20"]
    print(f"T=.20 mean P(top)={stats['topProbability']:.5f}; normalized entropy={stats['normalizedEntropy']:.5f}")


if __name__ == "__main__":
    main()
