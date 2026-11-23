"""Conservative evidence gate. Never deploy a model or open a holdout implicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import foundation as f
import seed_registry as seeds


def validate_report(report):
    games, seed = report["games"], report["seed"]
    f.validate_evaluation(games, seed)
    protocol = report["protocol"]
    if (protocol["games"] != games or protocol["seed"] != seed
            or protocol["modelSeats"] not in (1, 2, 3)
            or protocol["rounds"] not in ("fresh", "continued")
            or protocol["reward"] != "terminal-team-v1"
            or protocol["opponent"] != report["opponentSha256"]):
        raise ValueError("report protocol and measured fields disagree")
    pairs = report["pairedSeeds"]
    if len(pairs) != games // 2:
        raise ValueError("report paired-game count mismatch")
    outcomes = {"WIN": 0, "LOSS": 0, "DRAW_WE_COMPLETED": 0, "DRAW_THEY_COMPLETED": 0}
    means = []
    for index, pair in enumerate(pairs):
        if pair["seed"] != seed + index or len(pair["matches"]) != 2 or len(pair["scores"]) != 2:
            raise ValueError("report seed/pair structure mismatch")
        for score, match in zip(pair["scores"], pair["matches"]):
            outcome = match["outcome"]
            if outcome not in outcomes:
                raise ValueError("unknown terminal category")
            expected = 1.0 if outcome == "WIN" else 0.0 if outcome == "LOSS" else 0.5
            if score != expected or match["score"] != expected:
                raise ValueError("terminal category and score disagree")
            outcomes[outcome] += 1
        means.append(float(np.mean(pair["scores"])))
    if not np.isclose(report["score"], np.mean(means), atol=1e-12, rtol=0):
        raise ValueError("report score disagrees with paired data")
    if report["terminalDistribution"] != outcomes:
        raise ValueError("terminal distribution disagrees with paired data")
    if (report["wins"] != outcomes["WIN"] or report["losses"] != outcomes["LOSS"]
            or report["draws"] != outcomes["DRAW_WE_COMPLETED"] + outcomes["DRAW_THEY_COMPLETED"]):
        raise ValueError("W/L/D totals disagree")
    expected_fingerprint = hashlib.sha256(json.dumps(report["protocol"], sort_keys=True).encode()).hexdigest()
    if report["protocolFingerprint"] != expected_fingerprint:
        raise ValueError("report protocol fingerprint mismatch")
    return f.paired_bootstrap(means, seed ^ 0x9E3779B9)


def check_evidence(bundle, minimum_games=200, latency_limit_ms=1000):
    gates = {}
    evidence = {}
    for name in ("mathAudit", "privacyAudit", "actionAudit", "warmStartAudit", "onlineEvidence", "full", "fullAncestor",
                 "mixed", "mixedAncestor", "continued", "continuedAncestor"):
        path = bundle.get(name)
        if path is not None and Path(path).is_file():
            evidence[name] = json.loads(Path(path).read_text())
        else:
            gates[f"available:{name}"] = {"passed": False, "reason": "required measured evidence missing"}
    for name in ("full", "fullAncestor", "mixed", "mixedAncestor", "continued", "continuedAncestor"):
        if name in evidence:
            validate_report(evidence[name])
            gates[f"sample:{name}"] = {"passed": evidence[name]["games"] >= minimum_games,
                                     "games": evidence[name]["games"], "required": minimum_games}
    if "mathAudit" in evidence:
        audit = evidence["mathAudit"]
        gates["math"] = {"passed": audit.get("passed") is True and not audit.get("errors"),
                         "strengthClaim": False}
    if "privacyAudit" in evidence:
        audit = evidence["privacyAudit"]
        gates["privacy"] = {"passed": audit.get("passed") is True and audit.get("worlds", 0) >= 10000
                            and audit.get("productionSnapshotBlocked") is True
                            and audit.get("automaticHighestTributeConstrained") is True}
    if "warmStartAudit" in evidence:
        audit = evidence["warmStartAudit"]
        choices = [row for row in audit.get("positionsData", []) if row["legalActions"] > 1]
        agreement = float(np.mean([row["topAgreement"] for row in choices])) if choices else None
        gates["warmStartBehaviorRetention"] = {
            "passed": audit.get("behaviorPreserved") is True and agreement is not None and agreement > 0.98,
            "actorTransferredExactly": audit.get("actorTransferredExactly"),
            "topAgreement": agreement,
            "meanEpsilonToCategoricalKL": audit.get("meanEpsilonToCategoricalKL"),
            "reason": "calibration/distillation may change weights; retained behavior, not identical weights, is required"}
    if "actionAudit" in evidence:
        audit = evidence["actionAudit"]
        gates["samplingMeasured"] = {"passed": audit.get("positions", 0) >= 10000}
    if "onlineEvidence" in evidence:
        online = evidence["onlineEvidence"]
        gates["onlineLegalSupport"] = {
            "passed": online.get("passed") is True and online.get("allLegalActionsScored") is True
                      and online.get("maxLegalActions", 0) > 512
                      and 0 < online.get("maxBatchActions", 0) <= 512,
            "reason": "browser already chunks complete legal support into requests of at most 512"}
        gates["onlineLatency"] = {
            "passed": online.get("measuredDecisions", 0) >= 100
                      and online.get("p95DecisionLatencyMs", float("inf")) <= latency_limit_ms}
    if "full" in evidence:
        report = evidence["full"]
        gates["currentRules"] = {"passed": report["protocol"]["rules"] == f.rules_fingerprint()}
        confidence = validate_report(report)
        gates["fullTeamBeatsHeuristic"] = {
            "passed": report["protocol"]["modelSeats"] == 3
                      and report["protocol"]["rounds"] == "fresh"
                      and report["opponentSha256"] == "heuristic" and confidence[0] > 0.5,
            "score": report["score"], "confidence95": confidence}
        latency = report["latencyMs"]["p95"]
        gates["latency"] = {"passed": latency is not None and latency <= latency_limit_ms,
                            "p95Ms": latency, "limitMs": latency_limit_ms}
        candidate = bundle.get("candidateCheckpoint")
        gates["frozenCandidateIdentity"] = {
            "passed": bool(candidate and Path(candidate).is_file()
                           and f.sha256_file(candidate) == report["checkpointSha256"])}
        if "onlineEvidence" in evidence:
            online = evidence["onlineEvidence"]
            gates["onlinePolicyMatchesEvaluation"] = {
                "passed": online["checkpointSha256"] == report["checkpointSha256"]
                          and online["sampling"] == report["policy"]["sampling"] == "greedy"}
        if "fullAncestor" in evidence:
            comparison = f.compare_reports(report, evidence["fullAncestor"])
            gates["improvesAncestor"] = {"passed": comparison["confidence95"][0] > 0, **comparison}
            gates["ancestryIdentity"] = {
                "passed": report["metadata"].get("sourceSha256") == evidence["fullAncestor"]["checkpointSha256"]}
        if "warmStartAudit" in evidence:
            warm = evidence["warmStartAudit"]
            gates["warmStartIdentity"] = {
                "passed": warm.get("sourceSha256") == report["metadata"].get("sourceSha256")
                          and warm.get("initialSha256") == report["metadata"].get("initialSha256")}
        for name in ("mixed", "continued"):
            if name in evidence:
                other = evidence[name]
                gates[f"candidateIdentity:{name}"] = {
                    "passed": other["checkpointSha256"] == report["checkpointSha256"]}
                gates[f"protocol:{name}"] = {
                    "passed": (other["protocol"]["modelSeats"] == 1 and other["protocol"]["rounds"] == "fresh")
                    if name == "mixed" else other["protocol"]["rounds"] == "continued"}
                if f"{name}Ancestor" in evidence:
                    comparison = f.compare_reports(other, evidence[f"{name}Ancestor"])
                    gates[f"noEstablishedRegression:{name}"] = {
                        "passed": comparison["confidence95"][1] >= 0, **comparison}
                if name == "continued":
                    returns = sum(m["returnDecisions"] for p in other["pairedSeeds"] for m in p["matches"])
                    gates["realReturnCoverage"] = {"passed": returns > 0, "returnDecisions": returns}
    passed = bool(gates) and all(gate["passed"] for gate in gates.values())
    return {
        "schemaVersion": 1, "createdAt": f.now(), "gates": gates,
        "preselectionGatesPassed": passed, "deploymentAllowed": False,
        "finalHoldoutOpened": False,
        "decision": "eligible for preselected final evaluation, not deployment" if passed
                    else "blocked: no promotion and no long-run/structural upgrade justified by this evidence",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-games", type=int, default=200)
    parser.add_argument("--preregister-final", type=Path)
    parser.add_argument("--final-games", type=int, default=2000)
    parser.add_argument("--final-seed", type=int, default=seeds.SPLITS["final"][0])
    args = parser.parse_args()
    if args.minimum_games < 200:
        parser.error("release evidence requires at least 200 games per protocol")
    if args.report.exists() or (args.preregister_final and args.preregister_final.exists()):
        parser.error("evidence decision/plan files are immutable; choose a new path")
    bundle = json.loads(args.evidence.read_text())
    result = check_evidence(bundle, args.minimum_games)
    if args.preregister_final:
        if not result["preselectionGatesPassed"]:
            parser.error("failed evidence gates: cannot register/open a final holdout")
        full = json.loads(Path(bundle["full"]).read_text())
        low, high = seeds.SPLITS["final"]
        f.validate_evaluation(args.final_games, args.final_seed)
        if not low <= args.final_seed <= args.final_seed + args.final_games // 2 - 1 <= high:
            parser.error("final seed range outside sealed family")
        plan = {
            "schemaVersion": 1, "status": "sealed", "createdAt": f.now(),
            "candidateSha256": full["checkpointSha256"], "policy": full["policy"],
            "seed": args.final_seed, "games": args.final_games, "modelSeats": 3,
            "rounds": "fresh", "opponentSha256": "heuristic", "rules": f.rules_fingerprint(),
            "ancestorSha256": json.loads(Path(bundle["fullAncestor"]).read_text())["checkpointSha256"],
            "selectionEvidence": str(args.evidence.resolve()),
            "note": "preselected candidate only; completing validation never launches this",
        }
        f.atomic_json_save(plan, args.preregister_final)
    f.atomic_json_save(result, args.report)
    print(result["decision"])


if __name__ == "__main__":
    main()
