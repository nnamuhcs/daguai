"""Explicit evaluation splits; final results require a preselected release plan."""

from __future__ import annotations

import json
import fcntl
from contextlib import contextmanager
from pathlib import Path
import uuid

import foundation as f

SPLITS = {
    "train": (3_000_000, 3_999_999),
    "audit": (10_000_000, 10_999_999),
    "debug": (20_000_000, 20_999_999),
    "validation": (1_100_000_000, 1_100_999_999),
    "confirmation": (1_200_000_000, 1_200_999_999),
    "continued": (2_500_000_000, 2_500_999_999),
    "final": (3_500_000_000, 3_500_999_999),
}
REGISTRY = f.ROOT / "manifests" / "seed-splits-v1.json"


@contextmanager
def registry_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def validate_split(split, seed, games, release_plan=None, candidate_sha=None):
    f.validate_evaluation(games, seed)
    if split not in SPLITS:
        raise ValueError("unknown evaluation split")
    low, high = SPLITS[split]
    if not low <= seed <= seed + games // 2 - 1 <= high:
        raise ValueError(f"seed range lies outside {split} split")
    if split == "final":
        if release_plan is None:
            raise ValueError("sealed final split requires a preselected release plan")
        plan = json.loads(Path(release_plan).read_text())
        if (plan.get("status") != "sealed" or plan.get("candidateSha256") != candidate_sha
                or plan.get("seed") != seed or plan.get("games") != games):
            raise ValueError("release plan does not match the frozen candidate and evaluation")
    return seed, seed + games // 2 - 1


def validate_final_protocol(plan_path, policy, model_seats, rounds, opponent_sha):
    plan = json.loads(Path(plan_path).read_text())
    expected = {"policy": policy, "modelSeats": model_seats, "rounds": rounds,
                "opponentSha256": opponent_sha, "rules": f.rules_fingerprint()}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("final protocol differs from the preselected release plan")


def discover_existing_usage(root=None):
    root = Path(root or f.ROOT)
    usage = []
    for path in sorted(root.glob("**/*.json")):
        if path.name == REGISTRY.name:
            continue
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            continue
        seed, games = value.get("seed"), value.get("games")
        if type(seed) is int and type(games) is int and games >= 2 and games % 2 == 0:
            usage.append({"report": str(path.resolve()), "seedStart": seed,
                          "seedEnd": seed + games // 2 - 1, "games": games,
                          "checkpointSha256": value.get("checkpointSha256"),
                          "classification": "previously observed; not sealed"})
    return usage


def ensure_registry(path=REGISTRY):
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("schemaVersion") != 1 or data.get("splits") != {
                name: list(bounds) for name, bounds in SPLITS.items()}:
            raise ValueError("seed registry configuration mismatch")
        return data
    historical = discover_existing_usage()
    final_low, final_high = SPLITS["final"]
    if any(row["seedStart"] <= final_high and row["seedEnd"] >= final_low for row in historical):
        raise ValueError("proposed final split already appears in historical reports")
    data = {"schemaVersion": 1, "splits": {name: list(bounds) for name, bounds in SPLITS.items()},
            "historicalUsage": historical, "evaluations": [],
            "notes": ["Old 1000000000 validation was reused for selection.",
                      "Old 2000000000 prefix was used by actual tiny integration evaluations.",
                      "Neither old prefix is represented as a newly sealed holdout."]}
    f.atomic_json_save(data, path)
    return data


def record_evaluation(split, seed, games, candidate_sha, protocol_digest, path=REGISTRY,
                      release_plan=None):
    validate_split(split, seed, games, release_plan, candidate_sha)
    identifier = uuid.uuid4().hex
    with registry_lock(path):
        data = ensure_registry(path)
        if split == "final" and any(
                row["split"] == "final" and row["seed"] == seed for row in data["evaluations"]):
            raise ValueError("final seed family already opened; do not retune on it")
        data["evaluations"].append({
            "id": identifier, "split": split, "seed": seed, "games": games,
            "candidateSha256": candidate_sha, "protocolDigest": protocol_digest,
            "status": "reserved", "createdAt": f.now(),
        })
        f.atomic_json_save(data, path)
    return identifier


def finish_evaluation(identifier, status, report=None, path=REGISTRY):
    if status not in ("completed", "failed"):
        raise ValueError("invalid evaluation status")
    with registry_lock(path):
        data = ensure_registry(path)
        row = next((row for row in data["evaluations"] if row["id"] == identifier), None)
        if row is None or row["status"] != "reserved":
            raise ValueError("evaluation reservation not found or already finalized")
        row.update(status=status, finishedAt=f.now(), report=str(report) if report else None)
        f.atomic_json_save(data, path)
