"""Enforceable production trace/GAE and frozen-policy replay diagnostics.

CLI uses bounded diagnostic rounds, never evaluation/promotion seeds. Metrics are
plumbing evidence, not evidence of playing strength or calibrated probabilities.
Optional JSONL is TRAINING-PRIVATE own/public diagnostics, NOT actor input:
each episode contains future public events, which must be sliced at row.prefix.
"""

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.distributions import Categorical

from contracts import (
    CONTRACT_VERSION, REWARD_VERSION, SHAPING_VERSION, Decision, Episode,
    model_digest, validate_episode,
)
import foundation as f
import strategic_protocol as strategic

TRACE_SCHEMA_VERSION = 1
DEBUG_SEED_START = 20_000_000
DEBUG_SEED_END = 21_000_000
MAX_TRACE_JSONL_BYTES = 128 * 1024 * 1024


class TraceAuditError(ValueError):
    def __init__(self, report):
        self.report = report
        details = ""
        if "maximumValueReplayError" in report:
            details = (
                f" (logprob={report['maximumLogprobReplayError']:.9g},"
                f" value={report['maximumValueReplayError']:.9g},"
                f" ratio={report['maximumRatioReplayError']:.9g},"
                f" tolerance={report['replayTolerance']:.9g})")
        super().__init__("PPO trace audit failed: " + "; ".join(report["errors"]) + details)


def export_trace_jsonl(path, batch, config, seed, checkpoint=None):
    """Exclusively create a bounded private manifest/episode/decision JSONL.

    Each decision exposes only its actor's observation and legal support. The
    container combines multiple actors' own observations; it is never a public
    artifact or inference request. Successor indices are episode-local.
    """
    if (not 1 <= len(batch.episodes) <= 4 or type(seed) is not int
            or not DEBUG_SEED_START <= seed < DEBUG_SEED_END
            or seed + len(batch.episodes) > DEBUG_SEED_END or config["seed"] != seed):
        raise ValueError("trace export requires 1..4 rounds in the 20m debug seed family")
    audit_batch(batch, config=config)
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError("trace export path must end in .jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, "x", encoding="utf-8",
                  opener=lambda name, flags: os.open(name, flags, 0o600))
    digest = hashlib.sha256()
    size, count = 0, 0

    def emit(record):
        nonlocal size, count
        line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        encoded = line.encode("utf-8")
        size += len(encoded)
        if size > MAX_TRACE_JSONL_BYTES:
            raise ValueError("bounded debug trace JSONL size limit exceeded")
        stream.write(line)
        digest.update(encoded)
        count += 1

    def contract(record, contract_type, omit=()):
        return {field.name: (
            getattr(record, field.name).tolist()
            if isinstance(getattr(record, field.name), np.ndarray)
            else getattr(record, field.name))
            for field in fields(contract_type) if field.name not in omit}

    try:
        with stream:
            emit({
                "record_type": "manifest", "trace_schema_version": TRACE_SCHEMA_VERSION,
                "classification": "training-private-own-public-diagnostics",
                "not_public_or_inference_input": True,
                "history_rule": "relative_history(episode.events[:decision.prefix], decision.player)",
                "warning": "Complete public events include future events. Never merge other seats' own observations into actor input.",
                "protocol": strategic.SIZES if batch.episodes[0].encoding_version == 3 else f.SIZES,
                "decision_contract_version": CONTRACT_VERSION,
                "reward_version": REWARD_VERSION, "shaping_version": SHAPING_VERSION,
                "seed": seed, "seed_family": "20m-debug", "config": config,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "behavior_version": batch.behavior_version,
                "behavior_digest_before": batch.behavior_digest_before,
                "behavior_digest_after": batch.behavior_digest_after,
                "episodes": len(batch.episodes),
                "decisions": sum(len(e.decisions) for e in batch.episodes),
            })
            for episode in batch.episodes:
                emit({"record_type": "episode", "trace_schema_version": TRACE_SCHEMA_VERSION,
                      "episode": contract(episode, Episode, omit=("decisions",)),
                      "decision_count": len(episode.decisions)})
                for index, decision in enumerate(episode.decisions):
                    emit({"record_type": "decision", "trace_schema_version": TRACE_SCHEMA_VERSION,
                          "decision_index": index,
                          "decision": {**contract(decision, Decision), "team": decision.team,
                                       "chosen_action_identity": decision.chosen_action_identity}})
    except BaseException:
        path.unlink()
        raise
    return {"path": str(path), "traceSchemaVersion": TRACE_SCHEMA_VERSION,
            "records": count, "bytes": size, "sha256": digest.hexdigest(),
            "classification": "training-private-own-public-diagnostics"}


@torch.no_grad()
def replay_predictions(model, entries, temperature, batch_size=64):
    """Full-prefix production forward on the exact saved ordered legal support."""
    import team_ppo as p

    logprobs, values, entropies, normalized, maximum, support = [], [], [], [], [], []
    device = next(model.parameters()).device
    for start in range(0, len(entries), batch_size):
        chunk = entries[start:start + batch_size]
        states = [(d.observation, d.actions, e.history(d)) for e, d in chunk]
        scores, predictions = model.policy_value(*p._tensor_states(model, states))
        values.extend(predictions.detach().cpu().tolist())
        for logits, (_, d) in zip(scores, chunk):
            dist = Categorical(logits=logits / temperature)
            entropy = float(dist.entropy())
            logprobs.append(float(dist.log_prob(torch.tensor(d.index, device=device))))
            entropies.append(entropy)
            normalized.append(entropy / np.log(len(d.actions)) if len(d.actions) > 1 else 0.0)
            maximum.append(float(dist.probs.max()))
            support.append(float(np.exp(entropy)))
    return {key: np.asarray(value, np.float64) for key, value in (
        ("logprob", logprobs), ("value", values), ("entropy", entropies),
        ("normalizedEntropy", normalized), ("pmax", maximum), ("effectiveSupport", support))}


def audit_batch(batch, model=None, config=None, enforce=True):
    """Validate contracts; call production seat_gae(lambda=1) as the MC oracle."""
    import team_ppo as p

    config = config or p.CORE_DEFAULTS
    errors, telescope_errors, mc_errors = [], [], []
    terminal_counts, signs = Counter(), Counter()
    if config["gamma"] != 1:
        errors.append("gamma must equal 1")
    if batch.behavior_digest_before != batch.behavior_digest_after:
        errors.append("behavior digest changed during collection")
    ids = set()
    expected_encoding = (getattr(model, "sizes", f.SIZES)["encodingVersion"] if model is not None
                         else batch.episodes[0].encoding_version if batch.episodes else f.ENCODING_VERSION)
    for episode in batch.episodes:
        try:
            validate_episode(episode, p.potential)
            if episode.encoding_version != expected_encoding:
                raise ValueError("trace feature encoding does not match batch/model")
            if episode.episode_id in ids:
                raise ValueError("duplicate episode ID")
            ids.add(episode.episode_id)
            if episode.behavior_version != batch.behavior_version:
                raise ValueError("mixed behavior version")
            if episode.behavior_digest != batch.behavior_digest_before:
                raise ValueError("episode/batch behavior digest mismatch")
            if episode.potential_scale != config["potential_scale"]:
                raise ValueError("trace/update potential scale mismatch")
            if episode.temperature != config["temperature"]:
                raise ValueError("trace/update behavior temperature mismatch")
            advantage, returns = p.seat_gae(
                episode, gamma=config["gamma"], gae_lambda=1,
                potential_scale=config["potential_scale"])
            for actor in range(6):
                indices = [i for i, d in enumerate(episode.decisions) if d.player == actor]
                if not indices:
                    continue
                ds = [episode.decisions[i] for i in indices]
                shaped = [d.terminal_reward + d.potential_successor - d.potential_before for d in ds]
                outcome = f.team_reward(actor, episode.winner)
                telescope_errors.append(abs(sum(shaped) - (outcome - ds[0].potential_before)))
                if sum(d.next_actor_index is None for d in ds) != 1:
                    raise ValueError("terminal assignment must occur exactly once per learner actor")
                terminal_counts[str(actor)] += 1
                signs[str(int(outcome))] += 1
                mc = 0.0
                for offset in reversed(range(len(indices))):
                    mc += shaped[offset]
                    i = indices[offset]
                    mc_errors.extend((abs(float(returns[i]) - mc),
                                      abs(float(advantage[i]) - (mc - ds[offset].old_value))))
        except (ValueError, TypeError, IndexError, KeyError) as error:
            errors.append(f"episode {episode.episode_id}: {error}")
    result = {
        "contractVersion": CONTRACT_VERSION, "episodes": len(batch.episodes),
        "decisions": sum(len(e.decisions) for e in batch.episodes),
        "maximumTelescopeError": max(telescope_errors, default=0.0),
        "maximumLambdaOneMCError": max(mc_errors, default=0.0),
        "terminalAssignmentsByActor": dict(terminal_counts), "terminalSigns": dict(signs),
        "behaviorDigestBefore": batch.behavior_digest_before,
        "behaviorDigestAfter": batch.behavior_digest_after,
    }
    if result["maximumTelescopeError"] > 1e-6 or result["maximumLambdaOneMCError"] > 1e-5:
        errors.append("production GAE/telescope oracle mismatch")
    if model is not None and not errors:
        digest = model_digest(model)
        result["replayBehaviorDigest"] = digest
        if digest != batch.behavior_digest_before:
            errors.append("replay model is not frozen rollout behavior")
        entries = [(e, d) for e in batch.episodes for d in e.decisions]
        result["replayGruTokens"] = sum(d.prefix for _, d in entries)
        if entries:
            current = replay_predictions(model, entries, config["temperature"], config["batch_size"])
            old = np.asarray([d.old_logprob for _, d in entries])
            old_values = np.asarray([d.old_value for _, d in entries])
            result.update(
                maximumLogprobReplayError=float(np.max(np.abs(current["logprob"] - old))),
                maximumValueReplayError=float(np.max(np.abs(current["value"] - old_values))),
                maximumRatioReplayError=float(np.max(np.abs(np.exp(current["logprob"] - old) - 1))),
            )
            tolerance = 1e-5 if next(model.parameters()).device.type == "cpu" else 3e-4
            result["replayTolerance"] = tolerance
            if any(result[key] > tolerance for key in (
                    "maximumLogprobReplayError", "maximumValueReplayError", "maximumRatioReplayError")):
                errors.append("frozen behavior logprob/value/ratio replay mismatch")
    result["errors"] = errors
    result["passed"] = not errors
    if enforce and errors:
        raise TraceAuditError(result)
    return result


def main():
    import team_ppo as p

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEBUG_SEED_START + 1)
    parser.add_argument("--output", type=Path, default=f.ROOT / "reports" / "p0-math-audit.json",
                        help="new audit JSON report path, or '-' for stdout only; never overwrite")
    parser.add_argument("--trace-jsonl", type=Path,
                        help="new training-private own/public complete trace; never inference input")
    args = parser.parse_args()
    if (not 1 <= args.games <= 4 or not DEBUG_SEED_START <= args.seed < DEBUG_SEED_END
            or args.seed + args.games > DEBUG_SEED_END):
        parser.error("bounded audit: games 1..4; seeds must stay in the 20m debug family")
    stdout_only = args.output == Path("-")
    if (not stdout_only and args.output.exists()) or (args.trace_jsonl and args.trace_jsonl.exists()):
        parser.error("diagnostic outputs must be new paths; existing reports/traces are immutable")
    if args.trace_jsonl and args.trace_jsonl.suffix != ".jsonl":
        parser.error("--trace-jsonl path must end in .jsonl")
    if args.trace_jsonl and not stdout_only and args.trace_jsonl.resolve() == args.output.resolve():
        parser.error("audit report and private trace must use different paths")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = p.load_checkpoint(args.checkpoint)[0] if args.checkpoint else p.ActorCritic().eval()
    config = {**p.CORE_DEFAULTS, "workers": 1, "rollout_games": args.games, "seed": args.seed}
    if args.checkpoint:
        config.update(model.metadata.get("config", {}))
        config.update(workers=1, rollout_games=args.games, seed=args.seed)
    sizes = model.sizes
    bridge = f.ROOT / "strategic-env.js" if sizes == strategic.SIZES else None
    workers = [{"env": f.NodeEnvironment(script=bridge, sizes=sizes,
                                         max_actions=config["max_actions"]), "state": None}]
    try:
        batch, coverage, _ = p.collect_batch(
            model, workers, args.games, random.Random(args.seed), config, 0, league=[model])
        result = audit_batch(batch, model, config)
        result.update(coverage=coverage, rolloutDiagnostics=batch.diagnostics, seed=args.seed,
                      checkpoint=str(args.checkpoint) if args.checkpoint else None,
                      protocol=sizes, algorithm=p.ALGORITHM, strengthClaim=False,
                      rewardVersion=REWARD_VERSION, shapingVersion=SHAPING_VERSION,
                      rulesFingerprint=f.rules_fingerprint(),
                      decisionFields=[field.name for field in fields(Decision)],
                      episodeFields=[field.name for field in fields(Episode)],
                      config=config, device="cpu", precision="float32",
                      coreSourceSha256=hashlib.sha256(
                          Path(p.__file__).read_bytes()
                          + Path(__file__).with_name("contracts.py").read_bytes()
                          + Path(__file__).read_bytes()).hexdigest())
        if args.trace_jsonl:
            result["traceExport"] = export_trace_jsonl(
                args.trace_jsonl, batch, config, args.seed, args.checkpoint)
        payload = json.dumps(result, indent=2, allow_nan=False)
        if not stdout_only:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload + "\n")
        print(payload)
    finally:
        for worker in workers:
            worker["env"].close()


if __name__ == "__main__":
    main()
