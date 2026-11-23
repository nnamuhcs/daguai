"""Frozen paired tournaments with observable terminal diagnostics and continued rounds."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

import foundation as f
import seed_registry as seeds
import strategic_protocol as strategic


class Policy:
    def __init__(self, checkpoint="heuristic", mode="greedy", temperature=0.2, epsilon=0.15,
                 device="cpu", cache=True):
        if mode not in ("greedy", "categorical", "epsilon"):
            raise ValueError("invalid sampling mode")
        if not math.isfinite(temperature) or temperature <= 0 or not 0 <= epsilon <= 1:
            raise ValueError("invalid sampling temperature or epsilon")
        self.mode, self.temperature, self.epsilon = mode, temperature, epsilon
        self.cache_enabled = cache
        self.checkpoint = checkpoint
        self.model = None
        self.digest = checkpoint
        self.version = None
        if checkpoint not in ("heuristic", "random"):
            self.model, self.data, self.version = f.load_policy(checkpoint, device, allow_strategic=True)
            self.digest = f.sha256_file(checkpoint)
        self.reset()

    def reset(self):
        self.hidden = {}
        self.gru_steps = 0
        self.strategic_cache = None

    @torch.no_grad()
    def scores(self, state, events):
        if self.model is None:
            raise ValueError("a rule/random policy does not have neural scores")
        if self.version == 1:
            indices = [i for i, a in enumerate(state["actions"]) if a["kind"] != "ask"]
            import daguai_rl
            values = daguai_rl.score_many(self.model, [
                (state["observation"], [state["actions"][i]["features"][:71] for i in indices])
            ])[0]
            return indices, values
        if self.version == 3:
            import team_ppo as p
            request = p._cached_request({"state": state, "events": events, "episode_id": 0}, strategic.SIZES)
            if self.cache_enabled:
                if self.strategic_cache is None:
                    self.strategic_cache = p.SeatHistoryCache(self.model)
                # One environment per tournament; the cache key is the seat.
                request.stream_id = 0
                scores, _ = self.strategic_cache.policy_value([request], 0)
                self.gru_steps = self.strategic_cache.tokens
            else:
                self.gru_steps += len(events)
                scores, _ = self.model.policy_value(*p._tensor_states(self.model, [
                    (request.observation, request.actions, f.relative_history(events, state["player"]))]))
            return list(range(len(state["actions"]))), scores[0].cpu().numpy()
        features = [action["features"][:f.ACTION_SIZE] for action in state["actions"]]
        observable = state["observation"][:f.OBSERVATION_SIZE]
        player = state["player"]
        if not self.cache_enabled:
            self.gru_steps += len(events)
            return list(range(len(features))), f.score_many(self.model, [
                (observable, features, f.relative_history(events, player))
            ])[0]
        device = next(self.model.parameters()).device
        offset, hidden = self.hidden.get(player, (0, None))
        if offset > len(events):
            raise ValueError("public history moved backwards without a cache reset")
        new_events = f.relative_history(events[offset:], player)
        if len(new_events):
            tensor = torch.from_numpy(new_events).unsqueeze(0).to(device)
            _, hidden = self.model.history(tensor, hidden)
            self.gru_steps += len(new_events)
        self.hidden[player] = (len(events), hidden)
        context = torch.zeros((1, 128), device=device) if hidden is None else hidden[-1]
        observation = torch.tensor([observable], dtype=torch.float32, device=device)
        base = self.model.base(observation)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        encoded = torch.cat((base, context), dim=1).expand(len(features), -1)
        values = self.model.head(torch.cat((encoded, features_tensor), dim=1)).squeeze(-1)
        if not torch.isfinite(values).all():
            raise RuntimeError("cached policy produced nonfinite scores")
        return list(range(len(features))), values.cpu().numpy()

    def choose(self, state, events, rng):
        if self.checkpoint == "random":
            return rng.randrange(len(state["actions"]))
        indices, scores = self.scores(state, events)
        top = int(np.argmax(scores))
        if self.mode == "greedy":
            return indices[top]
        if self.mode == "epsilon":
            return indices[f.choose_action(scores, self.epsilon, rng)]
        probabilities = softmax(scores, self.temperature)
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        return indices[int(np.searchsorted(cumulative, rng.random(), side="right"))]


def softmax(scores, temperature):
    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("invalid categorical inputs")
    logits = (values - values.max()) / temperature
    weights = np.exp(logits)
    return weights / weights.sum()


def controlled_seats(team, model_seats, seed):
    if model_seats not in (1, 2, 3):
        raise ValueError("model seats must be 1, 2, or 3")
    return [team + 2 * ((seed % 3 + offset) % 3) for offset in range(model_seats)]


def terminal_category(summary, team):
    if summary["winner"] is not None:
        return "WIN" if summary["winner"] == team else "LOSS"
    return "DRAW_WE_COMPLETED" if summary["completedTeam"] == team else "DRAW_THEY_COMPLETED"


def episode(env, candidate, opponent, team, model_seats, state, deal_seed, strategic_goals=False):
    candidate.reset()
    opponent.reset()
    candidate_seats = controlled_seats(team, model_seats, deal_seed)
    controller = {seat: candidate for seat in candidate_seats}
    controller.update({seat: opponent for seat in range(6) if seat % 2 != team})
    neural_seats = [seat for seat, policy in controller.items() if policy.checkpoint != "heuristic"]
    events = list(state["events"])
    rngs = {seat: random.Random(deal_seed * 97 + seat * 193 + team * 389) for seat in range(6)}
    latencies = []
    goals = Counter()
    state = env.state("advance", controlled=neural_seats)
    events.extend(state["events"])
    while not state["terminal"]:
        policy = controller[state["player"]]
        started = time.perf_counter()
        if strategic_goals and policy is candidate:
            indices, scores = policy.scores(state, events)
            if indices != list(range(len(state["actions"]))):
                raise ValueError("strategic goal selection requires scores for the full legal set")
            choice = env.request("strategic_choice", scores=scores.tolist())
            index = choice["index"]
            goals[choice["goal"]] += 1
            goals["changedAction"] += int(choice.get("changedAction", index != int(np.argmax(scores))))
            goals["acceptedWorlds"] += choice["samples"]
            goals["attemptedWorlds"] += choice["attempts"]
        else:
            index = policy.choose(state, events, rngs[state["player"]])
        latencies.append((time.perf_counter() - started) * 1000)
        state = env.state("agent_step", index=index, controlled=neural_seats)
        events.extend(state["events"])
        if len(events) > f.MAX_HISTORY:
            raise RuntimeError("evaluation history overflow")
    summary = env.request("summary")
    category = terminal_category(summary, team)
    score = 1.0 if category == "WIN" else 0.0 if category == "LOSS" else 0.5
    seats = [team, team + 2, team + 4]
    own = [row for row in summary["diagnostics"] if row["actor"] in seats]
    return {
        "score": score, "outcome": category, "learnerSeats": candidate_seats,
        "headFinisher": int(summary["headTeam"] == team),
        "teamCompleted": int(summary["completedTeam"] == team),
        "finishPosition": float(np.mean([summary["order"].index(p) + 1 for p in seats])),
        "cardsLeft": float(np.mean([summary["remainingCounts"][p] for p in seats])),
        "jokersUsed": sum(row["jokerCount"] for row in own),
        "allyOverrides": sum(row["allyOverride"] for row in own),
        "receivesLeavingLegalRemainder": sum(row["receiveWithLegalRemainder"] for row in own),
        "tricksHeld": sum(row["trickWinner"] in seats for row in summary["diagnostics"]),
        "returnDecisions": sum(row["kind"] == "return" for row in own),
        "tributeCount": summary["tributeCount"],
        "latencyMs": latencies, "actions": summary["moves"],
        **({"strategicGoalCounts": dict(goals)} if strategic_goals else {}),
    }


def evaluate(candidate_path="heuristic", opponent_path="heuristic", games=200,
             seed=1_100_000_000, model_seats=3, mode="greedy", temperature=0.2,
             epsilon=0.15, rounds="fresh", device="cpu", cache=True, env_factory=None,
             progress_callback=None, strategic_goals=False):
    f.validate_evaluation(games, seed)
    if rounds not in ("fresh", "continued"):
        raise ValueError("invalid round setting")
    candidate = Policy(candidate_path, mode, temperature, epsilon, device, cache)
    opponent = Policy(opponent_path, device=device, cache=cache)
    if strategic_goals and (candidate.version not in (2, 3) or model_seats != 3 or mode != "greedy"):
        raise ValueError("goal-planner ablation requires an all-neural greedy candidate team")
    evolved = strategic_goals or 3 in (candidate.version, opponent.version)
    bridge = "goal-env.js" if strategic_goals else "strategic-env.js" if evolved else "audit-env.js"
    env = (env_factory or f.NodeEnvironment)(
        script=f.ROOT / bridge, max_actions=6000,
        **({"sizes": strategic.SIZES} if evolved else {}))
    pairs, totals, latencies = [], Counter(), []
    try:
        for pair_index in range(games // 2):
            deal_seed = seed + pair_index
            continuation = None
            if rounds == "continued":
                env.state("reset", seed=deal_seed)
                env.state("advance", controlled=[])
                env.state("reset", seed=deal_seed + 500_000, **{"continue": True})
                continuation = env.request("snapshot")["snapshot"]
            matches = []
            for team in (0, 1):
                state = (env.state("load_snapshot", snapshot=continuation) if continuation is not None
                         else env.state("reset", seed=deal_seed))
                result = episode(env, candidate, opponent, team, model_seats, state, deal_seed,
                                 **({"strategic_goals": True} if strategic_goals else {}))
                totals[result["outcome"]] += 1
                latencies.extend(result.pop("latencyMs"))
                matches.append(result)
            pairs.append({"seed": deal_seed, "scores": [m["score"] for m in matches],
                          "learnerSeats": [m["learnerSeats"] for m in matches], "matches": matches})
            if progress_callback:
                progress_callback({"completedGames": 2 * len(pairs), "totalGames": games,
                                   "completedPairs": len(pairs)})
    finally:
        env.close()
    for policy in (candidate, opponent):
        if policy.model is not None and f.sha256_file(policy.checkpoint) != policy.digest:
            raise RuntimeError("evaluated checkpoint changed; use an immutable frozen copy")
    protocol = {
        "games": games, "seed": seed, "modelSeats": model_seats, "rounds": rounds,
        "opponent": opponent.digest, "rules": f.rules_fingerprint(),
        "sideMapping": "same seeded world, candidate controls parity0 then parity1",
        "reward": "terminal-team-v1", "schemaVersion": 1,
    }
    score = float(np.mean([np.mean(pair["scores"]) for pair in pairs]))
    metrics = {}
    for name in ("headFinisher", "teamCompleted", "finishPosition", "cardsLeft", "jokersUsed",
                 "allyOverrides", "receivesLeavingLegalRemainder", "tricksHeld", "returnDecisions"):
        metrics[name] = float(np.mean([m[name] for p in pairs for m in p["matches"]]))
    extra = {}
    if strategic_goals:
        counts = Counter()
        for pair in pairs:
            for match in pair["matches"]:
                counts.update(match["strategicGoalCounts"])
        names = ("training/goal-env.js", "strategic-goals.js", "strategic-beliefs.js",
                 "training/strategic-env.js", "strategic-policy.js", "training/manifests/strategic-v3.json")
        extra["strategicGoals"] = {
            "enabled": True, "counts": dict(counts), "classification": "bounded rule-based tactical ablation",
            "sources": {name: f.sha256_file(f.ROOT.parent / name) for name in names},
        }
    return {
        "schemaVersion": 1, "createdAt": f.now(), "checkpointSha256": candidate.digest,
        "policy": {"sampling": mode, "temperature": temperature, "epsilon": epsilon},
        "opponentSha256": opponent.digest, "metadata": candidate.data["metadata"] if candidate.model else {},
        "protocol": protocol, "protocolFingerprint": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "games": games, "seed": seed, "wins": totals["WIN"], "losses": totals["LOSS"],
        "draws": totals["DRAW_WE_COMPLETED"] + totals["DRAW_THEY_COMPLETED"],
        "terminalDistribution": {name: totals[name] for name in
                                 ("WIN", "LOSS", "DRAW_WE_COMPLETED", "DRAW_THEY_COMPLETED")},
        "score": score, "rawWinRate": totals["WIN"] / games,
        "drawRate": (totals["DRAW_WE_COMPLETED"] + totals["DRAW_THEY_COMPLETED"]) / games,
        "confidence95": f.paired_bootstrap([np.mean(p["scores"]) for p in pairs], seed ^ 0x9E3779B9),
        "pairedSeeds": pairs, "metrics": metrics,
        "latencyMs": {f"p{q}": float(np.percentile(latencies, q)) if latencies else None for q in (50, 95, 99)},
        "rankIncludesTrapOrdering": True, "promotion": "never automatic",
        "scope": "continued-round outcomes" if rounds == "continued" else "fresh rounds only",
        **extra,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="heuristic")
    parser.add_argument("--opponent", default="heuristic")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1_100_000_000)
    parser.add_argument("--model-seats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--sampling", choices=("greedy", "epsilon", "categorical"), default="greedy")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--rounds", choices=("fresh", "continued"), default="fresh")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--split", choices=tuple(seeds.SPLITS), default="validation")
    parser.add_argument("--release-plan", type=Path)
    parser.add_argument("--strategic-goals", action="store_true",
                        help="enable experimental belief/goal selection for the full candidate team only")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists; use a new immutable report path")
    if args.strategic_goals and (args.split == "final" or args.model_seats != 3 or args.sampling != "greedy"):
        parser.error("experimental strategy requires three greedy model seats and non-final seeds")
    torch.set_num_threads(1)
    checkpoint = args.checkpoint if args.checkpoint in ("heuristic", "random") else str(Path(args.checkpoint).resolve())
    digest = checkpoint if checkpoint in ("heuristic", "random") else f.sha256_file(checkpoint)
    if args.split == "final":
        if not args.release_plan:
            parser.error("sealed final split requires a preselected release plan")
        opponent_digest = (args.opponent if args.opponent in ("heuristic", "random")
                           else f.sha256_file(args.opponent))
        seeds.validate_final_protocol(
            args.release_plan, {"sampling": args.sampling, "temperature": args.temperature, "epsilon": args.epsilon},
            args.model_seats, args.rounds, opponent_digest)
    identifier = seeds.record_evaluation(args.split, args.seed, args.games, digest,
                                        "tournament-v1:strategic-goals" if args.strategic_goals else "tournament-v1",
                                        release_plan=args.release_plan)
    try:
        result = evaluate(checkpoint, args.opponent, args.games, args.seed, args.model_seats,
                          args.sampling, args.temperature, args.epsilon, args.rounds, args.device,
                          strategic_goals=args.strategic_goals)
        result["evaluationSplit"] = args.split
        f.atomic_json_save(result, args.report)
    except BaseException:
        seeds.finish_evaluation(identifier, "failed")
        raise
    seeds.finish_evaluation(identifier, "completed", args.report.resolve())
    print(f"score={result['score']:.4f} W/L/D={result['wins']}/{result['losses']}/{result['draws']} "
          f"paired95%CI={result['confidence95']} report={args.report.resolve()}")


if __name__ == "__main__":
    main()
