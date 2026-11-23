"""Shared recurrent, private-observation team PPO; no search or strength claims.

Initialization may be random, exact v2 actor transfer, or verified source-policy
distillation. PPO uses complete on-policy rounds, per-seat GAE and terminal team
outcomes. Public finisher potential preserves the terminal objective.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import foundation as f
import strategic_protocol as strategic
from contracts import (
    CONTRACT_VERSION, REWARD_VERSION, SHAPING_VERSION, Decision, Episode, RolloutBatch,
    action_identity, finalize_episode, legal_support_digest, model_digest,
)

ALGORITHM = "team-ppo-v1"
CHECKPOINTS = f.ROOT / "ppo" / "checkpoints"
MAX_ROLLOUT_BYTES = 512 * 1024 * 1024
METHODS = {
    "algorithm": ALGORITHM,
    "initialization": "exact actor transfer from self-trained v2 weights; zero final value layer",
    "policy": "shared recurrent policy replicas, separate action-independent value head",
    "objective": "clipped on-policy PPO with per-seat GAE and shared terminal TEAM reward",
    "reward": "terminal +1/-1/0; public team finish-count potential differences, terminal Phi=0",
    "shaping": "potential-based, not card-shedding/action/individual-finishing rewards",
    "information": "own observation and acting-seat-relative public history only, actor AND critic",
    "teamPlanning": "implicit recurrent context; no private-card pooling, explicit role labels or calibrated beliefs",
    "exploration": "stochastic legal-action categorical with temperature, not v2 epsilon-greedy",
    "opponents": "60% selfplay, 20% uniform frozen history, 20% rules; optional one rule teammate",
    "replay": "none; complete bounded batches consumed once; frozen behavior throughout collection",
    "strength": "training loss is not evidence of improvement; no automatic promotion",
}
CORE_DEFAULTS = {
    "workers": 4, "rollout_games": 8, "epochs": 4, "batch_size": 64,
    "seed": 1, "gamma": 1.0, "gae_lambda": .95, "clip": .2,
    "entropy_coef": .01, "potential_scale": .3, "learning_rate": 1e-4,
    "temperature": .2, "mixed_teammate_probability": .25, "max_actions": 6000,
    "target_kl": .015,
}


def configure_precision(device):
    if torch.device(device).type == "cuda":
        # TF32 makes incremental and full-prefix GRU execution disagree.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


class ActorCritic(f.ActionValue):
    """Inherited .base/.history/.head retain raw-logit foundation inference."""

    sizes = f.SIZES

    def __init__(self):
        super().__init__()
        self.value_head = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def policy_value(self, observations, actions, histories):
        configure_precision(observations.device)
        if (observations.ndim != 2 or observations.shape[1] != self.sizes["observationSize"]
                or len(actions) != len(observations) or len(histories) != len(observations)):
            raise ValueError("unaligned policy/value state batch")
        if not len(observations) or any(
                a.ndim != 2 or a.shape[1] != self.sizes["actionSize"] or not len(a) for a in actions):
            raise ValueError("each state requires nonempty legal action features")
        return self.score_with_history(observations, actions, self.encode_history(histories))

    def score_with_history(self, observations, actions, hidden):
        context = torch.cat((self.base(observations), hidden), dim=1)
        counts = [len(a) for a in actions]
        repeated = torch.repeat_interleave(
            context, torch.tensor(counts, device=context.device), dim=0)
        logits = self.head(torch.cat((repeated, torch.cat(actions)), dim=1)).squeeze(-1)
        values = self.value_head(context).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise RuntimeError("non-finite policy/value output")
        return logits.split(counts), values


class StrategicActorCritic(ActorCritic):
    """Residual v3 experiment; zero residuals preserve both v2 outputs at transfer."""

    sizes = strategic.SIZES

    def __init__(self):
        super().__init__()
        self.rank_embedding = nn.Embedding(15, 16)
        self.suit_embedding = nn.Embedding(5, 8)
        self.card_projection = nn.Linear(30, 64)
        self.hand_set = nn.Sequential(*[
            nn.TransformerEncoderLayer(64, 4, 128, dropout=0, batch_first=True, norm_first=True)
            for _ in range(2)])
        self.hand_projection = nn.Linear(64, 128)
        self.private_projection = nn.Linear(62, 64)
        self.private_history = nn.GRU(64, 64, batch_first=True)
        self.context_projection = nn.Sequential(nn.LayerNorm(584), nn.SiLU(), nn.Linear(584, 384))
        self.actor_residual = nn.Sequential(nn.Linear(489, 256), nn.SiLU(),
                                           nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.critic_residual = nn.Sequential(nn.Linear(384, 256), nn.SiLU(),
                                            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        for branch in (self.actor_residual, self.critic_residual):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        self.register_buffer("card_ranks", torch.tensor([i % 13 for i in range(52)] + [13, 14]),
                             persistent=False)
        self.register_buffer("card_suits", torch.tensor([i // 13 for i in range(52)] + [4, 4]),
                             persistent=False)

    def strategic_context(self, observations, legacy):
        counts = observations[:, :54]
        ranks, suits = self.card_ranks, self.card_suits
        ordinary = ranks < 13
        trump = observations[:, 396:409].index_select(1, ranks.clamp(max=12)) * ordinary
        rank_features = torch.where(ordinary, ranks + 2, ranks + 7) / 21
        strength = torch.where(trump.bool(), 18 / 21, rank_features)
        shape = counts.shape
        features = torch.stack((counts, (counts > 0).float(), trump, (ranks == 13).expand(shape),
                                (ranks == 14).expand(shape), strength), dim=-1)
        identities = torch.cat((self.rank_embedding(ranks), self.suit_embedding(suits)), dim=1)
        cards = self.card_projection(torch.cat((features, identities.expand(len(counts), -1, -1)), dim=-1))
        cards = self.hand_projection(self.hand_set(cards).mean(dim=1))
        extra = observations[:, f.OBSERVATION_SIZE:]
        transfers = extra[:, 8:].reshape(-1, 2, 62)
        present = transfers[:, :, -2:].sum(dim=-1)
        if bool((present[:, 1] > present[:, 0]).any()):
            raise ValueError("private return requires its preceding observed tribute")
        codes, _ = self.private_history(self.private_projection(transfers))
        rows = torch.arange(len(counts), device=observations.device)
        private = codes[rows, present[:, 1].long()] * present[:, :1]
        return self.context_projection(torch.cat((legacy, private, cards, extra[:, :8]), dim=1))

    def score_with_history(self, observations, actions, hidden):
        legacy = torch.cat((self.base(observations[:, :f.OBSERVATION_SIZE]), hidden), dim=1)
        context = self.strategic_context(observations, legacy)
        counts = [len(a) for a in actions]
        repeats = torch.tensor(counts, device=observations.device)
        candidates = torch.cat(actions)
        old = self.head(torch.cat((legacy.repeat_interleave(repeats, dim=0),
                                   candidates[:, :f.ACTION_SIZE]), dim=1)).squeeze(-1)
        residual = self.actor_residual(torch.cat((context.repeat_interleave(repeats, dim=0),
                                                  candidates), dim=1)).squeeze(-1)
        logits = old + residual
        values = (self.value_head(legacy) + self.critic_residual(context)).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise RuntimeError("non-finite strategic policy/value output")
        return logits.split(counts), values


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def initialization_methods(metadata):
    methods = dict(METHODS)
    if metadata.get("lineage", {}).get("actorTransfer") == "source-behavior-distilled":
        methods["initialization"] = "source-policy behavior distillation; source base/GRU retained; zero final value layer"
    elif metadata.get("lineage", {}).get("actorTransfer") == "random":
        methods["initialization"] = "random actor from scratch; zero final value layer; no v2 transfer"
    if "ablationParent" in metadata:
        methods["initialization"] = "exact PPO parent scores and values; new optimizer and game counter"
    if metadata.get("encodingVersion") == strategic.SIZES["encodingVersion"]:
        methods["information"] = "own hand, own transfer memory, and public context only"
        methods["teamPlanning"] = "structural opportunities and recurrent context; no fixed roles or calibrated beliefs"
    return methods


def checkpoint_data(model, metadata, optimizer=None, rng=None):
    sizes = getattr(model, "sizes", f.SIZES)
    data = {
        "metadata": {**copy.deepcopy(metadata), **sizes, "algorithm": ALGORITHM,
                     **({"strategicFeatureSha256": strategic.fingerprint()} if sizes == strategic.SIZES else {}),
                     "learnReturns": True, "methods": initialization_methods(metadata),
                     "decisionContractVersion": CONTRACT_VERSION,
                     "rewardVersion": REWARD_VERSION, "shapingVersion": SHAPING_VERSION,
                     "rulesFingerprint": f.rules_fingerprint(),
                     "coreSourceSha256": hashlib.sha256(
                         Path(__file__).read_bytes()
                         + Path(__file__).with_name("contracts.py").read_bytes()
                         + Path(__file__).with_name("trace_audit.py").read_bytes()).hexdigest()},
        "model": _cpu_copy(model.state_dict()),
    }
    if optimizer is not None:
        if rng is None:
            raise ValueError("training checkpoints require RNG state")
        data.update(optimizer=_cpu_copy(optimizer.state_dict()), rng=f.rng_state(rng))
    return data


def load_checkpoint(path, device="cpu"):
    configure_precision(device)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be an object")
    metadata = data.get("metadata")
    v3 = isinstance(metadata, dict) and metadata.get("encodingVersion") == strategic.SIZES["encodingVersion"]
    f.check_encoding(metadata, strategic.SIZES if v3 else f.SIZES)
    if v3 and metadata.get("strategicFeatureSha256") != strategic.fingerprint():
        raise ValueError("incompatible strategic feature manifest/producer")
    if metadata.get("algorithm") != ALGORITHM or metadata.get("learnReturns") is not True:
        raise ValueError("not a team-ppo-v1 checkpoint")
    for key in ("games", "ppoGames", "updates", "sourceGames", "sourceUpdates"):
        if type(metadata.get(key)) is not int or metadata[key] < 0:
            raise ValueError(f"invalid checkpoint {key}")
    if metadata["games"] != metadata["sourceGames"] + metadata["ppoGames"]:
        raise ValueError("cumulative games must equal sourceGames + ppoGames")
    lineage = metadata.get("lineage")
    transfer = lineage.get("actorTransfer") if isinstance(lineage, dict) else None
    initialization = {
        "exact": "transferred self-trained v2 actor weights",
        "source-behavior-distilled": "source-behavior-distilled v2 actor",
        "random": "random actor from scratch",
    }.get(transfer)
    if (not isinstance(lineage, dict) or lineage.get("sourceGames") != metadata["sourceGames"]
            or lineage.get("sourceSha256") != metadata.get("sourceSha256")
            or initialization is None or metadata.get("initialization") != initialization):
        raise ValueError("invalid initialization lineage")
    if transfer == "random":
        if (metadata["sourceGames"] != 0 or metadata["sourceUpdates"] != 0
                or metadata.get("sourceSha256") is not None or metadata.get("sourceCheckpoint") is not None
                or lineage.get("sourceAlgorithm") is not None):
            raise ValueError("random initialization cannot claim inherited weights or source games")
    elif (not isinstance(metadata.get("sourceSha256"), str)
          or len(metadata["sourceSha256"]) != 64
          or any(c not in "0123456789abcdef" for c in metadata["sourceSha256"])
          or lineage.get("sourceAlgorithm") != f.METHODS["algorithm"]):
        raise ValueError("invalid transferred-weight lineage")
    if transfer == "source-behavior-distilled":
        distillation = metadata.get("behaviorDistillation", {})
        confirmation = distillation.get("confirmation", {})
        if (distillation.get("sourceSha256") != metadata["sourceSha256"]
                or distillation.get("behaviorPreserved") is not True
                or confirmation.get("passed") is not True
                or not 0 <= confirmation.get("meanKL", float("inf")) < .05
                or not .98 < confirmation.get("topAgreement", 0) <= 1
                or distillation.get("temperature") != metadata.get("config", {}).get("temperature")):
            raise ValueError("distilled checkpoint lacks independently verified behavior retention")
    if metadata.get("methods", {}).get("algorithm") != ALGORITHM:
        raise ValueError("checkpoint methods algorithm mismatch")
    for key, expected in (("decisionContractVersion", CONTRACT_VERSION),
                          ("rewardVersion", REWARD_VERSION), ("shapingVersion", SHAPING_VERSION)):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"incompatible checkpoint {key}")
    with torch.random.fork_rng(devices=[]):
        model = (StrategicActorCritic() if v3 else ActorCritic()).to(device)
    model.load_state_dict(data["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("non-finite checkpoint parameters")
    model.eval()
    model.metadata = metadata
    model.checkpoint = str(Path(path).resolve())
    return model, data


def warm_start(path, device="cpu"):
    """Copy the exact frozen source inode; never load MC replay into PPO."""
    with Path(path).open("rb") as stream:
        import hashlib
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        data = torch.load(stream, map_location="cpu", weights_only=True)
    source = data.get("metadata", {})
    f.check_encoding(source)
    if (source.get("algorithm") not in (None, "foundation-v2")
            or source.get("methods", {}).get("algorithm") != f.METHODS["algorithm"]):
        raise ValueError("--warm-start requires v2 action-value weights, not PPO/v1")
    if source.get("learnReturns") is not True:
        raise ValueError("source must learn returns")
    for key in ("games", "updates"):
        if type(source.get(key)) is not int or source[key] < 0:
            raise ValueError("invalid source training counts")
    with torch.random.fork_rng(devices=[]):
        actor = f.ActionValue()
        model = ActorCritic().to(device)
    actor.load_state_dict(data["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("non-finite source parameters")
    model.load_state_dict({**model.state_dict(), **actor.state_dict()}, strict=True)
    sha = digest.hexdigest()
    metadata = {
        "games": source["games"], "sourceGames": source["games"], "ppoGames": 0, "updates": 0,
        "sourceSha256": sha, "sourceCheckpoint": str(Path(path).resolve()),
        "sourceUpdates": source["updates"],
        "lineage": {"sourceSha256": sha, "sourceGames": source["games"],
                    "sourceAlgorithm": f.METHODS["algorithm"], "actorTransfer": "exact"},
        "initialization": "transferred self-trained v2 actor weights",
    }
    model.eval()
    model.metadata = {**metadata, **f.SIZES, "algorithm": ALGORITHM, "methods": METHODS,
                      "learnReturns": True}
    return model, metadata


def restore_training(data, optimizer, rng, device):
    if "replay" in data or any(key not in data for key in ("optimizer", "rng")):
        raise ValueError("PPO resume requires optimizer/RNG and forbids stale replay")
    optimizer.load_state_dict(data["optimizer"])
    if any(not math.isfinite(group["lr"]) or group["lr"] <= 0 for group in optimizer.param_groups):
        raise ValueError("non-finite or nonpositive optimizer learning rate")
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError("non-finite optimizer state")
                if key != "step":
                    state[key] = value.to(device)
    state = data["rng"]
    rng.setstate(state["local"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    array = state["numpy"]
    np.random.set_state((array[0], array[1].cpu().numpy().astype(np.uint32), *array[2:]))
    if device.type == "cuda" and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device-count mismatch")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def potential(observation, scale=.3):
    finished = np.asarray(observation)[390:396] > 0
    return float(scale * (int(finished[::2].sum()) - int(finished[1::2].sum())) / 3)


def seat_gae(episode, gamma=1.0, gae_lambda=.95, potential_scale=.3):
    """Each seat's own decision clock; teammate/enemy turns never bootstrap it."""
    if gamma != 1.0:
        raise ValueError("gamma must equal 1 until elapsed-time discounting is supported")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("invalid GAE lambda")
    advantages = np.zeros(len(episode.decisions), np.float32)
    returns = np.zeros_like(advantages)
    for player in range(6):
        indices = [i for i, d in enumerate(episode.decisions) if d.player == player]
        carry = 0.0
        for offset in reversed(range(len(indices))):
            i = indices[offset]
            decision = episode.decisions[i]
            last = offset == len(indices) - 1
            following = None if last else episode.decisions[indices[offset + 1]]
            next_phi = 0.0 if last else potential(following.observation, potential_scale)
            reward = f.team_reward(player, episode.winner) if last else 0.0
            reward += gamma * next_phi - potential(decision.observation, potential_scale)
            next_value = 0.0 if last else following.old_value
            delta = reward + gamma * next_value - decision.old_value
            carry = delta + (0.0 if last else gamma * gae_lambda * carry)
            advantages[i] = carry
            returns[i] = carry + decision.old_value
    return advantages, returns


def clipped_policy_loss(new_logprob, old_logprob, advantages, clip=.2):
    ratio = (new_logprob - old_logprob.detach()).exp()
    objective = torch.minimum(
        ratio * advantages.detach(), ratio.clamp(1 - clip, 1 + clip) * advantages.detach())
    return -objective.mean(), ratio


def normalize_advantages(values):
    spread = values.std(unbiased=False)
    return (values - values.mean()) / spread if spread > 1e-8 else values - values.mean()


def _tensor_states(model, states):
    device = next(model.parameters()).device
    observations = torch.as_tensor(np.stack([s[0] for s in states]), device=device, dtype=torch.float32)
    actions = [torch.as_tensor(s[1], device=device, dtype=torch.float32) for s in states]
    histories = [torch.as_tensor(s[2], device=device, dtype=torch.float32) for s in states]
    return observations, actions, histories


@dataclass
class CachedState:
    """Public absolute events plus the acting seat's own observation/legal set."""
    observation: np.ndarray
    actions: np.ndarray
    events: object
    player: int
    stream_id: object
    round_id: object


class SeatHistoryCache:
    """Inference-only incremental GRU, separate hidden state for each seat/view.

    Bind to ONE model. Call with a changed behavior_version whenever its weights
    change (or reset explicitly). Full histories remain in Episode for gradients.
    """

    def __init__(self, model):
        self.model = model
        self.sizes = getattr(model, "sizes", f.SIZES)
        configure_precision(next(model.parameters()).device)
        self.reset()
        self.tokens = 0
        self.full_prefix_tokens = 0
        self.decisions = 0

    def reset(self):
        self.entries = {}
        self.version = None

    @torch.no_grad()
    def policy_value(self, requests, behavior_version):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

        if self.model.training:
            raise ValueError("history cache is inference-only; model must be eval")
        if self.version != behavior_version:
            self.reset()
            self.version = behavior_version
        if not requests:
            return (), torch.empty(0, device=next(self.model.parameters()).device)
        keys = [(s.stream_id, s.player) for s in requests]
        if len(set(keys)) != len(keys):
            raise ValueError("one ready request per worker/seat per cache batch")
        device = next(self.model.parameters()).device
        hidden = torch.zeros((len(requests), 128), device=device)
        windows, active, initials = [], [], []
        for i, (s, key) in enumerate(zip(requests, keys)):
            previous = self.entries.get(key)
            if previous is None or previous[0] != s.round_id:
                prefix, code = 0, torch.zeros(128, device=device)
            else:
                _, prefix, code = previous
            if len(s.events) < prefix:
                raise ValueError("public history shrank without a round reset")
            new = f.relative_history(s.events[prefix:], s.player)
            self.tokens += len(new)
            self.full_prefix_tokens += len(s.events)
            self.decisions += 1
            hidden[i] = code
            if len(new):
                windows.append(torch.as_tensor(new, dtype=torch.float32, device=device))
                active.append(i)
                initials.append(code)
        if active:
            packed = pack_padded_sequence(
                pad_sequence(windows, batch_first=True), [len(w) for w in windows],
                batch_first=True, enforce_sorted=False)
            _, code = self.model.history(packed, torch.stack(initials).unsqueeze(0))
            hidden.index_copy_(0, torch.tensor(active, device=device), code[0])
        observations = torch.as_tensor(
            np.stack([s.observation for s in requests]), dtype=torch.float32, device=device)
        actions = [torch.as_tensor(s.actions, dtype=torch.float32, device=device) for s in requests]
        if (observations.shape != (len(requests), self.sizes["observationSize"])
                or any(a.ndim != 2 or a.shape[1] != self.sizes["actionSize"] or not len(a) for a in actions)):
            raise ValueError("invalid cached observation/legal features")
        if hasattr(self.model, "score_with_history"):
            result = self.model.score_with_history(observations, actions, hidden)
            for i, (s, key) in enumerate(zip(requests, keys)):
                self.entries[key] = (s.round_id, len(s.events), hidden[i].detach().clone())
            return result
        context = torch.cat((self.model.base(observations), hidden), dim=1)
        counts = [len(a) for a in actions]
        repeated = torch.repeat_interleave(context, torch.tensor(counts, device=device), dim=0)
        logits = self.model.head(torch.cat((repeated, torch.cat(actions)), dim=1)).squeeze(-1)
        values = (self.model.value_head(context).squeeze(-1) if hasattr(self.model, "value_head")
                  else torch.zeros(len(requests), device=device))
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise RuntimeError("nonfinite cached policy/value output")
        for i, (s, key) in enumerate(zip(requests, keys)):
            self.entries[key] = (s.round_id, len(s.events), hidden[i].detach().clone())
        return logits.split(counts), values


def _cached_request(worker, sizes=f.SIZES):
    state = worker["state"]
    return CachedState(
        np.asarray(state["observation"][:sizes["observationSize"]], np.float32),
        f.matrix([a["features"][:sizes["actionSize"]] for a in state["actions"]],
                 sizes["actionSize"], "actions", None),
        worker["events"], state["player"], id(worker), worker["episode_id"])


def _sample_logits(logits, values, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    selected = []
    for score, value in zip(logits, values):
        distribution = Categorical(logits=score / temperature)
        index = distribution.sample()
        selected.append((int(index), float(distribution.log_prob(index)), float(value)))
    return selected


@torch.no_grad()
def sample_states(model, states, temperature=.2):
    logits, values = model.policy_value(*_tensor_states(model, states))
    return _sample_logits(logits, values, temperature)


def update_batch(model, optimizer, batch, version, config, rng):
    from trace_audit import audit_batch, replay_predictions

    started = time.monotonic()
    _validate_config(config)
    batch.validate(version)
    prior_mode = model.training
    model.train()
    try:
        audit = audit_batch(batch, model, config)
    finally:
        model.train(prior_mode)
    entries, advantages, returns = [], [], []
    for episode in batch.episodes:
        a, r = seat_gae(episode, config["gamma"], config["gae_lambda"], config["potential_scale"])
        entries.extend((episode, decision) for decision in episode.decisions)
        advantages.extend(a)
        returns.extend(r)
    if not entries:
        batch.consumed = True
        return {**batch.diagnostics, "optimizerSteps": 0, "transitions": 0, "epochsRun": 0,
                "stopReason": "no_learner_decisions", "traceAudit": audit,
                "updateWallSeconds": time.monotonic() - started}
    device = next(model.parameters()).device
    raw_advantage = torch.tensor(np.asarray(advantages), device=device, dtype=torch.float32)
    advantage = normalize_advantages(raw_advantage)  # ONCE for the entire completed batch.
    targets = torch.tensor(np.asarray(returns), device=device, dtype=torch.float32)
    old_lp = torch.tensor([d.old_logprob for _, d in entries], device=device, dtype=torch.float32)
    old_values = torch.tensor([d.old_value for _, d in entries], device=device, dtype=torch.float32)
    batch.consumed = True  # Failed/partial updates cannot silently reuse this rollout.
    model.train()  # CuDNN recurrent backward requires a training-mode forward.
    totals = Counter()
    order = list(range(len(entries)))
    epochs_run = 0
    stop_reason = "epochs_completed"
    degenerate = 0
    samples = 0
    current = None
    try:
        for _ in range(config["epochs"]):
            rng.shuffle(order)
            for start in range(0, len(order), config["batch_size"]):
                chosen = order[start:start + config["batch_size"]]
                states = [(entries[i][1].observation, entries[i][1].actions,
                           entries[i][0].history(entries[i][1])) for i in chosen]
                logits, values = model.policy_value(*_tensor_states(model, states))
                distributions = [Categorical(logits=score / config["temperature"]) for score in logits]
                lp = torch.stack([dist.log_prob(torch.tensor(entries[i][1].index, device=device))
                                  for dist, i in zip(distributions, chosen)])
                entropy = torch.stack([dist.entropy() for dist in distributions]).mean()
                ids = torch.tensor(chosen, device=device)
                degenerate += int(float(raw_advantage[ids].std(unbiased=False)) <= 1e-8)
                policy_loss, ratio = clipped_policy_loss(lp, old_lp[ids], advantage[ids], config["clip"])
                clipped_values = old_values[ids] + (values - old_values[ids]).clamp(
                    -config["clip"], config["clip"])
                value_loss = .5 * torch.maximum(
                    (values - targets[ids]).square(), (clipped_values - targets[ids]).square()).mean()
                loss = policy_loss + .5 * value_loss - config["entropy_coef"] * entropy
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise RuntimeError("non-finite updated PPO parameters")
                samples += len(chosen)
                totals.update(optimizerSteps=1, policyLoss=float(policy_loss.detach()) * len(chosen),
                              valueLoss=float(value_loss.detach()) * len(chosen),
                              totalLoss=float(loss.detach()) * len(chosen), gradNorm=float(norm))
            epochs_run += 1
            # Clipping constrains the surrogate, NOT the policy KL. Evaluate the
            # complete batch after each epoch, not a noisy/pre-update minibatch.
            current = replay_predictions(
                model, entries, config["temperature"], config["batch_size"])
            logratio = current["logprob"] - old_lp.detach().cpu().numpy()
            kl = float(np.mean(np.expm1(logratio) - logratio))
            if not math.isfinite(kl):
                raise RuntimeError("nonfinite full-batch approximate KL")
            if kl > config["target_kl"]:
                stop_reason = "target_kl_exceeded"
                break
    finally:
        model.train(prior_mode)
    steps = totals["optimizerSteps"]
    target_array = targets.detach().cpu().numpy()
    raw_array = raw_advantage.detach().cpu().numpy()
    old_array = old_values.detach().cpu().numpy()
    logratio = current["logprob"] - old_lp.detach().cpu().numpy()
    metrics = {key: value if key == "optimizerSteps" else value / (
        steps if key == "gradNorm" else samples) for key, value in totals.items()}

    def stats(values):
        return {"mean": float(np.mean(values)), "std": float(np.std(values)),
                "min": float(np.min(values)), "max": float(np.max(values))}

    legal_counts = np.asarray([len(d.actions) for _, d in entries])
    grouped, modes = {}, Counter()
    for label in sorted({d.kind for _, d in entries}):
        grouped[label] = stats([raw_array[i] for i, (_, d) in enumerate(entries) if d.kind == label])
        grouped[label]["count"] = sum(d.kind == label for _, d in entries)
    for episode in batch.episodes:
        modes[episode.mode] += len(episode.decisions)
    variance = float(np.var(target_array))
    metrics.update(
        transitions=len(entries), epochsRun=epochs_run, stopReason=stop_reason,
        targetKL=config["target_kl"], approxKL=float(np.mean(np.expm1(logratio) - logratio)),
        clipFraction=float(np.mean(np.abs(np.exp(logratio) - 1) > config["clip"])),
        entropy=float(current["entropy"].mean()),
        normalizedEntropy=float(current["normalizedEntropy"].mean()),
        pmax=float(current["pmax"].mean()), effectiveSupport=float(current["effectiveSupport"].mean()),
        legalCountStats={**stats(legal_counts), **{
            f"p{pct}": float(np.percentile(legal_counts, pct)) for pct in (50, 90, 99)}},
        oldValueStats=stats(old_array), currentValueStats=stats(current["value"]),
        targetStats=stats(target_array), advantageStats=stats(raw_array),
        advantageByKind=grouped, learnerDecisionsByMode=dict(modes),
        explainedVariance=(1 - float(np.var(target_array - current["value"])) / variance
                           if variance > 1e-12 else None),
        explainedVarianceUndefinedReason=("zero_target_variance" if variance <= 1e-12 else None),
        degenerateMinibatchFraction=degenerate / steps,
        advantageNormalization="once_per_complete_batch",
        updateWallSeconds=time.monotonic() - started,
        **batch.diagnostics,
    )
    metrics["traceAudit"] = audit
    full_prefix_tokens = sum(d.prefix for _, d in entries)
    metrics["updateGradientGruTokens"] = epochs_run * full_prefix_tokens
    metrics["updateReplayGruTokens"] = (1 + epochs_run) * full_prefix_tokens
    metrics["updateGruTokens"] = (1 + 2 * epochs_run) * full_prefix_tokens
    metrics["totalBatchGruTokens"] = (
        metrics["updateGruTokens"] + batch.diagnostics.get("rolloutGruTokens", 0))
    return metrics


def collect_batch(model, workers, games, rng, config, version, next_episode=0,
                  league=(), progress=None, replay_dir=None, replay_recorders=None):
    """All workers finish before return; no optimizer access or cross-batch replay."""
    if not 1 <= games <= config["rollout_games"] or not workers:
        raise ValueError("invalid bounded rollout size/workers")
    _validate_config(config)
    started = time.monotonic()
    digest_before = model_digest(model)
    cache = SeatHistoryCache(model)
    sizes = cache.sizes
    opponent_caches = {}
    progress = progress if progress is not None else {}
    progress.update(decisions=0, completedEpisodes=0, inFlightEpisodes=0, rolloutBytes=0)
    episodes = []
    coverage = Counter()
    launched = 0
    prior_mode = model.training
    model.eval()
    recorders = replay_recorders if replay_recorders is not None else []
    if replay_dir is not None and replay_recorders is None:
        raise ValueError("replay recording requires a caller-owned publication transaction")

    def launch(worker):
        nonlocal launched
        number = next_episode + launched
        launched += 1
        draw = rng.random()
        mode = "selfplay" if draw < .6 else "historical" if draw < .8 else "heuristic"
        team = None if mode == "selfplay" else rng.randrange(2)
        opponent = rng.choice(league) if mode == "historical" else None
        heuristic = set()
        if team is not None and rng.random() < config["mixed_teammate_probability"]:
            heuristic.add(rng.choice([team, team + 2, team + 4]))
        continuing = worker.get("finished", False)
        state = worker["env"].state("reset", seed=(config["seed"] + number) & 0xFFFFFFFF,
                                    **({"continue": True} if continuing else {}))
        f.validate_state(state, sizes)
        if state["terminal"] or state["events"]:
            raise RuntimeError("reset requires live state and cleared history")
        controlled = tuple(p for p in range(6) if p not in heuristic
                           and (team is None or p % 2 == team))
        worker.update(state=state, events=[], decisions=[], steps=0, finished=False,
                      episode_id=number, controlled_seats=controlled,
                      opponent=opponent, team=team, heuristic=heuristic, mode=mode)
        if replay_dir is not None:
            from game_replay import GameReplay
            worker["replay"] = GameReplay(
                replay_dir, number, (config["seed"] + number) & 0xFFFFFFFF,
                worker["env"].request("snapshot")["snapshot"],
                {"mode": mode, "controlledSeats": list(controlled), "learnerTeam": team,
                 "behaviorVersion": version, "behaviorDigest": digest_before,
                 "continued": continuing, "rules": f.rules_fingerprint()})
            recorders.append(worker["replay"])
        coverage["continuedRounds"] += int(continuing)
        coverage["mixedTeamGames"] += bool(heuristic)
        progress["inFlightEpisodes"] += 1

    def step(worker, op, index=None):
        before = worker["state"]
        action = before["actions"][index] if index is not None else None
        coverage[f"phase:{worker['state']['phase']}"] += 1
        state = worker["env"].state(op, **({"index": index} if index is not None else {}))
        f.validate_state(state, sizes)
        f.append_events(worker, state)
        progress["rolloutBytes"] += len(state["events"]) * f.HISTORY_SIZE * 4
        if progress["rolloutBytes"] > MAX_ROLLOUT_BYTES:
            raise RuntimeError("bounded rollout storage exceeded; discard batch, no stale replay")
        worker.update(state=state, steps=worker["steps"] + 1)
        if replay_dir is not None:
            worker["replay"].step(worker["env"].request("snapshot")["snapshot"],
                                  before["player"], op, action, before["phase"])
        coverage["maxObservedHistoryLength"] = max(
            coverage["maxObservedHistoryLength"], len(worker["events"]))

    try:
        while len(episodes) < games:
            for worker in workers:
                if worker.get("state") is None and launched < games:
                    launch(worker)
            learners = []
            frozen = {}
            for worker in workers:
                state = worker.get("state")
                if state is None or state["terminal"]:
                    continue
                if worker["steps"] >= config["max_actions"]:
                    raise RuntimeError("max-actions reached; no fabricated truncated reward")
                player = state["player"]
                if player not in worker["heuristic"] and (
                        worker["team"] is None or player % 2 == worker["team"]):
                    learners.append(worker)
                elif worker["opponent"] is not None and player not in worker["heuristic"]:
                    frozen.setdefault(worker["opponent"], []).append(worker)
                else:
                    step(worker, "heuristic")
            for opponent, ready in frozen.items():
                if opponent.training:
                    raise ValueError("frozen opponent must be in eval mode")
                opponent_cache = opponent_caches.setdefault(opponent, SeatHistoryCache(opponent))
                scores, _ = opponent_cache.policy_value(
                    [_cached_request(w, opponent_cache.sizes) for w in ready], version)
                for worker, score in zip(ready, scores):
                    step(worker, "step", int(score.argmax()))
            if learners:
                sampled = _sample_logits(
                    *cache.policy_value([_cached_request(w, sizes) for w in learners], version),
                    config["temperature"])
                for worker, (index, logprob, value) in zip(learners, sampled):
                    state = worker["state"]
                    kind = state["actions"][index]["kind"]
                    worker["decisions"].append(Decision(
                        np.array(state["observation"], dtype=np.float32),
                        f.matrix([a["features"] for a in state["actions"]], sizes["actionSize"],
                                 "actions", None).copy(),
                        state["player"], len(worker["events"]), kind, index, logprob, value,
                        contract_version=CONTRACT_VERSION, episode_id=worker["episode_id"],
                        global_step=worker["steps"],
                        action_identities=tuple(action_identity(a) for a in state["actions"]),
                        behavior_digest=digest_before))
                    recorded = worker["decisions"][-1]
                    recorded.legal_support_digest = legal_support_digest(
                        recorded.action_identities, recorded.actions)
                    progress["decisions"] += 1
                    progress["rolloutBytes"] += (
                        worker["decisions"][-1].observation.nbytes
                        + worker["decisions"][-1].actions.nbytes
                        + sum(sys.getsizeof(identity) for identity in
                              worker["decisions"][-1].action_identities))
                    if progress["rolloutBytes"] > MAX_ROLLOUT_BYTES:
                        raise RuntimeError("bounded rollout storage exceeded; discard batch")
                    coverage[f"decision:{kind}"] += 1
                    step(worker, "step", index)
            for worker in workers:
                if worker.get("state") is not None and worker["state"]["terminal"]:
                    episodes.append(finalize_episode(Episode(
                        f.matrix(worker["events"], f.HISTORY_SIZE, "events", f.MAX_HISTORY, empty=True).copy(),
                        worker["decisions"], worker["state"]["result"]["winner"], version,
                        contract_version=CONTRACT_VERSION, episode_id=worker["episode_id"],
                        behavior_digest=digest_before, controlled_seats=worker["controlled_seats"],
                        global_steps=worker["steps"], mode=worker["mode"],
                        potential_scale=config["potential_scale"],
                        temperature=config["temperature"], encoding_version=sizes["encodingVersion"]), potential))
                    coverage[f"mode:{worker['mode']}"] += 1
                    worker.update(state=None, events=[], decisions=[], opponent=None, finished=True)
                    worker.pop("replay", None)
                    progress["completedEpisodes"] += 1
                    progress["inFlightEpisodes"] -= 1
        digest_after = model_digest(model)
        caches = [cache, *opponent_caches.values()]
        diagnostics = {
            "rolloutWallSeconds": time.monotonic() - started,
            "gruTokens": sum(c.tokens for c in caches),
            "rolloutGruTokens": sum(c.tokens for c in caches),
            "gruFullPrefixTokensAvoided": sum(c.full_prefix_tokens - c.tokens for c in caches),
            "gruInferenceDecisions": sum(c.decisions for c in caches),
            "learnerGruTokens": cache.tokens,
            "frozenGruTokens": sum(c.tokens for c in opponent_caches.values()),
        }
        diagnostics["gruTokensPerInferenceDecision"] = (
            diagnostics["gruTokens"] / max(diagnostics["gruInferenceDecisions"], 1))
        batch = RolloutBatch(episodes, version, behavior_digest_before=digest_before,
                             behavior_digest_after=digest_after, diagnostics=diagnostics)
        from trace_audit import audit_batch
        diagnostics["traceAudit"] = audit_batch(batch, config=config)
        return batch, dict(coverage), next_episode + launched
    except BaseException:
        for recorder in recorders:
            recorder.abort()
        raise
    finally:
        model.train(prior_mode)


def _validate_config(config):
    for key in ("workers", "rollout_games", "epochs", "batch_size", "max_actions"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["workers"] > 32 or config["rollout_games"] > 64:
        raise ValueError("workers <=32 and rollout-games <=64 bound rollout memory")
    if config["max_actions"] > 6000:
        raise ValueError("max-actions <=6000 bounds rollout memory")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 0xFFFFFFFF:
        raise ValueError("seed must be uint32")
    for key in ("gamma", "gae_lambda", "clip", "entropy_coef", "potential_scale",
                "learning_rate", "temperature", "mixed_teammate_probability", "target_kl"):
        if not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["gamma"] != 1:
        raise ValueError("gamma must equal 1 until elapsed-time discounting is supported")
    if not 0 <= config["gae_lambda"] <= 1:
        raise ValueError("invalid lambda")
    if config["target_kl"] <= 0:
        raise ValueError("target-kl must be positive")
    if not 0 < config["clip"] < 1 or min(config["temperature"], config["learning_rate"]) <= 0:
        raise ValueError("clip, temperature, and learning-rate must be positive")
    if min(config["entropy_coef"], config["potential_scale"]) < 0:
        raise ValueError("entropy/potential scales must be nonnegative")
    if not 0 <= config["mixed_teammate_probability"] <= 1:
        raise ValueError("invalid mixed teammate probability")


def resolve_config(args, metadata=None):
    config = dict(CORE_DEFAULTS)
    overrides = {}
    saved = metadata.get("config", {}) if metadata else {}
    missing = set(config) - set(saved)
    if metadata and (set(saved) - set(config) or missing - {"target_kl"}):
        raise ValueError("resume requires complete PPO config")
    if metadata and missing:
        overrides["migrationDefaults"] = {key: config[key] for key in sorted(missing)}
        saved = {**config, **saved}
    for flag in config:
        value = getattr(args, flag)
        if metadata:
            if flag in args.explicit and value != saved[flag]:
                raise ValueError(f"explicit --{flag.replace('_', '-')} differs from resume config")
            value = saved[flag]
            setattr(args, flag, value)
        config[flag] = value
    for flag, key in (("save_every", "saveEvery"), ("log_every", "logEvery")):
        if metadata:
            old = metadata[key]
            if flag not in args.explicit:
                setattr(args, flag, old)
            elif getattr(args, flag) != old:
                overrides[key] = {"from": old, "to": getattr(args, flag)}
    _validate_config(config)
    if min(args.save_every, args.log_every) < 1:
        raise ValueError("save/log cadence must be positive")
    return config, overrides


def _protect_outputs(args):
    forbidden = [(f.ROOT / "checkpoints").resolve(), (f.ROOT / "foundation").resolve()]
    outputs = [args.checkpoint_dir, args.tensorboard_dir]
    if args.replay_dir:
        outputs.append(args.replay_dir)
    for path in outputs:
        resolved = path.resolve()
        if any(resolved == old or old in resolved.parents for old in forbidden):
            raise ValueError("PPO must not write v1/v2 checkpoint/log directories")
    source = args.warm_start or args.prepared_start or args.ablate_from or args.upgrade_to_strategic
    if source and args.checkpoint_dir.resolve() == source.resolve().parent:
        raise ValueError("warm-start source and PPO output directory must differ")


def train(args):
    _protect_outputs(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (args.checkpoint_dir / ".team-ppo-train.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another PPO trainer owns this checkpoint directory") from error
        return _train(args)


def _train(args):
    from training_writer import make_writer

    device = f.device_for(args.device)
    configure_precision(device)
    torch.set_num_threads(1 if device.type == "cpu" else 4)
    if args.resume:
        model, data = load_checkpoint(args.resume, device)
        metadata = copy.deepcopy(data["metadata"])
    elif args.upgrade_to_strategic:
        source_model, source_data = load_checkpoint(args.upgrade_to_strategic, device)
        parent_metadata = source_data["metadata"]
        if source_model.sizes != f.SIZES or parent_metadata["status"] != "completed":
            raise ValueError("v3 upgrade requires a completed v2 PPO checkpoint")
        torch.manual_seed(args.seed)
        model = StrategicActorCritic().to(device).eval()
        # Copy the entire learned v2 actor/critic.  The new v3 residual's final
        # layers are zero, so on identical legacy inputs the action logits
        # are exactly preserved before any further training.
        model.load_state_dict({**model.state_dict(), **source_model.state_dict()})
        metadata = {key: copy.deepcopy(parent_metadata[key]) for key in
                    ("sourceGames", "sourceUpdates", "sourceSha256", "sourceCheckpoint",
                     "lineage", "initialization")}
        metadata.update(games=metadata["sourceGames"], ppoGames=0, updates=0,
                        ablationParent={
                            "kind": "strategic-v3-upgrade", "checkpoint": str(args.upgrade_to_strategic.resolve()),
                            "sha256": f.sha256_file(args.upgrade_to_strategic),
                            "priorPpoGames": parent_metadata["ppoGames"],
                            "priorTotalGames": parent_metadata["games"],
                            "optimizerReset": True, "scorePreservingResidual": True,
                        })
        data = None
    elif args.ablate_from:
        parent, frozen = load_checkpoint(args.ablate_from, device)
        parent_metadata = frozen["metadata"]
        if (parent.sizes != f.SIZES or parent_metadata["status"] != "completed"
                or parent_metadata["lineage"]["actorTransfer"] != "random"):
            raise ValueError("ablation requires a completed v2 PPO parent with random-origin lineage")
        torch.manual_seed(args.seed)
        model = StrategicActorCritic().to(device).eval() if args.strategic else parent
        model.load_state_dict({**model.state_dict(), **parent.state_dict()})
        metadata = {key: copy.deepcopy(parent_metadata[key]) for key in
                    ("sourceGames", "sourceUpdates", "sourceSha256", "sourceCheckpoint", "lineage", "initialization")}
        metadata.update(games=0, ppoGames=0, updates=0, ablationParent={
            "checkpoint": str(args.ablate_from.resolve()), "sha256": f.sha256_file(args.ablate_from),
            "ppoGames": parent_metadata["ppoGames"], "optimizerReset": True,
            "scorePreservingResidual": args.strategic})
        data = None
    elif args.prepared_start:
        model, prepared = load_checkpoint(args.prepared_start, device)
        metadata = copy.deepcopy(prepared["metadata"])
        if (metadata["ppoGames"] != 0 or metadata["updates"] != 0
                or metadata["lineage"]["actorTransfer"] != "source-behavior-distilled"):
            raise ValueError("--prepared-start requires a verified, untrained distilled actor")
        if "optimizer" in prepared or "replay" in prepared:
            raise ValueError("prepared initialization must not carry training/replay state")
        metadata["preparedCheckpoint"] = str(args.prepared_start.resolve())
        metadata["preparedSha256"] = f.sha256_file(args.prepared_start)
        data = None
    elif args.from_scratch:
        torch.manual_seed(args.seed)
        model = ActorCritic().to(device).eval()
        metadata = {
            "games": 0, "sourceGames": 0, "sourceUpdates": 0, "ppoGames": 0, "updates": 0,
            "sourceSha256": None, "sourceCheckpoint": None,
            "lineage": {"sourceSha256": None, "sourceGames": 0, "sourceAlgorithm": None,
                        "actorTransfer": "random"},
            "initialization": "random actor from scratch",
        }
        data = None
    else:
        # Seed the new critic's hidden layer too; actor transfer remains exact.
        torch.manual_seed(args.seed)
        model, metadata = warm_start(args.warm_start, device)
        data = None
    config, overrides = resolve_config(args, metadata if data else None)
    sizes = model.sizes
    metadata.update(sizes)
    if args.prepared_start and config["temperature"] != metadata["behaviorDistillation"]["temperature"]:
        raise ValueError("prepared policy temperature must match its verified distillation")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(config["seed"])
    rng = random.Random(config["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    directory = args.checkpoint_dir.resolve()
    initial = directory / "initial.pt"
    if data:
        restore_training(data, optimizer, rng, device)
        if "decisionContractVersion" not in metadata:
            metadata.setdefault("contractMigrations", []).append({
                "from": "legacy team-ppo-v1 metadata without trace contract fields",
                "to": CONTRACT_VERSION, "games": metadata["games"],
                "oldWeightsAndSamplerUnchanged": True, "mcReplayConverted": False})
        initial_model, initial_data = load_checkpoint(initial, "cpu")
        if (initial_model.sizes != sizes or f.sha256_file(initial) != metadata.get("initialSha256")
                or initial_data["metadata"]["ppoGames"] != 0
                or initial_data["metadata"]["updates"] != 0
                or initial_data["metadata"]["sourceSha256"] != metadata["sourceSha256"]):
            raise ValueError("immutable initial-policy lineage mismatch")
        metadata["resumeFreshRoundRestarts"] = metadata.get("resumeFreshRoundRestarts", 0) + 1
    else:
        if initial.exists() or (directory / "latest.pt").exists():
            raise ValueError("new run directory is occupied; use --resume or a separate directory")
        metadata.update(config=config, nextEpisode=0, snapshotPool=[], snapshotsSeen=0,
                        modeCounts={}, decisionCounts={}, phaseCounts={}, continuedRounds=0,
                        mixedTeamGames=0, transitionsCollected=0, maxObservedHistoryLength=0,
                        learnerAskDecisions=0, learnerReturnDecisions=0,
                        saveEvery=args.save_every, logEvery=args.log_every, status="initialized")
        f.immutable_torch_save(checkpoint_data(model, metadata), initial)
        initial_model, _ = load_checkpoint(initial, "cpu")
    metadata.update(initialCheckpoint=str(initial), initialSha256=f.sha256_file(initial),
                    config=config, saveEvery=args.save_every, logEvery=args.log_every,
                    roundResetMode="continue", continuationProbability=1.0,
                    algorithm=ALGORITHM, status="running",
                    resumeRestartsFreshRounds=True, replay="none",
                    onPolicyBatchGames=config["rollout_games"], temperature=config["temperature"],
                    potentialScale=config["potential_scale"], learningRate=config["learning_rate"],
                    maxRolloutBytes=MAX_ROLLOUT_BYTES,
                    decisionContractVersion=CONTRACT_VERSION,
                    rewardVersion=REWARD_VERSION, shapingVersion=SHAPING_VERSION,
                    rulesFingerprint=f.rules_fingerprint(),
                    cudaPrecisionPolicy="FP32 recurrent and matrix operations; TF32 disabled",
                    valueMeaning="expected shaped team return; not a win probability")
    replay_dir = args.replay_dir.resolve() if args.replay_dir else None
    metadata["replayDirectory"] = str(replay_dir) if replay_dir else None
    metadata.pop("error", None)
    if overrides:
        metadata.setdefault("resumeCadenceOverrides", []).append(
            {"games": metadata["games"], "changes": overrides})
        if "migrationDefaults" in overrides:
            metadata.setdefault("configMigrations", []).append({
                "games": metadata["games"], "defaults": overrides["migrationDefaults"],
                "reason": "legacy team-ppo-v1 checkpoint: add explicit diagnostic KL guard",
                "samplerTemperatureUnchanged": config["temperature"]})
        print(f"resume operational cadence overrides: {overrides}", flush=True)
    start_games = metadata["games"]
    goal = args.target_games if args.target_games is not None else start_games + args.games
    if goal < start_games:
        raise ValueError("--target-games is TOTAL cumulative games and cannot be below source/current games")
    metadata["targetGames"] = goal
    league = [(str(initial), initial_model.requires_grad_(False))]
    for path in metadata.get("snapshotPool", []):
        resolved = Path(path).resolve()
        if resolved.parent != directory or not resolved.name.startswith("snapshot-"):
            raise ValueError("league snapshot must belong to this PPO directory")
        opponent, _ = load_checkpoint(resolved, "cpu")
        if opponent.sizes != sizes:
            raise ValueError("historical snapshot uses a different PPO feature protocol")
        league.append((str(resolved), opponent.requires_grad_(False)))
    if len(league) > 6:
        raise ValueError("historical league exceeds six policies")
    metadata["snapshotPool"] = [path for path, _ in league[1:]]
    keep = set(metadata["snapshotPool"])
    for orphan in directory.glob("snapshot-*.pt"):
        # A crash can leave an unpublished candidate, but never delete initial,
        # immutable evaluations, foreign directories, or arbitrary filenames.
        number = orphan.stem.removeprefix("snapshot-")
        if (len(number) == 9 and number.isdigit() and orphan.resolve().parent == directory
                and str(orphan.resolve()) not in keep):
            orphan.unlink()
    workers = []
    writer = None
    progress = {}
    boundary = checkpoint_data(model, metadata, optimizer, rng)
    started = time.monotonic()
    next_save = (start_games // args.save_every + 1) * args.save_every
    next_log = (start_games // args.log_every + 1) * args.log_every
    next_evaluation = (start_games // 1000 + 1) * 1000
    garbage = []
    replay_recorders = []

    def save(snapshot=False, evaluation=False):
        if snapshot:
            metadata["snapshotsSeen"] += 1
            seen = metadata["snapshotsSeen"]
            slot = len(league) - 1 if len(league) < 6 else rng.randrange(seen)
            if slot < 5:
                path = directory / f"snapshot-{metadata['games']:09d}.pt"
                f.immutable_torch_save(checkpoint_data(model, metadata), path)
                frozen, _ = load_checkpoint(path, "cpu")
                frozen.requires_grad_(False)
                if len(league) < 6:
                    league.append((str(path), frozen))
                else:
                    old, _ = league[slot + 1]
                    league[slot + 1] = (str(path), frozen)
                    owned = Path(old)
                    if owned.parent == directory and owned.name.startswith("snapshot-"):
                        garbage.append(owned)
            metadata["snapshotPool"] = [path for path, _ in league[1:]]
        if evaluation:
            path = directory / f"milestone-{metadata['games']:09d}.pt"
            if not path.exists():
                f.immutable_torch_save(checkpoint_data(model, metadata), path)
        f.atomic_torch_save(checkpoint_data(model, metadata, optimizer, rng), directory / "latest.pt")
        f.atomic_json_save({**metadata, "methods": initialization_methods(metadata)},
                           directory / "progress.json")

    try:
        writer = make_writer(args.tensorboard_dir)
        writer.add_text("team_ppo/methods", __doc__ + "\n" + str(initialization_methods(metadata)), start_games)
        for _ in range(min(config["workers"], config["rollout_games"], goal - start_games)):
            bridge = ("strategic-env.js" if sizes == strategic.SIZES else "audit-env.js" if replay_dir else None)
            workers.append({"env": f.NodeEnvironment(
                script=f.ROOT / bridge if bridge else None,
                **({"sizes": sizes} if sizes != f.SIZES else {}),
                max_actions=config["max_actions"]),
                            "state": None, "finished": False})
        save()
        boundary = checkpoint_data(model, metadata, optimizer, rng)
        while metadata["games"] < goal:
            progress.clear()
            replay_recorders.clear()
            batch, coverage, next_episode = collect_batch(
                model, workers, min(config["rollout_games"], goal - metadata["games"]),
                rng, config, metadata["updates"], metadata["nextEpisode"],
                tuple(policy for _, policy in league), progress,
                replay_dir=replay_dir, replay_recorders=replay_recorders)
            metrics = update_batch(model, optimizer, batch, metadata["updates"], config, rng)
            count = len(batch.episodes)
            metadata["ppoGames"] += count
            metadata["games"] += count
            metadata["updates"] += metrics["optimizerSteps"]
            metadata["nextEpisode"] = next_episode
            metadata["transitionsCollected"] += metrics["transitions"]
            for key in ("continuedRounds", "mixedTeamGames"):
                metadata[key] += coverage.get(key, 0)
            metadata["maxObservedHistoryLength"] = max(
                metadata["maxObservedHistoryLength"], coverage.get("maxObservedHistoryLength", 0))
            for prefix, field in (("decision:", "decisionCounts"), ("mode:", "modeCounts"),
                                  ("phase:", "phaseCounts")):
                counter = Counter(metadata[field])
                counter.update({key[len(prefix):]: value for key, value in coverage.items()
                                if key.startswith(prefix)})
                metadata[field] = dict(counter)
            metadata["learnerAskDecisions"] = metadata["decisionCounts"].get("ask", 0)
            metadata["learnerReturnDecisions"] = metadata["decisionCounts"].get("return", 0)
            metadata["lastBatch"] = {"games": count, "decisions": metrics["transitions"],
                                     "behaviorVersion": batch.behavior_version, **metrics}
            boundary = checkpoint_data(model, metadata, optimizer, rng)
            progress.clear()
            game = metadata["games"]
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "games": game, "sourceGames": metadata["sourceGames"],
                    "ppoGames": metadata["ppoGames"], "updates": metadata["updates"],
                    "behaviorVersion": batch.behavior_version, "coverage": coverage,
                    **metrics}, allow_nan=False) + "\n")

            def write_metrics(values, prefix="team_ppo/training"):
                for key, value in values.items():
                    if isinstance(value, dict):
                        write_metrics(value, prefix + "/" + key)
                    elif isinstance(value, (int, float)) and math.isfinite(value):
                        writer.add_scalar(prefix + "/" + key, value, game)
                    elif isinstance(value, str):
                        writer.add_text(prefix + "/" + key, value, game)

            write_metrics(metrics)
            for key in ("learnerAskDecisions", "learnerReturnDecisions", "maxObservedHistoryLength",
                        "transitionsCollected", "continuedRounds", "mixedTeamGames"):
                writer.add_scalar(f"team_ppo/coverage/{key}", metadata[key], game)
            if game >= next_log or game == goal:
                print(f"games={game} ppoGames={metadata['ppoGames']} updates={metadata['updates']} "
                      f"asks={metadata['learnerAskDecisions']} returns={metadata['learnerReturnDecisions']} "
                      f"games/sec={(game-start_games)/max(time.monotonic()-started, 1e-9):.3f}", flush=True)
                next_log = (game // args.log_every + 1) * args.log_every
            snapshot = game >= next_save
            evaluation = game >= next_evaluation
            if snapshot or evaluation:
                save(snapshot, evaluation)
                next_save = (game // args.save_every + 1) * args.save_every
                next_evaluation = (game // 1000 + 1) * 1000
            if replay_dir:
                save()
                boundary = checkpoint_data(model, metadata, optimizer, rng)
                for recorder in replay_recorders:
                    recorder.commit()
            boundary = checkpoint_data(model, metadata, optimizer, rng)
            for obsolete in garbage:
                obsolete.unlink(missing_ok=True)
            garbage.clear()
        metadata["status"] = "completed"
        save(evaluation=True)
        return metadata
    except BaseException as error:
        for recorder in replay_recorders:
            recorder.abort()
        # Collection and optimizer updates are one transaction. Roll back partial
        # optimizer steps AND RNG; never attach a made-up terminal reward.
        interrupted = isinstance(error, KeyboardInterrupt)
        saved = boundary["metadata"]
        saved.update(status="interrupted" if interrupted else "failed",
                     discardedBatchDecisions=progress.get("decisions", 0),
                     discardedCompleteEpisodes=progress.get("completedEpisodes", 0),
                     discardedInFlightEpisodes=progress.get("inFlightEpisodes", 0),
                     error=f"{type(error).__name__}: {error}")
        from trace_audit import TraceAuditError
        if isinstance(error, TraceAuditError):
            saved["failedTraceAudit"] = error.report
        f.atomic_torch_save(boundary, directory / "latest.pt")
        f.atomic_json_save(saved, directory / "progress.json")
        print(f"{saved['status']}: discarded batch decisions={saved['discardedBatchDecisions']} "
              f"complete episodes={saved['discardedCompleteEpisodes']} "
              f"in-flight episodes={saved['discardedInFlightEpisodes']}", flush=True)
        if not interrupted:
            raise
        return saved
    finally:
        for worker in workers:
            worker["env"].close()
        if writer is not None:
            writer.flush()
            writer.close()


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    source = training.add_mutually_exclusive_group(required=True)
    source.add_argument("--warm-start", type=Path)
    source.add_argument("--prepared-start", type=Path,
                        help="verified source-behavior-distilled PPO initialization, without optimizer state")
    source.add_argument("--from-scratch", action="store_true",
                        help="random actor with no inherited v2 weights or source games")
    source.add_argument("--resume", type=Path)
    source.add_argument("--ablate-from", type=Path, help="fork completed random-origin v2 PPO for matched ablation")
    source.add_argument("--upgrade-to-strategic", type=Path,
                        help="fork ANY completed v2 PPO (including transferred lineage) into score-preserving v3")
    training.add_argument("--strategic", action="store_true", help="add a zero-initialized v3 residual to --ablate-from")
    count = training.add_mutually_exclusive_group()
    count.add_argument("--games", type=int, default=100000,
                       help="ADDITIONAL PPO completed games; cumulative games include warm-start source")
    count.add_argument("--target-games", type=int, help="TOTAL source + PPO games, e.g. 100000")
    training.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    training.add_argument("--tensorboard-dir", type=Path, default=f.ROOT / "ppo" / "runs")
    training.add_argument("--replay-dir", type=Path,
                          help="record all complete games for private offline playback; never inference inputs")
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for key, default in CORE_DEFAULTS.items():
        training.add_argument("--" + key.replace("_", "-"), type=type(default), default=default)
    training.add_argument("--save-every", type=int, default=1000)
    training.add_argument("--log-every", type=int, default=100)
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    args.explicit = {token[2:].split("=")[0].replace("-", "_") for token in argv if token.startswith("--")}
    if args.command == "train":
        if args.strategic and not args.ablate_from:
            parser.error("--strategic requires --ablate-from; resume detects its checkpoint encoding")
        if args.games < 0 or (args.target_games is not None and args.target_games < 0):
            parser.error("game counts must be nonnegative")
        if args.gamma != 1.0:
            parser.error("gamma must equal 1 until elapsed-time discounting is supported")
    return args


if __name__ == "__main__":
    args = arguments()
    if args.command == "train":
        train(args)
    else:
        model = load_checkpoint(args.checkpoint, f.device_for(args.device))[0]
        if model.sizes != f.SIZES:
            raise ValueError("v3 checkpoints require the opt-in training/serve_strategic.py worker")
        f.serve(model)
