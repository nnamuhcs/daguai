"""Versioned, actor-private PPO trace records (not a new feature protocol).

One absolute public-event array is shared by each complete round. Decision.prefix
is the pre-action boundary; observations and legal features belong ONLY to actor.
No checkpoint/replay conversion is performed by these contracts.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np

import foundation as f
import strategic_protocol as strategic

CONTRACT_VERSION = 1
REWARD_VERSION = "terminal-team-v1"
SHAPING_VERSION = "same-actor-finisher-potential-v1"


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def action_identity(action):
    """Exact action identity, without conflating wildcard/card-copy semantics.

    Retain every non-feature field and original list order; do not deduplicate.
    """
    return json.dumps({k: v for k, v in action.items() if k != "features"},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def legal_support_digest(identities, features):
    values = np.ascontiguousarray(features, dtype=np.float32)
    return hashlib.sha256(
        json.dumps(list(identities), separators=(",", ":")).encode()
        + str(values.shape).encode() + values.tobytes()).hexdigest()


@dataclass
class Decision:
    observation: np.ndarray
    actions: np.ndarray
    player: int
    prefix: int
    kind: str
    index: int
    old_logprob: float
    old_value: float
    contract_version: int = 0  # Legacy in-memory fixtures, never serialized MC replay.
    episode_id: int = -1
    global_step: int = -1
    action_identities: tuple[str, ...] = ()
    behavior_digest: str = ""
    legal_support_digest: str = ""
    next_actor_index: int | None = None
    elapsed_events: int = 0
    elapsed_steps: int = 0
    terminal_reward: float = 0.0
    potential_before: float = 0.0
    potential_successor: float = 0.0

    @property
    def team(self):
        return self.player % 2

    @property
    def chosen_action_identity(self):
        return self.action_identities[self.index]


@dataclass
class Episode:
    events: np.ndarray
    decisions: list[Decision]
    winner: int | None
    behavior_version: int
    contract_version: int = 0
    episode_id: int = -1
    behavior_digest: str = ""
    controlled_seats: tuple[int, ...] = ()
    global_steps: int = 0
    terminal: bool = True
    mode: str = "fixture"
    potential_scale: float = .3
    reward_version: str = REWARD_VERSION
    shaping_version: str = SHAPING_VERSION
    temperature: float = .2
    encoding_version: int = f.ENCODING_VERSION

    def history(self, decision):
        if not 0 <= decision.prefix <= len(self.events):
            raise ValueError("invalid public history prefix")
        return f.relative_history(self.events[:decision.prefix], decision.player)


@dataclass
class RolloutBatch:
    episodes: list[Episode]
    behavior_version: int
    consumed: bool = False
    behavior_digest_before: str = ""
    behavior_digest_after: str = ""
    diagnostics: dict = field(default_factory=dict)

    def validate(self, version):
        if self.consumed or self.behavior_version != version:
            raise ValueError("stale or already-consumed on-policy batch")
        if any(e.behavior_version != version for e in self.episodes):
            raise ValueError("mixed behavior-policy versions")
        if self.behavior_digest_before != self.behavior_digest_after:
            raise ValueError("behavior changed during rollout")


def actor_links(episode):
    next_index = [None] * len(episode.decisions)
    latest = {}
    for i in reversed(range(len(episode.decisions))):
        player = episode.decisions[i].player
        next_index[i] = latest.get(player)
        latest[player] = i
    return next_index


def finalize_episode(episode, potential):
    """Attach same-actor interval diagnostics after genuine terminal completion."""
    if not episode.terminal:
        raise ValueError("cannot finalize an unfinished/truncated round")
    for i, following in enumerate(actor_links(episode)):
        d = episode.decisions[i]
        successor = None if following is None else episode.decisions[following]
        d.next_actor_index = following
        d.elapsed_events = (len(episode.events) if successor is None else successor.prefix) - d.prefix
        d.elapsed_steps = (episode.global_steps if successor is None else successor.global_step) - d.global_step
        d.terminal_reward = f.team_reward(d.player, episode.winner) if successor is None else 0.0
        d.potential_before = potential(d.observation, episode.potential_scale)
        d.potential_successor = 0.0 if successor is None else potential(
            successor.observation, episode.potential_scale)
    return episode


def validate_episode(episode, potential, tolerance=1e-6):
    if episode.contract_version != CONTRACT_VERSION or episode.terminal is not True:
        raise ValueError("trace requires current contract and complete terminal round")
    if episode.reward_version != REWARD_VERSION or episode.shaping_version != SHAPING_VERSION:
        raise ValueError("incompatible trace reward/shaping version")
    sizes = {f.ENCODING_VERSION: f.SIZES, strategic.SIZES["encodingVersion"]: strategic.SIZES}.get(
        episode.encoding_version)
    if sizes is None:
        raise ValueError("unknown trace feature encoding")
    f.matrix(episode.events, f.HISTORY_SIZE, "events", f.MAX_HISTORY, empty=True)
    f.team_reward(0, episode.winner)
    if type(episode.episode_id) is not int or episode.episode_id < 0:
        raise ValueError("invalid episode ID")
    if (len(set(episode.controlled_seats)) != len(episode.controlled_seats)
            or any(type(actor) is not int or not 0 <= actor < 6 for actor in episode.controlled_seats)):
        raise ValueError("invalid controlled learner seats")
    if (len(episode.behavior_digest) != 64 or any(
            c not in "0123456789abcdef" for c in episode.behavior_digest)):
        raise ValueError("invalid behavior digest")
    previous_prefix, previous_step = -1, -1
    for i, following in enumerate(actor_links(episode)):
        d = episode.decisions[i]
        if d.contract_version != CONTRACT_VERSION or d.episode_id != episode.episode_id:
            raise ValueError("decision/episode contract mismatch")
        if type(d.player) is not int or d.player not in episode.controlled_seats or not 0 <= d.player < 6:
            raise ValueError("non-learner decision in PPO trace")
        if d.behavior_digest != episode.behavior_digest:
            raise ValueError("decision behavior digest mismatch")
        if not previous_prefix <= d.prefix <= len(episode.events):
            raise ValueError("future/nonmonotone pre-action public prefix")
        if not previous_step < d.global_step < episode.global_steps:
            raise ValueError("invalid pre-action global step")
        previous_prefix, previous_step = d.prefix, d.global_step
        f.finite_vector(d.observation, sizes["observationSize"], "observation")
        f.matrix(d.actions, sizes["actionSize"], "actions", None)
        if (len(d.actions) == 0 or type(d.index) is not int or not 0 <= d.index < len(d.actions)
                or len(d.action_identities) != len(d.actions)):
            raise ValueError("invalid ordered legal support/chosen index")
        if d.legal_support_digest != legal_support_digest(d.action_identities, d.actions):
            raise ValueError("ordered legal support changed after pre-action capture")
        for identity in d.action_identities:
            parsed = json.loads(identity)
            if not isinstance(parsed, dict) or action_identity(parsed) != identity:
                raise ValueError("noncanonical action identity")
        if json.loads(d.chosen_action_identity).get("kind") != d.kind:
            raise ValueError("chosen action kind mismatch")
        if not math.isfinite(d.old_value) or not math.isfinite(d.old_logprob) or d.old_logprob > tolerance:
            raise ValueError("nonfinite/invalid old policy/value")
        successor = None if following is None else episode.decisions[following]
        if d.next_actor_index != following:
            raise ValueError("incorrect next SAME actor index")
        expected_events = (len(episode.events) if successor is None else successor.prefix) - d.prefix
        expected_steps = (episode.global_steps if successor is None else successor.global_step) - d.global_step
        if d.elapsed_events != expected_events or d.elapsed_steps != expected_steps:
            raise ValueError("incorrect elapsed global interval")
        reward = f.team_reward(d.player, episode.winner) if successor is None else 0.0
        phi = 0.0 if successor is None else potential(successor.observation, episode.potential_scale)
        if (d.terminal_reward != reward or abs(d.potential_successor - phi) > tolerance
                or abs(d.potential_before - potential(d.observation, episode.potential_scale)) > tolerance):
            raise ValueError("terminal reward/sign or same-actor potential mismatch")
