"""V2 random-initialized, terminal-team-reward learning with public-history GRU.

No shaping, imitation labels, hidden-card teacher, promotion, or v1 file writes.
Replay stores one public event array per episode and decision prefix references.
Evaluation measures fresh-round play only, not tribute/return-round strength.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from daguai_rl import (
    NodeEnvironment as LegacyEnvironment, atomic_json_save, atomic_torch_save,
    choose_action, device_for, finite_vector, team_reward,
)

ENCODING_VERSION = 2
OBSERVATION_SIZE = 922
ACTION_SIZE = 81
HISTORY_SIZE = 71
MAX_HISTORY = 4096
SIZES = dict(encodingVersion=2, observationSize=922, actionSize=81, historySize=71)
ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "foundation" / "checkpoints"
METHODS = {
    "algorithm": "Monte Carlo terminal TEAM action-value regression",
    "initialization": "random, from scratch",
    "reward": "terminal team outcome +1/-1/0 only",
    "history": "GRU over full observed public episode, acting-seat-relative actors/recipients",
    "asking": "learned legal action, including reported public responses",
    "returns": "learned legal action in continued rounds",
    "opponents": ">=50% all-learner selfplay, heuristics, old snapshots; mixed heuristic teammates",
    "labels": "current learner decisions only; no imitation, shaping, or hidden-card teacher",
    "replay": "bounded decisions and public-history bytes; shared episode arrays with prefixes",
    "strength": "loss is not evidence of strength; no automatic promotion",
}


def check_encoding(metadata, sizes=SIZES):
    if not isinstance(metadata, dict):
        raise ValueError("missing encoding metadata")
    for key, expected in sizes.items():
        if type(metadata.get(key)) is not int or metadata[key] != expected:
            raise ValueError(f"incompatible {key}: expected {expected}, got {metadata.get(key)!r}")


def matrix(value, width, name, maximum, *, empty=False):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be an array")
    if not empty and not len(value):
        raise ValueError(f"{name} must be nonempty")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{name} must have at most {maximum} rows")
    raw = np.asarray(value)
    if not len(value) and empty:
        if raw.shape not in ((0,), (0, width)):
            raise ValueError(f"{name} rows must contain {width} finite numbers")
        return np.empty((0, width), dtype=np.float32)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float32)
    if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError(f"{name} rows must contain {width} finite numbers")
    return result


def relative_history(events, player):
    if type(player) is not int or not 0 <= player < 6:
        raise ValueError("invalid acting seat")
    result = matrix(events, HISTORY_SIZE, "history", MAX_HISTORY, empty=True).copy()
    result[:, 54:60] = np.roll(result[:, 54:60], -player, axis=1)
    result[:, 60:66] = np.roll(result[:, 60:66], -player, axis=1)
    return result


def validate_state(state, sizes=SIZES):
    if not isinstance(state, dict):
        raise ValueError("environment state must be an object")
    finite_vector(state.get("observation"), sizes["observationSize"], "observation")
    player = state.get("player")
    if type(player) is not int or not 0 <= player < 6:
        raise ValueError("invalid environment player")
    if state.get("phase") not in ("playing", "returning", "over"):
        raise ValueError("invalid environment phase")
    if type(state.get("terminal")) is not bool:
        raise ValueError("invalid terminal flag")
    matrix(state.get("events"), HISTORY_SIZE, "events", MAX_HISTORY, empty=True)
    actions = state.get("actions")
    if not isinstance(actions, list):
        raise ValueError("invalid legal action list")
    if state["terminal"]:
        if state["phase"] != "over" or not isinstance(state.get("result"), dict):
            raise ValueError("terminal state requires result and over phase")
        if "winner" not in state["result"]:
            raise ValueError("terminal result requires winner")
        team_reward(player, state["result"]["winner"])
        if actions:
            raise ValueError("terminal state must have no actions")
    elif not actions or state["phase"] == "over" or state.get("result") is not None:
        raise ValueError("inconsistent nonterminal state")
    for action in actions:
        if not isinstance(action, dict) or action.get("kind") not in ("play", "pass", "return", "ask"):
            raise ValueError("invalid action kind")
        ids = action.get("ids")
        if not isinstance(ids, list) or any(type(card) is not int or card < 0 for card in ids):
            raise ValueError("invalid action card IDs")
        finite_vector(action.get("features"), sizes["actionSize"], "action")
        if action["kind"] == "ask":
            target = action.get("target")
            if type(target) is not int or not 0 <= target < 6 or target == player or ids:
                raise ValueError("invalid asking recipient")


class NodeEnvironment(LegacyEnvironment):
    """Reuse only the v1 transport, never its encoding validator or script."""

    def __init__(self, script=None, timeout=60, max_actions=6000, sizes=SIZES):
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self.timeout = timeout
        self.sizes = sizes
        self._buffer = b""
        self.process = subprocess.Popen(
            ["node", str(script or ROOT / "foundation-env.js")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=0,
            env={**os.environ, "RL_MAX_ACTIONS": str(max_actions)},
        )
        try:
            check_encoding(self.request("hello"), sizes)
        except BaseException:
            self.close()
            raise

    def state(self, op, **kwargs):
        result = self.request(op, **kwargs)
        validate_state(result, self.sizes)
        if op == "reset" and result["events"]:
            raise ValueError("reset must start an empty public history")
        return result


class ActionValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 256), nn.ReLU())
        self.history = nn.GRU(HISTORY_SIZE, 128, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(465, 256), nn.ReLU(), nn.Linear(256, 128),
            nn.ReLU(), nn.Linear(128, 1),
        )
        self.metadata = {}
        self.checkpoint = None

    def encode_history(self, histories):
        device = next(self.parameters()).device
        output = torch.zeros((len(histories), 128), device=device)
        indices = [i for i, history in enumerate(histories) if len(history)]
        if indices:
            values = [histories[i].to(device=device, dtype=torch.float32) for i in indices]
            packed = pack_padded_sequence(
                pad_sequence(values, batch_first=True),
                [len(value) for value in values], batch_first=True, enforce_sorted=False,
            )
            _, hidden = self.history(packed)
            output = output.index_copy(0, torch.tensor(indices, device=device), hidden[0])
        return output

    def forward(self, observations, actions, histories):
        if (observations.ndim != 2 or observations.shape[1] != OBSERVATION_SIZE
                or actions.shape != (len(observations), ACTION_SIZE)
                or len(histories) != len(observations)):
            raise ValueError("model requires aligned observation/action/history batches")
        encoded = torch.cat((self.base(observations), self.encode_history(histories), actions), dim=1)
        return self.head(encoded).squeeze(-1)


@torch.no_grad()
def score_many(model, states):
    """Score all actions; histories supplied here are already acting-seat-relative."""
    prepared = []
    for observation, actions, history in states:
        prepared.append((
            finite_vector(observation, OBSERVATION_SIZE, "observation"),
            matrix(actions, ACTION_SIZE, "actions", None),
            matrix(history, HISTORY_SIZE, "history", MAX_HISTORY, empty=True),
        ))
    if not prepared:
        return []
    device = next(model.parameters()).device
    history_codes = model.encode_history([torch.from_numpy(state[2]) for state in prepared])
    base_codes = model.base(torch.from_numpy(np.stack([state[0] for state in prepared])).to(device))
    results = []
    for i, (_, actions, _) in enumerate(prepared):
        features = torch.from_numpy(actions).to(device)
        combined = torch.cat((
            base_codes[i:i + 1].expand(len(actions), -1),
            history_codes[i:i + 1].expand(len(actions), -1), features,
        ), dim=1)
        scores = model.head(combined).squeeze(-1).cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("non-finite model scores")
        results.append(scores)
    return results


class ReplayBuffer:
    def __init__(self, capacity=10000, history_bytes=128 * 1024 * 1024):
        if type(capacity) is not int or not 1 <= capacity <= 10000:
            raise ValueError("replay capacity must be 1..10000")
        if type(history_bytes) is not int or history_bytes < MAX_HISTORY * HISTORY_SIZE * 4:
            raise ValueError("history byte bound must accommodate one full episode")
        self.capacity = capacity
        self.history_limit = history_bytes
        self.entries = deque()
        self.episodes = {}
        self.references = Counter()
        self.history_bytes = 0
        self.next_episode = 0

    def __len__(self):
        return len(self.entries)

    def evict(self):
        entry = self.entries.popleft()
        episode = entry[5]
        self.references[episode] -= 1
        if not self.references[episode]:
            self.history_bytes -= self.episodes.pop(episode).nbytes
            del self.references[episode]

    def add_episode(self, events, decisions, winner):
        history = matrix(events, HISTORY_SIZE, "events", MAX_HISTORY, empty=True).copy()
        checked = []
        for observation, action, player, prefix, kind in decisions:
            if type(prefix) is not int or not 0 <= prefix <= len(history):
                raise ValueError("invalid public-history decision prefix")
            if kind not in ("play", "pass", "return", "ask"):
                raise ValueError("invalid replay action kind")
            checked.append((
                finite_vector(observation, OBSERVATION_SIZE, "observation").copy(),
                finite_vector(action, ACTION_SIZE, "action").copy(),
                player, prefix, team_reward(player, winner), kind,
            ))
        if not checked:
            return
        episode = self.next_episode
        self.next_episode += 1
        history.flags.writeable = False
        self.episodes[episode] = history
        self.history_bytes += history.nbytes
        # Capacity trimming before insertion avoids evicting the new shared array mid-insert.
        checked = checked[-self.capacity:]
        while self.entries and (
            len(self.entries) + len(checked) > self.capacity
            or self.history_bytes > self.history_limit
        ):
            self.evict()
        for observation, action, player, prefix, reward, kind in checked:
            self.entries.append((observation, action, player, prefix, reward, episode, kind))
            self.references[episode] += 1

    def sample(self, size, rng, device):
        selected = rng.sample(list(self.entries), size)
        histories = [
            torch.from_numpy(relative_history(self.episodes[episode][:prefix], player))
            for _, _, player, prefix, _, episode, _ in selected
        ]
        return (
            torch.from_numpy(np.stack([entry[0] for entry in selected])).to(device),
            torch.from_numpy(np.stack([entry[1] for entry in selected])).to(device),
            histories, torch.tensor([entry[4] for entry in selected], device=device),
        )

    def state_dict(self):
        entries = list(self.entries)
        return {
            "capacity": self.capacity, "historyLimit": self.history_limit,
            "nextEpisode": self.next_episode,
            "episodes": {key: torch.from_numpy(value.copy()) for key, value in self.episodes.items()},
            "observations": torch.from_numpy(np.stack([e[0] for e in entries])) if entries
            else torch.empty((0, OBSERVATION_SIZE)),
            "actions": torch.from_numpy(np.stack([e[1] for e in entries])) if entries
            else torch.empty((0, ACTION_SIZE)),
            "references": [list(e[2:]) for e in entries],
        }

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or state["historyLimit"] != self.history_limit:
            raise ValueError("resume replay capacity/history bound mismatch")
        self.entries.clear()
        self.episodes.clear()
        self.references.clear()
        self.history_bytes = 0
        observations = state["observations"].cpu().numpy()
        actions = state["actions"].cpu().numpy()
        refs = state["references"]
        if len(refs) > self.capacity or len(observations) != len(refs) or len(actions) != len(refs):
            raise ValueError("invalid replay checkpoint lengths")
        for key, value in state["episodes"].items():
            if type(key) is not int or key < 0:
                raise ValueError("invalid episode key")
            history = matrix(value.cpu().numpy(), HISTORY_SIZE, "events", MAX_HISTORY, empty=True).copy()
            history.flags.writeable = False
            self.episodes[key] = history
            self.history_bytes += history.nbytes
        if self.history_bytes > self.history_limit:
            raise ValueError("saved histories exceed memory bound")
        for observation, action, ref in zip(observations, actions, refs):
            player, prefix, reward, episode, kind = ref
            team_reward(player, None)
            if (episode not in self.episodes or type(prefix) is not int
                    or not 0 <= prefix <= len(self.episodes[episode])
                    or reward not in (-1, 0, 1) or kind not in ("ask", "return", "play", "pass")):
                raise ValueError("invalid saved decision")
            self.entries.append((
                finite_vector(observation, OBSERVATION_SIZE, "observation").copy(),
                finite_vector(action, ACTION_SIZE, "action").copy(),
                player, prefix, float(reward), episode, kind,
            ))
            self.references[episode] += 1
        if set(self.references) != set(self.episodes):
            raise ValueError("orphan replay histories")
        self.next_episode = state["nextEpisode"]
        if (type(self.next_episode) is not int or self.next_episode < 0
                or any(key >= self.next_episode for key in self.episodes)):
            raise ValueError("invalid next replay episode")


def update_model(model, optimizer, replay, batch_size, rng, device):
    observations, actions, histories, targets = replay.sample(batch_size, rng, device)
    previous_mode = model.training
    try:
        # CuDNN GRU backward requires its forward to run in training mode.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(observations, actions, histories), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError("non-finite model parameters")
        return float(loss.detach().cpu())
    finally:
        model.train(previous_mode)


def rng_state(rng):
    numpy_state = np.random.get_state()
    state = {
        "local": rng.getstate(), "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)),
                  int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }
    return state


def checkpoint_data(model, metadata, optimizer=None, replay=None, rng=None):
    result = {
        "metadata": {**metadata, **SIZES, "learnReturns": True, "methods": METHODS},
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if optimizer is not None:
        result.update(optimizer=optimizer.state_dict(), replay=replay.state_dict(), rng=rng_state(rng))
    return result


def load_checkpoint(path, device):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be an object")
    check_encoding(data.get("metadata"))
    for key in ("games", "updates"):
        if type(data["metadata"].get(key)) is not int or data["metadata"][key] < 0:
            raise ValueError(f"invalid checkpoint {key}")
    if data["metadata"].get("learnReturns") is not True:
        raise ValueError("v2 checkpoint must declare learned returns")
    algorithm = data["metadata"].get("algorithm")
    if algorithm == "team-ppo-v1":
        from team_ppo import load_checkpoint as load_ppo
        return load_ppo(path, device)
    if algorithm is not None:
        raise ValueError(f"unsupported foundation checkpoint algorithm: {algorithm}")
    # Loading a frozen opponent must not perturb the learner's RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = ActionValue().to(device)
    model.load_state_dict(data["model"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint contains non-finite parameters")
    model.eval()
    model.metadata = data["metadata"]
    model.checkpoint = str(Path(path).resolve())
    return model, data


def restore_training(data, optimizer, replay, rng, device):
    if any(key not in data for key in ("optimizer", "replay", "rng")):
        raise ValueError("resume requires optimizer, replay, and RNG state")
    optimizer.load_state_dict(data["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError("checkpoint contains non-finite optimizer state")
                if key != "step":
                    state[key] = value.to(device)
    replay.load_state_dict(data["replay"])
    state = data["rng"]
    rng.setstate(state["local"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    array_state = state["numpy"]
    np.random.set_state((
        array_state[0], array_state[1].cpu().numpy().astype(np.uint32), *array_state[2:],
    ))
    if device.type == "cuda" and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def handle_request(model, request):
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    if request.get("op") == "info":
        metadata = dict(model.metadata)
        if metadata.get("status") in ("running", "completed"):
            metadata.pop("error", None)
        return {
            **metadata, **SIZES, "available": True, "learnReturns": True,
            "games": model.metadata.get("games", 0), "updates": model.metadata.get("updates", 0),
            "device": str(next(model.parameters()).device), "checkpoint": model.checkpoint,
        }
    if request.get("op") == "score":
        matrix(request.get("actions"), ACTION_SIZE, "actions", 512)
        scores = score_many(model, [(
            request.get("observation"), request.get("actions"), request.get("history"),
        )])[0]
        return {"scores": scores.tolist()}
    if request.get("op") == "close":
        return {"closed": True}
    raise ValueError("unknown op; expected info, score, or close")


def serve(model, input_stream=None, output_stream=None):
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    maximum = 8 * 1024 * 1024
    while True:
        line = input_stream.readline(maximum + 1)
        if not line:
            return
        request = None
        try:
            if len(line) > maximum:
                while line and not line.endswith("\n"):
                    line = input_stream.readline(maximum + 1)
                raise ValueError("request exceeds 8 MiB")
            request = json.loads(line)
            response = handle_request(model, request)
        except Exception as error:
            response = {"error": str(error)}
        output_stream.write(json.dumps(response, allow_nan=False) + "\n")
        output_stream.flush()
        if isinstance(request, dict) and request.get("op") == "close" and "error" not in response:
            return


def append_events(worker, state):
    events = matrix(state["events"], HISTORY_SIZE, "events", MAX_HISTORY, empty=True)
    if len(worker["events"]) + len(events) > MAX_HISTORY:
        raise RuntimeError("public history exceeds 4096; no truncation or rewarded outcome")
    worker["events"].extend(events)


def learner_job(worker):
    state = worker["state"]
    return (
        state["observation"], [action["features"] for action in state["actions"]],
        relative_history(worker["events"], state["player"]),
    )


def record_decision(worker, index):
    state = worker["state"]
    action = state["actions"][index]
    worker["decisions"].append((
        np.array(state["observation"], dtype=np.float32),
        np.array(action["features"], dtype=np.float32), state["player"],
        len(worker["events"]), action["kind"],
    ))


def immutable_torch_save(value, destination):
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pt")
    try:
        atomic_torch_save(value, temporary)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def train(args):
    destination = args.checkpoint_dir.resolve()
    legacy = (ROOT / "checkpoints").resolve()
    if destination == legacy or legacy in destination.parents:
        raise ValueError("v2 output must not be inside the v1 checkpoints directory")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".foundation-train.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another v2 trainer holds this checkpoint directory") from error
        return _train(args)


def _train(args):
    from torch.utils.tensorboard import SummaryWriter

    device = device_for(args.device)
    torch.set_num_threads(1 if device.type == "cpu" else 4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(args.seed)
    metadata = {
        "games": 0, "updates": 0, "seed": args.seed, "nextEpisode": 0,
        "modeCounts": {}, "decisionCounts": {}, "phaseCounts": {},
        "continuedRounds": 0, "mixedTeamGames": 0, "snapshotPool": [],
        "learnerAskDecisions": 0, "learnerReturnDecisions": 0,
        "transitionsCollected": 0, "maxObservedHistoryLength": 0,
        "status": "running", "resumeRestartsInFlight": True,
    }
    data = None
    if args.resume:
        model, data = load_checkpoint(args.resume, device)
        if data["metadata"].get("algorithm") == "team-ppo-v1":
            raise ValueError("PPO checkpoints must resume with team_ppo.py, not Monte Carlo training")
        metadata.update(data["metadata"])
    else:
        model = ActionValue().to(device)
    saved_capacity = data["replay"]["capacity"] if data and "replay" in data else None
    capacity = args.replay_size if args.replay_size is not None else saved_capacity or 10000
    if saved_capacity is not None and capacity != saved_capacity:
        raise ValueError("explicit --replay-size differs from resume capacity; omit to inherit")
    replay = ReplayBuffer(capacity, data["replay"]["historyLimit"] if data else 128 * 1024 * 1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate or 1e-4)
    cadence_changes = {}
    if data:
        for flag, key in (("seed", "seed"), ("batch_size", "batchSize"), ("warmup", "warmup"),
                          ("updates_per_game", "updatesPerGame"),
                          ("epsilon", "epsilon"), ("learning_rate", "learningRate"),
                          ("mixed_teammate_probability", "mixedTeammateProbability"),
                          ("max_actions", "maxActions"), ("workers", "workers")):
            if flag in args.explicit and getattr(args, flag) != metadata.get(key):
                raise ValueError(f"explicit --{flag.replace('_', '-')} differs from resume metadata")
            if flag not in args.explicit and key in metadata:
                setattr(args, flag, metadata[key])
        for flag, key in (("save_every", "saveEvery"), ("log_every", "logEvery")):
            saved = metadata.get(key)
            if flag not in args.explicit and saved is not None:
                setattr(args, flag, saved)
            elif flag in args.explicit and saved != getattr(args, flag):
                cadence_changes[key] = {"from": saved, "to": getattr(args, flag)}
        if max(args.batch_size, args.warmup) > capacity or args.warmup < args.batch_size:
            raise ValueError("invalid resumed batch-size/warmup")
        restore_training(data, optimizer, replay, rng, device)
        del data
    else:
        if max(args.batch_size, args.warmup) > capacity:
            raise ValueError("batch-size and warmup must not exceed replay capacity")
        if args.warmup < args.batch_size:
            raise ValueError("warmup must be at least batch-size")
        args.learning_rate = optimizer.param_groups[0]["lr"]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_path = args.checkpoint_dir / "initial.pt"
    if not args.resume and (initial_path.exists() or (args.checkpoint_dir / "latest.pt").exists()):
        raise ValueError("new training directory already contains initial/latest; use --resume or a new directory")
    if args.resume and not initial_path.is_file():
        raise ValueError("resume requires preserved v2 initial.pt in checkpoint directory")
    # Initial parameters must be captured before any optimizer update, never overwritten.
    if not args.resume:
        immutable_torch_save(checkpoint_data(model, metadata), initial_path)
    else:
        _, initial_data = load_checkpoint(initial_path, "cpu")
        if initial_data["metadata"]["games"] != 0 or initial_data["metadata"]["updates"] != 0:
            raise ValueError("initial checkpoint is not a pre-learning model")
        if metadata.get("initialSha256") != sha256_file(initial_path):
            raise ValueError("resume initial checkpoint lineage mismatch")
    model.eval()
    snapshots = []
    for path in metadata.get("snapshotPool", [])[-3:]:
        snapshot, _ = load_checkpoint(path, "cpu")
        snapshots.append((str(Path(path).resolve()), snapshot))
    modes = Counter(metadata["modeCounts"])
    decisions = Counter(metadata["decisionCounts"])
    phases = Counter(metadata["phaseCounts"])
    workers = []
    start_games = metadata["games"]
    goal = start_games + args.games
    metadata.update(
        status="running", targetGames=goal,
        batchSize=args.batch_size, warmup=args.warmup, workers=args.workers,
        updatesPerGame=args.updates_per_game,
        epsilon=args.epsilon, learningRate=optimizer.param_groups[0]["lr"],
        mixedTeammateProbability=args.mixed_teammate_probability, maxActions=args.max_actions,
        saveEvery=args.save_every, logEvery=args.log_every,
        continuationProbability=1.0, roundResetMode="continue",
        initialCheckpoint=str(initial_path.resolve()), initialSha256=sha256_file(initial_path),
    )
    metadata.pop("error", None)
    if cadence_changes:
        override = {
            "games": start_games, "changes": cadence_changes,
            "snapshotRefreshChanged": "saveEvery" in cadence_changes,
        }
        metadata.setdefault("resumeCadenceOverrides", []).append(override)
        changes = ", ".join(
            f"{key}: {value['from'] if value['from'] is not None else 'unsaved'} -> {value['to']}"
            for key, value in cadence_changes.items()
        )
        print(f"resume cadence override at games={start_games}: {changes}; "
              "snapshot refresh follows saveEvery", flush=True)
    writer = None
    started = time.monotonic()
    loss = None

    def save(snapshot=False):
        metadata.update(
            modeCounts=dict(modes), decisionCounts=dict(decisions), phaseCounts=dict(phases),
            inFlightEpisodesDiscardedOnResume=sum(worker["state"] is not None for worker in workers),
        )
        if snapshot:
            path = args.checkpoint_dir / f"snapshot-{metadata['games']:09d}.pt"
            if not path.exists():
                atomic_torch_save(checkpoint_data(model, metadata), path)
            old, _ = load_checkpoint(path, "cpu")
            if str(path.resolve()) not in [item[0] for item in snapshots]:
                snapshots.append((str(path.resolve()), old))
            while len(snapshots) > 3:
                discarded, _ = snapshots.pop(0)
                owned = Path(discarded)
                if owned.parent == args.checkpoint_dir.resolve() and owned.name.startswith("snapshot-"):
                    owned.unlink(missing_ok=True)
        metadata["snapshotPool"] = [path for path, _ in snapshots]
        game = metadata["games"]
        if game == 1000 or (game > 0 and game % 10000 == 0) or metadata["status"] == "completed":
            milestone = args.checkpoint_dir / f"milestone-{game:09d}.pt"
            if not milestone.exists():
                immutable_torch_save(checkpoint_data(model, metadata), milestone)
        atomic_torch_save(checkpoint_data(model, metadata, optimizer, replay, rng),
                          args.checkpoint_dir / "latest.pt")

    def launch(worker):
        episode = metadata["nextEpisode"]
        metadata["nextEpisode"] += 1
        mode = ("selfplay", "selfplay", "heuristic", "snapshot")[episode % 4]
        old = rng.choice(snapshots)[1] if mode == "snapshot" and snapshots else None
        if mode == "snapshot" and old is None:
            mode = "heuristic_fallback"
        team = None if mode == "selfplay" else rng.randrange(2)
        heuristic = set()
        if team is not None and rng.random() < args.mixed_teammate_probability:
            heuristic.add(rng.choice([team, team + 2, team + 4]))
        continuation = worker["finished"]
        state = worker["env"].state(
            "reset", seed=(metadata["seed"] + episode) & 0xFFFFFFFF,
            **({"continue": True} if continuation else {}),
        )
        if state["terminal"] or state["events"]:
            raise RuntimeError("reset must return a live state and empty new history")
        metadata["continuedRounds"] += int(continuation)
        worker.update(
            state=state, events=[], decisions=[], steps=0, finished=False,
            mode=f"mixed_{mode}" if heuristic else mode, opponent=old,
            team=team, heuristic=heuristic,
        )

    def finish(worker):
        nonlocal loss
        replay.add_episode(worker["events"], worker["decisions"], worker["state"]["result"]["winner"])
        decisions.update(decision[4] for decision in worker["decisions"])
        metadata["learnerAskDecisions"] = decisions["ask"]
        metadata["learnerReturnDecisions"] = decisions["return"]
        metadata["transitionsCollected"] += len(worker["decisions"])
        metadata["games"] += 1
        modes[worker["mode"]] += 1
        metadata["mixedTeamGames"] += int(bool(worker["heuristic"]))
        if len(replay) >= args.warmup:
            for _ in range(args.updates_per_game):
                loss = update_model(model, optimizer, replay, args.batch_size, rng, device)
                metadata["updates"] += 1
                writer.add_scalar("foundation/training/loss", loss, metadata["updates"])
        writer.add_scalar("foundation/training/gradient_updates", metadata["updates"], metadata["games"])
        for key in ("learnerAskDecisions", "learnerReturnDecisions", "transitionsCollected",
                    "maxObservedHistoryLength"):
            writer.add_scalar(f"foundation/coverage/{key}", metadata[key], metadata["games"])
        writer.add_scalar("foundation/training/replay_size", len(replay), metadata["games"])
        writer.add_scalar("foundation/training/public_history_bytes", replay.history_bytes, metadata["games"])
        worker.update(state=None, finished=True, decisions=[], events=[], opponent=None)

    try:
        writer = SummaryWriter(str(args.tensorboard_dir))
        writer.add_text("foundation/methods", json.dumps(METHODS, indent=2), metadata["games"])
        for _ in range(min(args.workers, args.games)):
            workers.append(dict(env=NodeEnvironment(max_actions=args.max_actions), state=None, finished=False))
        save()
        while metadata["games"] < goal:
            remaining = goal - metadata["games"]
            active = sum(worker["state"] is not None for worker in workers)
            for worker in workers:
                if worker["state"] is None and active < remaining:
                    launch(worker)
                    active += 1
            current = []
            for worker in workers:
                state = worker["state"]
                if state is None:
                    continue
                if worker["steps"] >= args.max_actions:
                    raise RuntimeError("training max-actions exceeded; no truncated reward")
                player = state["player"]
                is_learner = player not in worker["heuristic"] and (
                    worker["team"] is None or player % 2 == worker["team"])
                if is_learner:
                    current.append(worker)
                    continue
                if worker["opponent"] is not None and player not in worker["heuristic"]:
                    index = int(np.argmax(score_many(worker["opponent"], [learner_job(worker)])[0]))
                    next_state = worker["env"].state("step", index=index)
                else:
                    next_state = worker["env"].state("heuristic")
                worker["steps"] += 1
                phases[state["phase"]] += 1
                append_events(worker, next_state)
                metadata["maxObservedHistoryLength"] = max(
                    metadata["maxObservedHistoryLength"], len(worker["events"]))
                worker["state"] = next_state
            if current:
                scores = score_many(model, [learner_job(worker) for worker in current])
                for worker, values in zip(current, scores):
                    index = choose_action(values, args.epsilon, rng)
                    record_decision(worker, index)
                    phases[worker["state"]["phase"]] += 1
                    next_state = worker["env"].state("step", index=index)
                    append_events(worker, next_state)
                    metadata["maxObservedHistoryLength"] = max(
                        metadata["maxObservedHistoryLength"], len(worker["events"]))
                    worker["steps"] += 1
                    worker["state"] = next_state
            for worker in workers:
                if worker["state"] is not None and worker["state"]["terminal"]:
                    finish(worker)
                    game = metadata["games"]
                    if game % args.log_every == 0 or game == goal or game == start_games + 1:
                        speed = (game - start_games) / max(time.monotonic() - started, 1e-9)
                        print(f"games={game} updates={metadata['updates']} loss={loss} "
                              f"games/sec={speed:.3f} asks={metadata['learnerAskDecisions']} "
                              f"returns={metadata['learnerReturnDecisions']} "
                              f"collected={metadata['transitionsCollected']} "
                              f"maxHistory={metadata['maxObservedHistoryLength']} modes={dict(modes)}", flush=True)
                    if game % args.save_every == 0:
                        save(snapshot=True)
        metadata["status"] = "completed"
        save(snapshot=True)
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = str(error)
        save()
        raise
    finally:
        for worker in workers:
            worker["env"].close()
        if writer:
            writer.close()
    return 0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_checkpoint(source, destination):
    """Open one inode of an atomically replaced latest file; publish an immutable copy."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("freeze destination must differ from live checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        with source.open("rb") as src, temporary.open("xb") as target:
            shutil.copyfileobj(src, target)
            target.flush()
            os.fsync(target.fileno())
        digest = sha256_file(temporary)
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError("immutable frozen checkpoint already exists with different contents")
        else:
            # link is atomic and cannot replace a previous frozen checkpoint.
            os.link(temporary, destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def load_policy(path, device="cpu", *, allow_strategic=False):
    data = torch.load(path, map_location="cpu", weights_only=True)
    version = data.get("metadata", {}).get("encodingVersion")
    if version == 2:
        return (*load_checkpoint(path, device), 2)
    if version == 1:
        from daguai_rl import load_checkpoint as load_v1
        return (*load_v1(path, device), 1)
    if version == 3 and allow_strategic:
        from team_ppo import load_checkpoint as load_ppo
        return (*load_ppo(path, device), 3)
    raise ValueError("unsupported checkpoint encoding")


def policy_index(model, version, state, events):
    if version == 2:
        values = score_many(model, [(
            state["observation"], [action["features"] for action in state["actions"]],
            relative_history(events, state["player"]),
        )])[0]
        return int(np.argmax(values))
    from daguai_rl import score_many as score_v1
    indices = [i for i, action in enumerate(state["actions"]) if action["kind"] != "ask"]
    if not indices:
        raise ValueError("v1 has no non-ask legal actions")
    values = score_v1(model, [(
        state["observation"], [state["actions"][i]["features"][:71] for i in indices],
    )])[0]
    return indices[int(np.argmax(values))]


def paired_bootstrap(values, seed, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires finite nonempty pair values")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 100):
        indices = rng.integers(0, len(values), (min(100, samples - start), len(values)))
        means.extend(values[indices].mean(axis=1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def rules_fingerprint():
    names = ("engine.js", "policy.js", "foundation-policy.js", "training/foundation-env.js")
    return {
        name: sha256_file(ROOT.parent / name)
        for name in names if (ROOT.parent / name).is_file()
    }


def protocol(games, seed, mode, max_actions, fingerprint=None):
    return {
        "games": games, "seed": seed, "mode": mode, "maxActions": max_actions,
        "rules": rules_fingerprint() if fingerprint is None else fingerprint,
        "environmentEncoding": SIZES, "rounds": "fresh",
        "opponent": "engine heuristic with its public asking",
        "seatPolicy": "team + 2*(seed % 3)" if mode == "heuristic" else "all three team seats",
    }


def evaluate_checkpoint(checkpoint, games=200, seed=1000000000, mode="model",
                        max_actions=6000, device="cpu", env_factory=None):
    validate_evaluation(games, seed)
    if mode not in ("model", "heuristic"):
        raise ValueError("unknown evaluation teammate mode")
    digest = sha256_file(checkpoint)
    model, data, version = load_policy(checkpoint, device)
    metadata = data["metadata"]
    del data
    evaluation_protocol = protocol(games, seed, mode, max_actions)
    totals = Counter(wins=0, losses=0, draws=0)
    pairs = []
    env = (env_factory or NodeEnvironment)(max_actions=max_actions)
    try:
        for pair in range(games // 2):
            deal_seed = seed + pair
            record = dict(seed=deal_seed, scores=[], outcomes=[], actions=[], learnerSeats=[])
            for team in (0, 1):
                seats = [team + 2 * (deal_seed % 3)] if mode == "heuristic" else [team, team + 2, team + 4]
                state = env.state("reset", seed=deal_seed)
                if state["terminal"] or state["events"]:
                    raise ValueError("evaluation reset must be live with empty history")
                worker = dict(events=[])
                steps = 0
                while not state["terminal"]:
                    if steps >= max_actions:
                        raise RuntimeError(f"evaluation seed={deal_seed}: action limit; no rewarded truncation")
                    if state["player"] in seats:
                        state = env.state("step", index=policy_index(model, version, state, worker["events"]))
                    else:
                        state = env.state("heuristic")
                    append_events(worker, state)
                    steps += 1
                reward = team_reward(team, state["result"]["winner"])
                outcome = "wins" if reward > 0 else "losses" if reward < 0 else "draws"
                totals[outcome] += 1
                record["scores"].append((reward + 1) / 2)
                record["outcomes"].append(outcome)
                record["actions"].append(steps)
                record["learnerSeats"].append(seats)
            pairs.append(record)
    finally:
        env.close()
    if sha256_file(checkpoint) != digest:
        raise RuntimeError("checkpoint changed during evaluation")
    pair_scores = [sum(pair["scores"]) / 2 for pair in pairs]
    return {
        "createdAt": now(), "checkpoint": str(Path(checkpoint).resolve()),
        "checkpointSha256": digest, "metadata": metadata,
        "policyEncoding": version, "protocol": evaluation_protocol,
        "protocolFingerprint": hashlib.sha256(json.dumps(evaluation_protocol, sort_keys=True).encode()).hexdigest(),
        "games": games, "seed": seed, "teammates": mode, **dict(totals),
        "score": float(np.mean(pair_scores)),
        "confidence95": paired_bootstrap(pair_scores, seed ^ 0x9E3779B9),
        "confidenceMethod": "paired-seed percentile bootstrap (10000 resamples)",
        "pairedSeeds": pairs, "learnerSeats": [pair["learnerSeats"] for pair in pairs],
        "scope": "Fresh rounds only: tribute/return strategy is not measured. "
                 "Heuristic performance is not a claim of expert strength.",
        "promotion": "never automatic",
    }


def compare_reports(current, baseline):
    if current["protocol"] != baseline["protocol"]:
        raise ValueError("paired comparison protocol/rules/seed mismatch")
    a, b = current["pairedSeeds"], baseline["pairedSeeds"]
    if len(a) != len(b) or not a:
        raise ValueError("paired comparison length mismatch")
    deltas = []
    for first, second in zip(a, b):
        if (first["seed"] != second["seed"] or first["learnerSeats"] != second["learnerSeats"]
                or len(first["scores"]) != 2 or len(second["scores"]) != 2):
            raise ValueError("paired comparison seed/learner-seat mismatch")
        deltas.append((sum(first["scores"]) - sum(second["scores"])) / 2)
    confidence = paired_bootstrap(deltas, current["seed"] ^ 0x85EBCA6B)
    status = "improved" if confidence[0] > 0 else "regressed" if confidence[1] < 0 else "not established"
    return {
        "delta": float(np.mean(deltas)), "confidence95": confidence, "assessment": status,
        "baselineSha256": baseline["checkpointSha256"],
        "method": "paired difference across identical seeds and swapped learner sides",
    }


def tactical_audit(checkpoints):
    script = ROOT / "foundation-scenarios.js"
    if not script.exists():
        return {"available": False, "notice": "No tactical fixtures supplied"}
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30, check=True)
    fixtures = json.loads(result.stdout)
    if isinstance(fixtures, dict):
        fixtures = fixtures.get("scenarios", fixtures.get("fixtures"))
    if not isinstance(fixtures, list):
        raise ValueError("tactical fixtures must be an array")
    records = []
    for label, checkpoint in checkpoints.items():
        model, _, version = load_policy(checkpoint)
        for fixture in fixtures:
            actions = fixture["actions"]
            features = [action.get("features", action.get("action")) for action in actions]
            if version == 2:
                indices = list(range(len(actions)))
                values = score_many(model, [(fixture["observation"], features, fixture["history"])])[0]
            else:
                from daguai_rl import score_many as score_v1
                indices = [i for i, action in enumerate(actions) if action.get("kind") != "ask"]
                values = score_v1(model, [(fixture["observation"], [features[i][:71] for i in indices])])[0]
            order = np.argsort(-values, kind="stable")[:3]
            records.append({
                "policy": label, "scenario": fixture["name"], "description": fixture.get("description"),
                "top3": [dict(label=actions[indices[i]]["label"], score=float(values[i]),
                              referenceAgreement=bool(actions[indices[i]].get("expected", False))) for i in order],
            })
    return {
        "available": True, "records": records,
        "notice": "Reference agreement only; these fixtures are not proven optimal strategy or explanations.",
    }


def now():
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid):
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def evaluation_milestones(target, interval=10000, start=0):
    if type(interval) is not int or interval < 1 or not 0 <= start < target:
        raise ValueError("invalid evaluation interval or starting game count")
    return sorted(set(
        [min(start + min(1000, interval), target), target]
        + list(range(start + interval, target + 1, interval))
    ))


def next_milestone(completed, target, interval=10000, start=0):
    previous = max(completed, default=start)
    return next((value for value in evaluation_milestones(target, interval, start)
                 if value > previous), target)


def monitor(args, evaluator=None, sleep=time.sleep, alive=process_alive):
    """Separate serial CPU evaluator. Failures are durable and always propagate."""
    from torch.utils.tensorboard import SummaryWriter

    evaluator = evaluator or evaluate_checkpoint
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.checkpoint.parent / ".foundation-monitor.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another monitor holds this checkpoint lock") from error
        writer = None
        latest_path = args.output_dir / "latest.json"
        state = {
            "status": "waiting", "stage": "exploratory validation", "updatedAt": now(),
            "targetGames": args.target_games, "currentGames": 0, "completedMilestones": [],
            "validationEvery": args.validation_every, "startGames": args.start_games,
            "initialKind": args.initial_kind,
            "waitingForResume": False,
            "methods": METHODS, "scope": "fresh rounds only; no return-round strength or expert claim",
            "validationSeed": args.seed, "finalSeed": args.final_seed,
            "automaticFinalEvaluation": False,
            "finalEvaluation": "separate preselected release plan required; never run by monitor",
            "checkpoint": str(args.checkpoint.resolve()),
        }
        try:
            if latest_path.exists():
                previous = json.loads(latest_path.read_text())
                identity = ("targetGames", "validationSeed", "finalSeed", "checkpoint")
                if any(previous.get(key) != state[key] for key in identity):
                    raise ValueError("monitor resume configuration mismatch")
                defaults = {"validationEvery": 10000, "startGames": 0, "initialKind": "random"}
                if any(previous.get(key, default) != state[key] for key, default in defaults.items()):
                    raise ValueError("monitor resume evaluation cadence or baseline kind mismatch")
                state.update(previous)
                if state["status"] == "completed":
                    return 0
            writer = SummaryWriter(str(args.tensorboard_dir))
            frozen_dir = args.output_dir / "frozen"
            initial = frozen_dir / "initial.pt"
            baseline = frozen_dir / "baseline-v1.pt"
            while not args.initial.exists():
                if not alive(args.trainer_pid):
                    raise RuntimeError("trainer exited before random initial checkpoint was available")
                atomic_json_save(state, latest_path)
                sleep(args.poll_seconds)
            state["initialSha256"] = freeze_checkpoint(args.initial, initial)
            state["v1Sha256"] = freeze_checkpoint(args.baseline, baseline)
            initial_model, initial_data = load_checkpoint(initial, "cpu")
            del initial_model
            initial_metadata = initial_data["metadata"]
            if args.initial_kind == "random":
                if initial_metadata["games"] != 0 or initial_metadata["updates"] != 0:
                    raise ValueError("initial baseline must be random pre-learning checkpoint")
                state["initialDescription"] = "random pre-learning policy"
            else:
                if (initial_metadata["games"] != args.start_games
                        or (initial_metadata.get("algorithm") == "team-ppo-v1"
                            and initial_metadata.get("ppoGames") != 0)):
                    raise ValueError("warm-start baseline must precede PPO learning at --start-games")
                state["initialDescription"] = "transferred self-trained policy before PPO, not random"
            state["initialGames"] = initial_metadata["games"]
            del initial_data
            _, _, baseline_version = load_policy(baseline)
            if baseline_version != 1:
                raise ValueError("--baseline must be the preserved v1 checkpoint")
            baseline_reports = {}

            def assessment(frozen, games_count, stage, eval_games, seed, milestone):
                reports = {}
                for mode in ("model", "heuristic"):
                    comparisons = {}
                    for label, path in (("initial", initial), ("v1", baseline)):
                        key = (stage, mode, label)
                        if key not in baseline_reports:
                            baseline_path = args.output_dir / f"{stage}-{label}-{mode}.json"
                            if baseline_path.exists():
                                report = json.loads(baseline_path.read_text())
                                if (report["checkpointSha256"] != sha256_file(path)
                                        or report["protocol"] != protocol(eval_games, seed, mode, args.max_actions)):
                                    raise ValueError("cached baseline protocol or checkpoint mismatch")
                            else:
                                report = evaluator(path, games=eval_games, seed=seed, mode=mode,
                                                   max_actions=args.max_actions, device="cpu")
                                atomic_json_save(report, baseline_path)
                            baseline_reports[key] = report
                    report = evaluator(frozen, games=eval_games, seed=seed, mode=mode,
                                       max_actions=args.max_actions, device="cpu")
                    for label in ("initial", "v1"):
                        comparisons[label] = compare_reports(report, baseline_reports[(stage, mode, label)])
                    report["comparisons"] = comparisons
                    report["stage"] = stage
                    prefix = f"foundation/strength/{stage}/{mode}"
                    for metric, value in (
                        ("score", report["score"]), ("win_rate", report["wins"] / report["games"]),
                        ("draw_rate", report["draws"] / report["games"]),
                        ("confidence_bounds/lower", report["confidence95"][0]),
                        ("confidence_bounds/upper", report["confidence95"][1]),
                    ):
                        writer.add_scalar(f"{prefix}/{metric}", value, games_count)
                    for label, comparison in comparisons.items():
                        writer.add_scalar(f"{prefix}/baseline_delta/{label}", comparison["delta"], games_count)
                        for bound, value in zip(("lower", "upper"), comparison["confidence95"]):
                            writer.add_scalar(f"{prefix}/baseline_delta/{label}_{bound}", value, games_count)
                    report_path = args.output_dir / f"{stage}-{games_count:09d}-{mode}.json"
                    atomic_json_save(report, report_path)
                    reports[mode] = {
                        key: report[key] for key in ("score", "wins", "losses", "draws", "games",
                                                    "confidence95", "comparisons", "checkpointSha256")
                    }
                    reports[mode]["report"] = str(report_path.resolve())
                entry = {
                    "status": "evaluated", "stage": stage, "checkpointGames": games_count,
                    "milestone": milestone, "reports": reports, "updatedAt": now(),
                    "tacticalAudit": tactical_audit({"current": frozen, "initial": initial, "v1": baseline}),
                    "interpretation": "exploratory, repeated validation checks; not final confirmation"
                    if stage == "validation" else "held-out final confirmation; fresh rounds only",
                }
                atomic_json_save(entry, args.output_dir / f"{stage}-{games_count:09d}-summary.json")
                with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                writer.flush()
                state.update(
                    status="watching", stage=entry["interpretation"], currentGames=games_count,
                    reports=reports, latestAssessment=entry,
                )
                return entry

            while True:
                state["updatedAt"] = now()
                if not args.checkpoint.exists():
                    if not alive(args.trainer_pid):
                        raise RuntimeError("trainer exited before any v2 checkpoint was available")
                    atomic_json_save(state, latest_path)
                    sleep(args.poll_seconds)
                    continue
                # Freeze once before reading metadata. No evaluator ever opens the live learner file.
                pending = frozen_dir / f".pending-{uuid.uuid4().hex}.pt"
                try:
                    digest = freeze_checkpoint(args.checkpoint, pending)
                    model, data = load_checkpoint(pending, "cpu")
                    del model
                    metadata = data["metadata"]
                    del data
                    count = metadata["games"]
                    state.update(currentGames=count, trainerStatus=metadata.get("status"),
                                 trainerTargetGames=metadata.get("targetGames"),
                                 methods=metadata.get("methods", METHODS),
                                 algorithm=metadata.get("algorithm", "monte-carlo-v2"),
                                 coverage={key: metadata.get(key, 0) for key in (
                                     "learnerAskDecisions", "learnerReturnDecisions",
                                     "transitionsCollected", "maxObservedHistoryLength",
                                 )})
                    if metadata.get("status") in ("failed", "interrupted"):
                        raise RuntimeError(f"trainer {metadata['status']}: {metadata.get('error', '')}")
                    complete = metadata.get("status") == "completed"
                    state["waitingForResume"] = False
                    if complete and count < args.target_games:
                        if args.trainer_pid is None or not alive(args.trainer_pid):
                            raise RuntimeError(f"trainer completed at {count}, below target {args.target_games}")
                        state.update(status="waiting", stage="awaiting resumed checkpoint",
                                     waitingForResume=True)
                    elif state.get("stage") == "awaiting resumed checkpoint":
                        state.update(status="watching", stage="exploratory validation")
                    milestone = next_milestone(state["completedMilestones"], args.target_games,
                                               args.validation_every, args.start_games)
                    while count >= milestone and milestone not in state["completedMilestones"]:
                        archived = args.checkpoint.parent / f"milestone-{milestone:09d}.pt"
                        source, assessed_count = pending, count
                        assessed_digest = digest
                        if archived.exists():
                            archived_model, archived_data = load_checkpoint(archived, "cpu")
                            del archived_model
                            if archived_data["metadata"]["games"] != milestone:
                                raise ValueError("archived milestone has mismatched game count")
                            del archived_data
                            source, assessed_count = archived, milestone
                            assessed_digest = sha256_file(archived)
                        frozen = frozen_dir / f"checkpoint-{assessed_count:09d}-{assessed_digest[:12]}.pt"
                        freeze_checkpoint(source, frozen)
                        state.update(status="evaluating", stage="exploratory validation")
                        atomic_json_save(state, latest_path)
                        assessment(frozen, assessed_count, "validation", args.games, args.seed, milestone)
                        crossed = [value for value in evaluation_milestones(
                            args.target_games, args.validation_every, args.start_games
                        ) if value <= assessed_count]
                        missed = [value for value in crossed if value < assessed_count
                                  and value not in state["completedMilestones"]]
                        state["skippedMilestones"] = sorted(set(state.get("skippedMilestones", []) + missed))
                        state["milestoneNotice"] = (
                            "Only available frozen checkpoints are evaluated; skipped milestones "
                            "are not claimed as separate assessments."
                        )
                        state["completedMilestones"] = sorted(set(
                            state["completedMilestones"] + [min(assessed_count, args.target_games)]))
                        state["currentGames"] = count
                        atomic_json_save(state, latest_path)
                        milestone = next_milestone(state["completedMilestones"], args.target_games,
                                                   args.validation_every, args.start_games)
                    if count >= args.target_games:
                        state.update(status="completed", stage="validation complete; final holdout not opened",
                                     automaticFinalEvaluation=False, updatedAt=now())
                        atomic_json_save(state, latest_path)
                        return 0
                    if not alive(args.trainer_pid):
                        # A serial assessment can outlast the trainer. Inspect its final
                        # atomic replacement before treating a now-dead PID as failure.
                        if sha256_file(args.checkpoint) != digest:
                            continue
                        raise RuntimeError(f"trainer exited at {count} before target {args.target_games}")
                finally:
                    pending.unlink(missing_ok=True)
                atomic_json_save(state, latest_path)
                sleep(args.poll_seconds)
        except BaseException as error:
            state.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error), updatedAt=now(),
            )
            atomic_json_save(state, latest_path)
            raise
        finally:
            if writer:
                writer.close()


def validate_evaluation(games, seed):
    if type(games) is not int or games < 2 or games % 2:
        raise ValueError("evaluation games must be a positive even number")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF or seed + games // 2 - 1 > 0xFFFFFFFF:
        raise ValueError("evaluation seed range must fit uint32")


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--games", type=int, default=100000, help="ADDITIONAL completed games")
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--checkpoint-dir", "--output-dir", dest="checkpoint_dir", type=Path, default=CHECKPOINTS)
    training.add_argument("--resume", type=Path)
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--updates-per-game", type=int, default=4,
                          help="gradient minibatches per completed game after warmup")
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--epsilon", type=float, default=0.15)
    training.add_argument("--mixed-teammate-probability", type=float, default=0.5)
    training.add_argument("--save-every", type=int, default=1000)
    training.add_argument("--log-every", type=int, default=100)
    training.add_argument("--replay-size", type=int)
    training.add_argument("--warmup", type=int, default=64)
    training.add_argument("--max-actions", type=int, default=6000)
    training.add_argument("--tensorboard-dir", type=Path, default=ROOT / "runs" / "main" / "foundation-train")
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    inference.add_argument("--stdio", action="store_true", help="accepted for compatibility; always uses JSON-lines")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--games", type=int, default=200)
    evaluation.add_argument("--seed", type=int, default=1000000000)
    evaluation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    evaluation.add_argument("--threads", type=int, default=1, help="CPU Torch threads; default 1")
    evaluation.add_argument("--teammates", choices=("model", "heuristic"), default="model")
    evaluation.add_argument("--max-actions", type=int, default=6000)
    evaluation.add_argument("--report", type=Path, default=ROOT / "foundation" / "progress" / "evaluation.json")
    watching = commands.add_parser("monitor")
    watching.add_argument("--checkpoint", type=Path, default=CHECKPOINTS / "latest.pt")
    watching.add_argument("--baseline", type=Path, default=ROOT / "progress" / "baseline-000000080.pt")
    watching.add_argument("--initial", type=Path, default=CHECKPOINTS / "initial.pt")
    watching.add_argument("--initial-kind", choices=("random", "warm-start"), default="random")
    watching.add_argument("--start-games", type=int, default=0)
    watching.add_argument("--validation-every", type=int, default=10000)
    watching.add_argument("--target-games", type=int, default=100000)
    watching.add_argument("--games", type=int, default=200)
    watching.add_argument("--final-games", type=int, default=2000,
                          help="legacy configuration only; monitor never opens a final holdout")
    watching.add_argument("--seed", type=int, default=1000000000)
    watching.add_argument("--final-seed", type=int, default=2000000000,
                          help="legacy configuration only; use a separate registered release evaluation")
    watching.add_argument("--max-actions", type=int, default=6000)
    watching.add_argument("--tensorboard-dir", type=Path, default=ROOT / "runs" / "main" / "foundation-eval")
    watching.add_argument("--output-dir", type=Path, default=ROOT / "foundation" / "progress")
    watching.add_argument("--poll-seconds", type=float, default=30)
    watching.add_argument("--trainer-pid", type=int)
    args = parser.parse_args(argv)
    args.explicit = {word.split("=", 1)[0][2:].replace("-", "_") for word in argv if word.startswith("--")}
    try:
        for name in ("games", "workers", "batch_size", "updates_per_game", "save_every", "log_every", "replay_size",
                     "warmup", "max_actions", "target_games", "trainer_pid", "validation_every"):
            value = getattr(args, name, None)
            if value is not None and value < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.command == "train":
            if args.workers > 128:
                raise ValueError("workers must be <=128")
            if args.replay_size is not None and args.replay_size > 10000:
                raise ValueError("replay-size must be <=10000")
            if not 0 <= args.seed <= 0xFFFFFFFF:
                raise ValueError("seed must fit uint32")
            for name in ("epsilon", "mixed_teammate_probability"):
                if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
                    raise ValueError(f"{name} must be finite and in [0,1]")
            if args.learning_rate is not None and (
                    not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
                raise ValueError("learning-rate must be finite and positive")
        if args.command in ("evaluate", "monitor"):
            validate_evaluation(args.games, args.seed)
        if args.command == "evaluate" and not 1 <= args.threads <= 128:
            raise ValueError("threads must be 1..128")
        if args.command == "evaluate" and args.report.resolve() == args.checkpoint.resolve():
            raise ValueError("report must not overwrite checkpoint")
        if args.command == "monitor":
            if not 0 <= args.start_games < args.target_games:
                raise ValueError("start-games must be nonnegative and below target-games")
            validate_evaluation(args.final_games, args.final_seed)
            if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
                raise ValueError("poll-seconds must be finite and positive")
            validation_range = (args.seed, args.seed + args.games // 2 - 1)
            final_range = (args.final_seed, args.final_seed + args.final_games // 2 - 1)
            if max(validation_range[0], final_range[0]) <= min(validation_range[1], final_range[1]):
                raise ValueError("final and validation seeds must be disjoint")
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        if args.command == "train":
            return train(args)
        if args.command == "infer":
            torch.set_num_threads(1)
            model, _ = load_checkpoint(args.checkpoint, device_for(args.device))
            serve(model)
            return 0
        if args.command == "evaluate":
            torch.set_num_threads(args.threads)
            report = evaluate_checkpoint(
                args.checkpoint, args.games, args.seed, args.teammates,
                args.max_actions, device_for(args.device),
            )
            atomic_json_save(report, args.report)
            print(f"score={report['score']:.4f} paired95%CI={report['confidence95']} report={args.report}")
            return 0
        return monitor(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"foundation {args.command} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
