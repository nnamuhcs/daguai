"""Shared action-value model, environment bridge, and checkpoint utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import select
import subprocess
import time
import uuid

import numpy as np
import torch
from torch import nn

ENCODING_VERSION = 1
OBSERVATION_SIZE = 922
ACTION_SIZE = 71
SIZES = {
    "encodingVersion": ENCODING_VERSION,
    "observationSize": OBSERVATION_SIZE,
    "actionSize": ACTION_SIZE,
}
ROOT = Path(__file__).resolve().parent


def device_for(name: str) -> torch.device:
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda" if name != "cpu" and torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def finite_vector(value, size: int, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be an array of {size} numbers")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        array = raw.astype(np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain exactly {size} finite numbers")
    return array


def feature_matrix(value, *, maximum: int | None = None) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)) or not len(value):
        raise ValueError("actions must be a nonempty array")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"at most {maximum} actions are allowed")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("action features must contain numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        array = raw.astype(np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_SIZE or not np.isfinite(array).all():
        raise ValueError(f"actions must contain arrays of {ACTION_SIZE} finite numbers")
    return array


class ActionValue(nn.Module):
    def __init__(self, dimensions=(256, 128)):
        super().__init__()
        if (not isinstance(dimensions, (list, tuple)) or not 1 <= len(dimensions) <= 4
                or any(type(width) is not int or not 1 <= width <= 4096 for width in dimensions)):
            raise ValueError("invalid model dimensions")
        self.dimensions = tuple(dimensions)
        layers = []
        previous = OBSERVATION_SIZE + ACTION_SIZE
        for width in self.dimensions:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if (observations.ndim != 2 or actions.ndim != 2
                or observations.shape[1] != OBSERVATION_SIZE
                or actions.shape[1] != ACTION_SIZE
                or observations.shape[0] != actions.shape[0]):
            raise ValueError("model requires matching [N,922] observations and [N,71] actions")
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            raise ValueError("model input contains non-finite values")
        return self.network(torch.cat((observations, actions), dim=1)).squeeze(-1)


@torch.no_grad()
def score_many(model: ActionValue, states, *, chunk_size: int = 4096):
    """Batch variable action sets without storing them in the replay buffer."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    device = next(model.parameters()).device
    results = [np.empty(len(actions), dtype=np.float32) for _, actions in states]
    pending_obs, pending_actions, destinations = [], [], []

    def flush():
        if not pending_actions:
            return
        observations = torch.from_numpy(np.stack(pending_obs)).to(device)
        actions = torch.from_numpy(np.stack(pending_actions)).to(device)
        scores = model(observations, actions).detach().cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("model produced non-finite action scores")
        for (state_index, action_index), score in zip(destinations, scores):
            results[state_index][action_index] = score
        pending_obs.clear()
        pending_actions.clear()
        destinations.clear()

    for state_index, (observation, actions) in enumerate(states):
        observation = finite_vector(observation, OBSERVATION_SIZE, "observation")
        actions = feature_matrix(actions)
        for action_index, features in enumerate(actions):
            pending_obs.append(observation)
            pending_actions.append(features)
            destinations.append((state_index, action_index))
            if len(pending_actions) == chunk_size:
                flush()
    flush()
    return results


def choose_action(scores, epsilon: float, rng: random.Random) -> int:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite vector")
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between zero and one")
    return rng.randrange(len(scores)) if rng.random() < epsilon else int(np.argmax(scores))


def team_reward(player: int, winner) -> float:
    if type(player) is not int or not 0 <= player < 6:
        raise ValueError("invalid player")
    if winner is None:
        return 0.0
    if type(winner) is not int or winner not in (0, 1):
        raise ValueError("terminal winner must be 0, 1, or null")
    return 1.0 if player % 2 == winner else -1.0


class NodeEnvironment:
    def __init__(self, script: Path | str | None = None, timeout: float = 60, max_actions: int = 1500):
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self.timeout = timeout
        self.process = subprocess.Popen(
            ["node", str(script or ROOT / "env.js")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            bufsize=0, env={**os.environ, "RL_MAX_ACTIONS": str(max_actions)},
        )
        self._buffer = b""
        try:
            response = self.request("hello")
            check_encoding(response)
        except BaseException:
            self.close()
            raise

    def request(self, op: str, **kwargs):
        if self.process.poll() is not None:
            raise RuntimeError(f"Node environment exited with code {self.process.returncode}")
        payload = (json.dumps({"op": op, **kwargs}, allow_nan=False) + "\n").encode()
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("Node environment input closed") from error
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise RuntimeError(f"Node environment timed out during {op}")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"Node environment closed output during {op}")
            self._buffer += chunk
            if len(self._buffer) > 64 * 1024 * 1024:
                raise RuntimeError("Node environment response exceeds 64 MiB")
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            result = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError(f"invalid Node JSON response during {op}") from error
        if not isinstance(result, dict):
            raise RuntimeError("Node environment response must be an object")
        if "error" in result:
            raise RuntimeError(f"Node environment: {result['error']}")
        return result

    def state(self, op: str, **kwargs):
        result = self.request(op, **kwargs)
        validate_state(result)
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                old_timeout = self.timeout
                self.timeout = min(self.timeout, 2)
                self.request("close")
                self.timeout = old_timeout
                self.process.wait(timeout=2)
            except (Exception, KeyboardInterrupt):
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()


def check_encoding(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("missing encoding metadata")
    for key, expected in SIZES.items():
        if type(metadata.get(key)) is not int or metadata[key] != expected:
            raise ValueError(f"incompatible {key}: expected {expected}, got {metadata.get(key)!r}")


def validate_state(state):
    finite_vector(state.get("observation"), OBSERVATION_SIZE, "observation")
    player = state.get("player")
    if type(player) is not int or not 0 <= player < 6:
        raise ValueError("environment returned invalid player")
    if state.get("phase") not in ("playing", "returning", "over"):
        raise ValueError("environment returned invalid phase")
    if type(state.get("terminal")) is not bool:
        raise ValueError("environment returned invalid terminal flag")
    actions = state.get("actions")
    if not isinstance(actions, list):
        raise ValueError("environment actions must be an array")
    if state["terminal"]:
        if (state["phase"] != "over" or not isinstance(state.get("result"), dict)
                or "winner" not in state["result"]):
            raise ValueError("terminal environment state has no result")
        team_reward(player, state["result"].get("winner"))
    else:
        if state["phase"] == "over" or state.get("result") is not None:
            raise ValueError("nonterminal environment state has inconsistent result")
        if not actions:
            raise ValueError("nonterminal environment state has no legal actions")
    for action in actions:
        if not isinstance(action, dict) or action.get("kind") not in ("play", "pass", "return"):
            raise ValueError("environment returned invalid action kind")
        ids = action.get("ids")
        if not isinstance(ids, list) or any(type(card) is not int or card < 0 for card in ids):
            raise ValueError("environment returned invalid card IDs")
        finite_vector(action.get("features"), ACTION_SIZE, "action features")


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.entries = []
        self.position = 0

    def __len__(self):
        return len(self.entries)

    def add(self, observation, features, reward):
        observation = finite_vector(observation, OBSERVATION_SIZE, "observation").copy()
        features = finite_vector(features, ACTION_SIZE, "features").copy()
        if reward not in (-1, 0, 1):
            raise ValueError("replay reward must be -1, 0, or 1")
        entry = (observation, features, float(reward))
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            self.entries[self.position] = entry
        self.position = (self.position + 1) % self.capacity

    def sample(self, size: int, rng: random.Random, device):
        entries = rng.sample(self.entries, size)
        observations, features, rewards = zip(*entries)
        return (
            torch.from_numpy(np.stack(observations)).to(device),
            torch.from_numpy(np.stack(features)).to(device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
        )

    def state_dict(self):
        if not self.entries:
            return {"capacity": self.capacity, "position": 0, "entries": None}
        observations, features, rewards = zip(*self.entries)
        return {
            "capacity": self.capacity, "position": self.position,
            "entries": (
                torch.from_numpy(np.stack(observations)),
                torch.from_numpy(np.stack(features)),
                torch.tensor(rewards, dtype=torch.float32),
            ),
        }

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError(
                f"resume replay-size {self.capacity} must match checkpoint capacity {state['capacity']}; "
                "omit --replay-size to reuse the saved capacity"
            )
        self.entries = []
        entries = state["entries"]
        if entries is not None:
            observations, features, rewards = [value.cpu().numpy() for value in entries]
            if not len(observations) <= self.capacity or not len(observations) == len(features) == len(rewards):
                raise ValueError("invalid replay checkpoint sizes")
            for observation, feature, reward in zip(observations, features, rewards):
                self.add(observation, feature, float(reward))
        position = state["position"]
        if type(position) is not int or not 0 <= position < self.capacity:
            raise ValueError("invalid replay position")
        self.position = position


def update_model(model, optimizer, replay, batch_size, rng, device):
    observations, actions, targets = replay.sample(batch_size, rng, device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(observations, actions)
    loss = nn.functional.mse_loss(prediction, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite training loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise RuntimeError("non-finite training gradient")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("non-finite model parameters after update")
    return float(loss.detach().cpu())


def atomic_torch_save(value, destination: Path | str):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(value, destination: Path | str):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_data(model, metadata, *, optimizer=None, rng=None, replay=None):
    meta = {"learnReturns": False, **metadata, **SIZES,
            "obsSize": OBSERVATION_SIZE, "modelDimensions": list(model.dimensions)}
    data = {
        "metadata": meta,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    if rng is not None:
        data["rng"] = rng.getstate()
        data["torchRng"] = torch.get_rng_state()
        data["cudaRng"] = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []
    if replay is not None:
        data["replay"] = replay.state_dict()
    return data


def load_checkpoint(path: Path | str, device):
    # weights_only forbids arbitrary pickled globals, including on resume.
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be a dictionary")
    metadata = data.get("metadata")
    check_encoding(metadata)
    metadata.setdefault("learnReturns", False)
    if type(metadata["learnReturns"]) is not bool:
        raise ValueError("checkpoint learnReturns must be a boolean")
    for key in ("games", "updates"):
        if type(metadata.get(key)) is not int or metadata[key] < 0:
            raise ValueError(f"checkpoint has invalid {key}")
    model = ActionValue(metadata.get("modelDimensions"))
    model.load_state_dict(data["state_dict"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint contains non-finite parameters")
    return model.to(device).eval(), data


def restore_training(data, optimizer, rng, replay, device):
    for key in ("optimizer", "rng", "torchRng", "replay"):
        if key not in data:
            raise ValueError(f"checkpoint cannot resume training: missing {key}")
    optimizer.load_state_dict(data["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError("checkpoint contains non-finite optimizer state")
                if key != "step":
                    state[key] = value.to(device)
    rng.setstate(data["rng"])
    torch.set_rng_state(data["torchRng"].cpu())
    if device.type == "cuda" and data.get("cudaRng"):
        if len(data["cudaRng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from resume checkpoint")
        torch.cuda.set_rng_state_all([value.cpu() for value in data["cudaRng"]])
    replay.load_state_dict(data["replay"])


def sha256_file(path: Path | str):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
