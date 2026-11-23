"""Focused CPU tests; Node integration checks run when its adapter is present."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import random
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np
import torch

from daguai_rl import (
    ACTION_SIZE, OBSERVATION_SIZE, ROOT, SIZES, ActionValue, NodeEnvironment,
    ReplayBuffer, atomic_json_save, atomic_torch_save, checkpoint_data, choose_action,
    device_for, load_checkpoint, restore_training, score_many, seed_everything,
    team_reward, update_model, validate_state,
)
from evaluate import paired_bootstrap, promote_checkpoint
from infer import handle_request, serve
from train import arguments as training_arguments, run as train_run
from evaluate import run as evaluate_run


class UnsafeCheckpoint:
    pass


class ToyEnvironment:
    """A deterministic six-decision game isolates Python orchestration tests."""
    instances = []

    def __init__(self, **kwargs):
        self.closed = False
        self.index = 0
        self.seed = 0
        self.reset_requests = []
        self.decisions = []
        self.instances.append(self)

    def state(self, op, **kwargs):
        if op == "reset":
            self.reset_requests.append(dict(kwargs))
            self.index = 0
            self.seed = kwargs["seed"]
        else:
            self.decisions.append({"seed": self.seed, "player": self.index % 6,
                                   "op": op, "round": len(self.reset_requests)})
            self.index += 1
        terminal = self.index == 6
        state = {
            "observation": [self.seed % 10 / 10] + [0] * (OBSERVATION_SIZE - 1),
            "actions": [] if terminal else [
                {"ids": [self.index], "kind": "play", "features": [index] + [0] * (ACTION_SIZE - 1)}
                for index in (0, 1)
            ],
            "player": self.index % 6, "phase": "over" if terminal else "playing",
            "terminal": terminal, "result": {"winner": self.seed % 2} if terminal else None,
        }
        validate_state(state)
        return state

    def close(self):
        self.closed = True


class ReturningToyEnvironment(ToyEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.return_steps = []

    def state(self, op, **kwargs):
        if op == "reset" and kwargs.get("continue"):
            if self.index != 6:
                raise RuntimeError("Cannot continue an unfinished round")
            self.reset_requests.append(dict(kwargs))
            self.seed = kwargs["seed"]
            self.index = -1
            state = {
                "observation": [0] * (OBSERVATION_SIZE - 1) + [1],
                "actions": [{"ids": [index], "kind": "return",
                             "features": [index] + [0] * (ACTION_SIZE - 2) + [1]}
                            for index in (0, 1)],
                "player": 0, "phase": "returning", "terminal": False, "result": None,
            }
            validate_state(state)
            return state
        if self.index == -1:
            self.return_steps.append((op, dict(kwargs)))
        return super().state(op, **kwargs)


class SeatToyEnvironment(ToyEnvironment):
    def state(self, op, **kwargs):
        state = super().state(op, **kwargs)
        state["observation"][-2] = self.seed
        state["observation"][-1] = state["player"] + 1
        return state


class RLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = ROOT / f".test-rl-{uuid.uuid4().hex}"
        cls.directory.mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory)

    def setUp(self):
        seed_everything(17)
        self.rng = random.Random(17)
        self.device = torch.device("cpu")
        self.model = ActionValue((32, 16)).eval()
        self.observation = np.linspace(-1, 1, OBSERVATION_SIZE, dtype=np.float32)
        self.actions = np.zeros((3, ACTION_SIZE), dtype=np.float32)
        self.actions[1, 0] = 1
        self.actions[2, 1] = 1
        self.metadata = {"games": 8, "updates": 3, "seed": 17}

    def test_forward_shape_finite_and_chunked_variable_actions(self):
        scores = score_many(self.model, [(self.observation, self.actions),
                                         (self.observation, self.actions[:1])], chunk_size=2)
        self.assertEqual([len(value) for value in scores], [3, 1])
        self.assertTrue(all(np.isfinite(value).all() for value in scores))
        with torch.no_grad():
            direct = self.model(torch.tensor(np.tile(self.observation, (3, 1))),
                                torch.tensor(self.actions)).numpy()
        np.testing.assert_allclose(scores[0], direct, rtol=1e-6, atol=1e-7)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, OBSERVATION_SIZE - 1), torch.zeros(1, ACTION_SIZE))
        with self.assertRaises(ValueError):
            self.model(torch.full((1, OBSERVATION_SIZE), float("nan")), torch.zeros(1, ACTION_SIZE))
        with self.assertRaises(ValueError):
            score_many(self.model, [(self.observation, [])])

    def test_action_selection_valid_deterministic_and_exploratory(self):
        self.assertEqual(choose_action([0, 2, 1], 0, self.rng), 1)
        choices = {choose_action([0, 2, 1], 1, self.rng) for _ in range(40)}
        self.assertEqual(choices, {0, 1, 2})
        first, second = random.Random(42), random.Random(42)
        self.assertEqual([choose_action([1, 2], .5, first) for _ in range(20)],
                         [choose_action([1, 2], .5, second) for _ in range(20)])
        for scores, epsilon in (([], 0), ([float("nan")], 0), ([1], -1)):
            with self.assertRaises(ValueError):
                choose_action(scores, epsilon, self.rng)

    def replay(self):
        replay = ReplayBuffer(8)
        for index in range(12):
            replay.add(self.observation + index / 100, self.actions[index % 3],
                       team_reward(index % 6, index % 2))
        return replay

    def test_checkpoint_atomic_reload_optimizer_rng_replay_resume(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        replay = self.replay()
        loss = update_model(self.model, optimizer, replay, 4, self.rng, self.device)
        self.assertTrue(np.isfinite(loss))
        path = self.directory / "resume.pt"
        atomic_torch_save(checkpoint_data(self.model, self.metadata, optimizer=optimizer,
                                         rng=self.rng, replay=replay), path)
        restored, data = load_checkpoint(path, self.device)
        np.testing.assert_array_equal(score_many(self.model, [(self.observation, self.actions)])[0],
                                      score_many(restored, [(self.observation, self.actions)])[0])
        expected_loss = update_model(self.model, optimizer, replay, 4, self.rng, self.device)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.5)
        restored_rng = random.Random(999)
        restored_replay = ReplayBuffer(8)
        restore_training(data, restored_optimizer, restored_rng, restored_replay, self.device)
        self.assertEqual(len(restored_replay), 8)
        actual_loss = update_model(restored, restored_optimizer, restored_replay, 4, restored_rng, self.device)
        self.assertEqual(expected_loss, actual_loss)
        np.testing.assert_array_equal(score_many(self.model, [(self.observation, self.actions)])[0],
                                      score_many(restored, [(self.observation, self.actions)])[0])
        self.assertFalse(list(self.directory.glob("*.partial")))
        self.assertEqual(data["metadata"]["modelDimensions"], [32, 16])
        self.assertEqual(data["metadata"]["observationSize"], OBSERVATION_SIZE)

    def test_checkpoint_rejects_unsafe_pickle_and_wrong_encoding(self):
        path = self.directory / "invalid.pt"
        atomic_torch_save({"arbitrary": UnsafeCheckpoint()}, path)
        with self.assertRaises(Exception):
            load_checkpoint(path, self.device)
        data = checkpoint_data(self.model, self.metadata)
        data["metadata"]["encodingVersion"] = 99
        atomic_torch_save(data, path)
        with self.assertRaisesRegex(ValueError, "encodingVersion"):
            load_checkpoint(path, self.device)
        data = checkpoint_data(self.model, self.metadata)
        next(iter(data["state_dict"].values())).fill_(float("inf"))
        atomic_torch_save(data, path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_checkpoint(path, self.device)

    def test_return_training_metadata_defaults_and_inference_capability(self):
        path = self.directory / "return-metadata.pt"
        data = checkpoint_data(self.model, self.metadata)
        self.assertIs(data["metadata"]["learnReturns"], False)
        del data["metadata"]["learnReturns"]
        atomic_torch_save(data, path)
        model, restored = load_checkpoint(path, self.device)
        self.assertIs(restored["metadata"]["learnReturns"], False)
        info = handle_request({"op": "info"}, model, restored["metadata"], self.device, path)
        self.assertIs(info["learnReturns"], False)
        atomic_torch_save(checkpoint_data(self.model, {**self.metadata, "learnReturns": True}), path)
        model, restored = load_checkpoint(path, self.device)
        self.assertIs(restored["metadata"]["learnReturns"], True)
        info = handle_request({"op": "info"}, model, restored["metadata"], self.device, path)
        self.assertIs(info["learnReturns"], True)

    def test_inference_protocol_valid_and_invalid_requests(self):
        request = {"op": "score", "observation": self.observation.tolist(), "actions": self.actions.tolist()}
        info = handle_request({"op": "info"}, self.model, self.metadata, self.device, "fixture.pt")
        self.assertTrue(info["available"])
        self.assertEqual(info["games"], 8)
        self.assertEqual(info["encodingVersion"], 1)
        result = handle_request(request, self.model, self.metadata, self.device, "fixture.pt")
        self.assertEqual(len(result["scores"]), 3)
        requests = [
            [], {"op": "unknown"}, {**request, "observation": [1]},
            {**request, "actions": []}, {**request, "actions": [[0] * ACTION_SIZE] * 513},
            {**request, "actions": [[float("nan")] * ACTION_SIZE]},
            {**request, "observation": ["0"] * OBSERVATION_SIZE},
        ]
        for invalid in requests:
            with self.subTest(request_type=type(invalid)):
                with self.assertRaises(ValueError):
                    handle_request(invalid, self.model, self.metadata, self.device, "fixture.pt")
        stream = io.StringIO("{malformed\n" + "\n".join(json.dumps(value) for value in
                             [requests[1], request, {"op": "close"}, request]) + "\n")
        output = io.StringIO()
        serve(self.model, self.metadata, self.device, "fixture.pt", stream, output)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(replies), 4)
        self.assertIn("error", replies[0])
        self.assertIn("error", replies[1])
        self.assertEqual(len(replies[2]["scores"]), 3)
        self.assertEqual(replies[3], {"closed": True})

    def test_inference_cli_loads_once_and_startup_failure_has_no_stdout(self):
        checkpoint = self.directory / "infer.pt"
        atomic_torch_save(checkpoint_data(self.model, self.metadata), checkpoint)
        command = [sys.executable, str(ROOT / "infer.py"), "--checkpoint", str(checkpoint), "--stdio"]
        result = subprocess.run(command, input='{"op":"info"}\n{"op":"close"}\n',
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[0])["games"], 8)
        self.assertEqual(json.loads(result.stdout.splitlines()[1]), {"closed": True})
        command[command.index(str(checkpoint))] = str(self.directory / "missing.pt")
        result = subprocess.run(command, input="", capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("error", result.stderr)

    def test_terminal_reward_and_state_contract(self):
        self.assertEqual([team_reward(player, 0) for player in range(6)], [1, -1, 1, -1, 1, -1])
        self.assertEqual(team_reward(3, None), 0)
        with self.assertRaises(ValueError):
            team_reward(0, 2)
        terminal = {"observation": [0] * OBSERVATION_SIZE, "player": 2, "phase": "over",
                    "terminal": True, "result": {"winner": None}, "actions": []}
        validate_state(terminal)
        with self.assertRaises(ValueError):
            validate_state({**terminal, "result": {}})
        with self.assertRaises(ValueError):
            validate_state({**terminal, "observation": []})

    def test_bootstrap_and_promotion_gate_atomic_report(self):
        confidence = paired_bootstrap([0, .5, 1, 1], 123, samples=500)
        self.assertEqual(confidence, paired_bootstrap([0, .5, 1, 1], 123, samples=500))
        self.assertEqual(paired_bootstrap([1, 1], 123, samples=500), [1, 1])
        self.assertLessEqual(confidence[0], confidence[1])
        source, destination = self.directory / "source.pt", self.directory / "promoted.pt"
        source.write_bytes(b"checkpoint-fixture")
        for games, ci in ((1998, [.9, 1]), (2000, [.5, .7])):
            with self.assertRaisesRegex(ValueError, "promotion refused"):
                promote_checkpoint(source, destination, games, ci)
            self.assertFalse(destination.exists())
        promote_checkpoint(source, destination, 2000, [.51, .9])
        self.assertEqual(source.read_bytes(), destination.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "changed while copying"):
            promote_checkpoint(source, destination, 2000, [.51, .9], expected_sha256="wrong")
        self.assertEqual(source.read_bytes(), destination.read_bytes())
        report = self.directory / "report.json"
        atomic_json_save({"confidence95": confidence}, report)
        self.assertEqual(json.loads(report.read_text())["confidence95"], confidence)

    def test_explicit_cuda_fails_if_unavailable(self):
        from unittest.mock import patch
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(device_for("auto").type, "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                device_for("cuda")

    def test_training_orchestration_modes_updates_resume_and_paired_evaluation(self):
        from argparse import Namespace
        from unittest.mock import patch
        checkpoint_dir = self.directory / "toy-checkpoints"
        args = training_arguments([
            "--games", "8", "--workers", "2", "--device", "cpu",
            "--checkpoint-dir", str(checkpoint_dir), "--tensorboard-dir", str(self.directory / "toy-runs"),
            "--batch-size", "4", "--warmup", "4", "--replay-size", "64", "--save-every", "4",
            "--mixed-teammate-probability", "0",
        ])
        ToyEnvironment.instances = []
        output = io.StringIO()
        with patch("daguai_rl.NodeEnvironment", ToyEnvironment), redirect_stdout(output):
            self.assertEqual(train_run(args), 0)
        progress = [line.split()[0] for line in output.getvalue().splitlines() if line.startswith("games=")]
        self.assertEqual(progress, ["games=1", "games=8"])
        self.assertIn("completed games=8", output.getvalue())
        self.assertIn("checkpoint=", output.getvalue())
        self.assertTrue(all(not request.get("continue", False)
                            for environment in ToyEnvironment.instances for request in environment.reset_requests))
        self.assertTrue(all(environment.closed for environment in ToyEnvironment.instances))
        checkpoint = checkpoint_dir / "latest.pt"
        _, data = load_checkpoint(checkpoint, self.device)
        self.assertEqual(data["metadata"]["games"], 8)
        self.assertEqual(data["metadata"]["updates"], 8)
        self.assertEqual(data["metadata"]["continuationProbability"], 0)
        self.assertEqual(data["metadata"]["modeCounts"].get("selfplay"), 4)
        self.assertEqual(sum(data["metadata"]["modeCounts"].values()), 8)
        self.assertGreaterEqual(data["metadata"]["modeCounts"].get("heuristic_fallback", 0), 1)
        # Six current decisions per self-play game, three per one-team game.
        self.assertEqual(len(data["replay"]["entries"][0]), 36)
        self.assertLessEqual(len(data["metadata"]["snapshotPool"]), 3)
        args.resume = checkpoint
        args.replay_size = None
        args.games = 4
        args.log_every = 2
        output = io.StringIO()
        with patch("daguai_rl.NodeEnvironment", ToyEnvironment), redirect_stdout(output):
            self.assertEqual(train_run(args), 0)
        progress = [line.split()[0] for line in output.getvalue().splitlines() if line.startswith("games=")]
        self.assertEqual(progress, ["games=9", "games=10", "games=12"])
        _, data = load_checkpoint(checkpoint, self.device)
        self.assertEqual(args.replay_size, 64)
        self.assertEqual(data["replay"]["capacity"], 64)
        self.assertEqual(data["metadata"]["games"], 12)
        self.assertEqual(data["metadata"]["updates"], 12)
        report_path = self.directory / "toy-evaluation.json"
        evaluate_args = Namespace(checkpoint=checkpoint, games=4, seed=1000000000,
                                  device="cpu", max_actions=10, promote=None, report=report_path)
        with patch("daguai_rl.NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate_run(evaluate_args), 0)
        report = json.loads(report_path.read_text())
        self.assertEqual((report["wins"], report["losses"], report["draws"]), (2, 2, 0))
        self.assertEqual(report["score"], .5)
        self.assertEqual(report["confidence95"], [.5, .5])
        self.assertEqual(len(report["checkpointSha256"]), 64)
        self.assertTrue(all(environment.closed for environment in ToyEnvironment.instances))

    def test_learn_returns_continues_rounds_and_records_current_return_decisions(self):
        from unittest.mock import patch
        checkpoint_dir = self.directory / "return-checkpoints"
        args = training_arguments([
            "--games", "2", "--workers", "1", "--device", "cpu", "--learn-returns",
            "--checkpoint-dir", str(checkpoint_dir), "--tensorboard-dir", str(self.directory / "return-runs"),
            "--batch-size", "4", "--warmup", "4", "--replay-size", "64",
        ])
        ToyEnvironment.instances = []
        with patch("daguai_rl.NodeEnvironment", ReturningToyEnvironment), redirect_stdout(io.StringIO()):
            self.assertEqual(train_run(args), 0)
        environment, = ToyEnvironment.instances
        self.assertFalse(environment.reset_requests[0].get("continue", False))
        self.assertIs(environment.reset_requests[1].get("continue"), True)
        self.assertEqual(len(environment.return_steps), 1)
        self.assertEqual(environment.return_steps[0][0], "step")
        self.assertIn(environment.return_steps[0][1]["index"], (0, 1))
        self.assertTrue(environment.closed)
        _, data = load_checkpoint(checkpoint_dir / "latest.pt", self.device)
        observations, features, rewards = data["replay"]["entries"]
        self.assertEqual(len(observations), 13)
        returns = features[:, -1] == 1
        self.assertEqual(int(returns.sum()), 1)
        self.assertEqual(float(observations[returns][0, -1]), 1)
        self.assertEqual(float(rewards[returns][0]), 1)
        self.assertEqual(data["metadata"]["roundResetMode"], "continue")
        self.assertEqual(data["metadata"]["continuationProbability"], 1)
        self.assertIs(data["metadata"]["learnReturns"], True)
        args.resume = checkpoint_dir / "latest.pt"
        args.learn_returns = False
        args.games = 1
        with patch("daguai_rl.NodeEnvironment", ReturningToyEnvironment), redirect_stdout(io.StringIO()):
            self.assertEqual(train_run(args), 0)
        _, resumed = load_checkpoint(args.resume, self.device)
        self.assertIs(resumed["metadata"]["learnReturns"], True)
        self.assertEqual(resumed["metadata"]["roundResetMode"], "fresh")

    def test_log_interval_requires_positive_integer(self):
        from contextlib import redirect_stderr
        self.assertEqual(training_arguments([]).log_every, 100)
        for value in ("0", "-1", "1.5"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training_arguments(["--log-every", value])

    def test_mixed_team_heuristic_decisions_excluded_from_replay(self):
        from unittest.mock import patch
        checkpoint_dir = self.directory / "mixed-checkpoints"
        args = training_arguments([
            "--games", "8", "--workers", "1", "--device", "cpu",
            "--mixed-teammate-probability", "1", "--save-every", "2",
            "--checkpoint-dir", str(checkpoint_dir), "--tensorboard-dir", str(self.directory / "mixed-runs"),
            "--batch-size", "4", "--warmup", "4", "--replay-size", "64",
        ])
        ToyEnvironment.instances = []
        with patch("daguai_rl.NodeEnvironment", SeatToyEnvironment), redirect_stdout(io.StringIO()):
            self.assertEqual(train_run(args), 0)
        _, data = load_checkpoint(checkpoint_dir / "latest.pt", self.device)
        self.assertEqual(data["metadata"]["modeCounts"],
                         {"selfplay": 4, "mixed_heuristic": 2, "mixed_snapshot": 2})
        self.assertEqual(data["metadata"]["mixedTeamGames"], 4)
        observations = data["replay"]["entries"][0].numpy()
        self.assertEqual(len(observations), 32)
        environment, = ToyEnvironment.instances
        for seed in (3, 4, 7, 8):
            decisions = [decision for decision in environment.decisions if decision["seed"] == seed]
            learner_team, = [team for team in (0, 1) if sum(
                decision["op"] == "step" and decision["player"] % 2 == team
                for decision in decisions) == 2]
            learner_steps = {decision["player"] for decision in decisions
                             if decision["op"] == "step" and decision["player"] % 2 == learner_team}
            heuristic_teammates = {decision["player"] for decision in decisions
                                   if decision["op"] == "heuristic" and decision["player"] % 2 == learner_team}
            self.assertEqual(len(heuristic_teammates), 1)
            replay_players = {int(observation[-1]) - 1 for observation in observations if observation[-2] == seed}
            self.assertEqual(replay_players, learner_steps)
            self.assertTrue(replay_players.isdisjoint(heuristic_teammates))
        self.assertTrue(environment.closed)

    def test_mixed_teammate_probability_bounds(self):
        from contextlib import redirect_stderr
        self.assertEqual(training_arguments([]).mixed_teammate_probability, .25)
        for value in ("-0.1", "1.1", "nan", "inf"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training_arguments(["--mixed-teammate-probability", value])

    def test_evaluation_heuristic_teammates_rotate_one_learner_and_refuse_promotion(self):
        from argparse import Namespace
        from unittest.mock import patch
        checkpoint = self.directory / "teammate-eval.pt"
        atomic_torch_save(checkpoint_data(self.model, self.metadata), checkpoint)
        report_path = self.directory / "teammate-eval.json"
        args = Namespace(checkpoint=checkpoint, games=6, seed=1000000000,
                         device="cpu", max_actions=10, promote=None, report=report_path,
                         teammates="heuristic")
        ToyEnvironment.instances = []
        with patch("daguai_rl.NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate_run(args), 0)
        environment, = ToyEnvironment.instances
        for round_index, request in enumerate(environment.reset_requests, 1):
            learner_team = (round_index - 1) % 2
            expected_seat = learner_team + 2 * (request["seed"] % 3)
            decisions = [decision for decision in environment.decisions if decision["round"] == round_index]
            self.assertEqual([decision["player"] for decision in decisions if decision["op"] == "step"],
                             [expected_seat])
            self.assertEqual(sum(decision["op"] == "heuristic" and decision["player"] % 2 == learner_team
                                 for decision in decisions), 2)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["teammates"], "heuristic")
        self.assertFalse(report["promotion"]["eligible"])
        self.assertEqual(report["pairedSeeds"][0]["learnerSeats"], [[2], [3]])
        self.assertTrue(environment.closed)
        args.games = 2000
        args.promote = self.directory / "must-not-promote.pt"
        with self.assertRaisesRegex(ValueError, "full model-team deployment"):
            evaluate_run(args)
        self.assertFalse(args.promote.exists())

    def test_action_limit_fails_without_reward_and_closes_all_workers(self):
        from unittest.mock import patch
        checkpoint_dir = self.directory / "limit-checkpoints"
        args = training_arguments([
            "--games", "2", "--workers", "2", "--device", "cpu", "--max-actions", "1",
            "--checkpoint-dir", str(checkpoint_dir), "--tensorboard-dir", str(self.directory / "limit-runs"),
            "--batch-size", "4", "--warmup", "4", "--replay-size", "64",
        ])
        ToyEnvironment.instances = []
        with patch("daguai_rl.NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "not a rewarded terminal truncation"):
                train_run(args)
        self.assertTrue(all(environment.closed for environment in ToyEnvironment.instances))
        _, data = load_checkpoint(checkpoint_dir / "latest.pt", self.device)
        self.assertEqual(data["metadata"]["status"], "failed")
        self.assertEqual(data["metadata"]["games"], 0)
        self.assertIsNone(data["replay"]["entries"])

    @unittest.skipUnless((ROOT / "env.js").exists() and (ROOT.parent / "policy.js").exists(),
                         "Node environment/policy adapter not installed yet")
    def test_node_bridge_seeded_reset_error_and_close(self):
        environment = NodeEnvironment(max_actions=1)
        try:
            first = environment.state("reset", seed=54321)
            second = environment.state("reset", seed=54321)
            self.assertEqual(first, second)
            self.assertEqual(len(first["observation"]), OBSERVATION_SIZE)
            with self.assertRaisesRegex(RuntimeError, "Invalid legal action index"):
                environment.state("step", index=-1)
            environment.state("heuristic")
            with self.assertRaisesRegex(RuntimeError, "action limit"):
                environment.state("heuristic")
        finally:
            environment.close()
        self.assertIsNotNone(environment.process.poll())


if __name__ == "__main__":
    unittest.main()
