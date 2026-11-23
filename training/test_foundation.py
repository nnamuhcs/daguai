"""Focused CPU tests; toy rewards are plumbing checks, never strength evidence."""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import fcntl
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import unittest
from unittest import mock
import uuid

import numpy as np
import torch

import daguai_rl
import foundation as f

torch.set_num_threads(1)


def observation(marker=0):
    values = np.zeros(f.OBSERVATION_SIZE, np.float32)
    values[0] = marker
    return values.tolist()


def features(marker=0):
    values = np.zeros(f.ACTION_SIZE, np.float32)
    values[-1] = marker
    return values.tolist()


def event(actor=0, recipient=None, kind="play", answer=0):
    values = np.zeros(f.HISTORY_SIZE, np.float32)
    values[0] = 1 / 3 if kind == "play" else 0
    values[54 + actor] = 1
    if recipient is not None:
        values[60 + recipient] = 1
    values[66 + ("play", "pass", "ask").index(kind)] = 1
    values[69] = answer
    return values.tolist()


class ToyEnvironment:
    """An asking choice without changing seats, plus actual continued return phases."""

    def __init__(self, **kwargs):
        self.finished = False

    def state(self, op, **kwargs):
        events = []
        if op == "reset":
            if kwargs.get("continue"):
                if not self.finished:
                    raise ValueError("cannot continue live episode")
                self.stage = "return"
            else:
                self.stage = "ask"
            self.seed = kwargs["seed"]
            self.finished = False
        elif op in ("step", "heuristic"):
            if op == "step" and kwargs["index"] != 0:
                raise ValueError("invalid toy action")
            if self.stage == "return":
                self.stage = "ask"
            elif self.stage == "ask":
                events = [event(0, 1, "ask", 0.3)]
                self.stage = "play"
            else:
                events = [event(0, kind="play")]
                self.finished = True
        else:
            raise ValueError("invalid operation")
        action = {"kind": self.stage, "ids": [] if self.stage == "ask" else [0],
                  "features": features({"return": 3, "ask": 2, "play": 1}[self.stage])}
        if self.stage == "ask":
            action["target"] = 1
        result = {
            "observation": observation(self.seed % 2), "player": 0,
            "phase": "over" if self.finished else "returning" if self.stage == "return" else "playing",
            "terminal": self.finished, "result": {"winner": self.seed % 2} if self.finished else None,
            "actions": [] if self.finished else [action], "events": events,
        }
        f.validate_state(result)
        return result

    def close(self):
        pass


class EvaluationToy(ToyEnvironment):
    def state(self, op, **kwargs):
        if op == "step" and kwargs.get("index") == 1 and self.stage == "ask":
            self.stage = "play"
            kwargs["index"] = 0
        state = super().state(op, **kwargs)
        if not state["terminal"] and self.stage == "ask":
            state["actions"].append({"kind": "play", "ids": [0], "features": features(1)})
        return state


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.directory = f.ROOT / f".foundation-tests-{uuid.uuid4().hex}"
        self.directory.mkdir()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def checkpoint(self, name="model.pt", games=0, updates=0, **metadata):
        path = self.directory / name
        model = f.ActionValue()
        f.atomic_torch_save(f.checkpoint_data(model, dict(games=games, updates=updates, **metadata)), path)
        return path

    def v1(self):
        path = self.directory / "baseline.pt"
        f.atomic_torch_save(daguai_rl.checkpoint_data(
            daguai_rl.ActionValue(), {"games": 80, "updates": 4}), path)
        return path

    def monitor_args(self, **kwargs):
        initial = self.checkpoint("initial.pt")
        current = self.checkpoint("latest.pt", 2, 1, status="completed", targetGames=2)
        baseline = self.v1()
        argv = [
            "monitor", "--checkpoint", str(current), "--initial", str(initial),
            "--baseline", str(baseline), "--target-games", "2", "--games", "2",
            "--final-games", "2", "--output-dir", str(self.directory / "progress"),
            "--tensorboard-dir", str(self.directory / "tb"),
        ]
        args = f.arguments(argv)
        for key, value in kwargs.items():
            setattr(args, key, value)
        return args

    def test_relative_rotation_preserves_public_information(self):
        absolute = [event(5, 2, "ask", -1), event(3, kind="pass"), event(1)]
        relative = f.relative_history(absolute, 5)
        self.assertEqual(relative[0, 54], 1)
        self.assertEqual(relative[0, 63], 1)
        np.testing.assert_array_equal(relative[:, :54], np.array(absolute, np.float32)[:, :54])
        np.testing.assert_array_equal(relative[:, 66:], np.array(absolute, np.float32)[:, 66:])
        self.assertEqual(absolute[0][59], 1)
        np.testing.assert_array_equal(f.relative_history(relative, 1),
                                      f.relative_history(absolute, 0))
        with self.assertRaises(ValueError):
            f.relative_history(absolute, -1)

    def test_shared_prefixes_exclude_future_events_and_private_return_cards(self):
        replay = f.ReplayBuffer(3)
        history = [event(3), event(5, 2, "ask", -1)]
        decisions = [
            (observation(), features(), 5, 0, "return"),
            (observation(), features(2), 5, 1, "ask"),
            (observation(), features(1), 5, 2, "play"),
        ]
        replay.add_episode(history, decisions, 1)
        self.assertEqual(len(replay.episodes), 1)
        self.assertEqual(len(set(entry[5] for entry in replay.entries)), 1)
        histories = replay.sample(3, random.Random(1), torch.device("cpu"))[2]
        self.assertEqual(sorted(map(len, histories)), [0, 1, 2])
        self.assertEqual(sum(replay.references.values()), 3)
        history[0][54] = 1
        self.assertEqual(replay.episodes[0][0, 54], 0)
        # The return has no private card token; action IDs are never stored in public history.
        self.assertEqual(replay.entries[0][3], 0)
        replay.add_episode([event()], decisions[:1], 0)
        self.assertEqual(len(replay), 3)
        replay.add_episode([], [decisions[0]] * 3, None)
        self.assertEqual(len(replay.episodes), 1)
        self.assertEqual(replay.history_bytes, 0)

    def test_history_memory_bound_evicts_old_episode_references(self):
        bound = f.MAX_HISTORY * f.HISTORY_SIZE * 4
        replay = f.ReplayBuffer(10000, bound)
        history = np.zeros((f.MAX_HISTORY, f.HISTORY_SIZE), np.float32)
        decision = (observation(), features(), 0, f.MAX_HISTORY, "play")
        replay.add_episode(history, [decision], 0)
        replay.add_episode(history, [decision], 1)
        self.assertEqual(len(replay), 1)
        self.assertEqual(len(replay.episodes), 1)
        self.assertLessEqual(replay.history_bytes, bound)
        self.assertEqual(next(iter(replay.entries))[4], -1)

    def test_future_events_in_shared_final_array_cannot_change_prefix_scores(self):
        before = [event(3, 2, "ask", 0.3), event(1), event(2)]
        changed_future = [before[0], event(5, 0, "ask", -1), event(4, kind="pass")]
        model = f.ActionValue()
        scores = []
        for history in (before, changed_future):
            replay = f.ReplayBuffer(2)
            replay.add_episode(history, [(observation(), features(), 4, 1, "play")], 0)
            observations, actions, histories, _ = replay.sample(1, random.Random(0), "cpu")
            self.assertEqual(len(histories[0]), 1)
            np.testing.assert_array_equal(histories[0].numpy(), f.relative_history(before[:1], 4))
            scores.append(model(observations, actions, histories).detach())
        torch.testing.assert_close(scores[0], scores[1], rtol=0, atol=0)

    def test_replay_rejects_impossible_future_prefix_and_outcome(self):
        replay = f.ReplayBuffer()
        for prefix, winner in ((1, 0), (0, 2)):
            with self.assertRaises(ValueError):
                replay.add_episode([], [(observation(), features(), 0, prefix, "ask")], winner)
        self.assertEqual(len(replay), 0)

    def test_checkpoint_optimizer_replay_and_all_cpu_rng_resume(self):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        rng = random.Random(7)
        model = f.ActionValue()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        replay = f.ReplayBuffer(4)
        replay.add_episode([event()], [
            (observation(), features(), 0, 0, "return"),
            (observation(1), features(2), 0, 1, "ask"),
        ], 0)
        self.assertTrue(np.isfinite(f.update_model(model, optimizer, replay, 2, rng, "cpu")))
        path = self.directory / "resume.pt"
        f.atomic_torch_save(f.checkpoint_data(
            model, {"games": 1, "updates": 1}, optimizer, replay, rng), path)
        expected = (rng.random(), random.random(), np.random.random(), torch.rand(3))
        expected_loss = f.update_model(model, optimizer, replay, 2, rng, "cpu")
        loaded, data = f.load_checkpoint(path, "cpu")
        restored = f.ReplayBuffer(4)
        new_optimizer = torch.optim.Adam(loaded.parameters())
        new_rng = random.Random(0)
        f.restore_training(data, new_optimizer, restored, new_rng, torch.device("cpu"))
        actual = (new_rng.random(), random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[:3], actual[:3])
        torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
        self.assertEqual(new_optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(len(restored), 2)
        actual_loss = f.update_model(loaded, new_optimizer, restored, 2, new_rng, "cpu")
        self.assertEqual(actual_loss, expected_loss)
        for first, second in zip(model.parameters(), loaded.parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            f.ReplayBuffer(3).load_state_dict(data["replay"])

    def test_gru_updates_require_training_forward_and_preserve_callers_mode(self):
        model = f.ActionValue().eval()
        replay = f.ReplayBuffer(2)
        replay.add_episode([event()], [(observation(), features(), 0, 1, "play")], 0)
        optimizer = torch.optim.Adam(model.parameters())
        rng = random.Random(1)
        observed_modes = []

        def enforce_cudnn_contract(gru, inputs):
            if torch.is_grad_enabled():
                observed_modes.append(gru.training)
                if not gru.training:
                    raise RuntimeError("cudnn RNN backward requires a training-mode forward")

        hook = model.history.register_forward_pre_hook(enforce_cudnn_contract)
        try:
            for initial_mode in (False, True):
                model.train(initial_mode)
                f.score_many(model, [(observation(), [features()], [event()])])
                self.assertEqual(model.training, initial_mode)
                self.assertTrue(np.isfinite(f.update_model(model, optimizer, replay, 1, rng, "cpu")))
                self.assertEqual(model.training, initial_mode)
                self.assertEqual(model.history.training, initial_mode)
            self.assertEqual(observed_modes, [True, True])
            model.eval()
            with mock.patch.object(model, "forward", side_effect=RuntimeError("forward failure")):
                with self.assertRaisesRegex(RuntimeError, "forward failure"):
                    f.update_model(model, optimizer, replay, 1, rng, "cpu")
            self.assertFalse(model.training)
            self.assertFalse(model.history.training)
        finally:
            hook.remove()

    def test_load_rejects_encoding_and_preserves_torch_rng(self):
        path = self.checkpoint()
        state = torch.get_rng_state().clone()
        f.load_checkpoint(path, "cpu")
        torch.testing.assert_close(torch.get_rng_state(), state)
        data = torch.load(path, weights_only=True)
        data["metadata"]["historySize"] = 70
        f.atomic_torch_save(data, path)
        with self.assertRaisesRegex(ValueError, "historySize"):
            f.load_checkpoint(path, "cpu")
        with self.assertRaisesRegex(ValueError, "encodingVersion"):
            f.load_checkpoint(self.v1(), "cpu")

    def test_inference_history_changes_scores_without_truncation(self):
        torch.manual_seed(2)
        model = f.ActionValue()
        request = {"op": "score", "observation": observation(), "actions": [features()] * 512, "history": []}
        first = f.handle_request(model, request)["scores"]
        second = f.handle_request(model, {**request, "history": [event(3, 1, "ask", -1)]})["scores"]
        self.assertEqual(len(first), 512)
        self.assertNotAlmostEqual(first[0], second[0], places=7)
        full = [event(0)] * 8
        changed_early = [event(3, 1, "ask", -1)] + full[1:]
        old_history = f.handle_request(model, {**request, "history": full})["scores"]
        new_history = f.handle_request(model, {**request, "history": changed_early})["scores"]
        self.assertNotEqual(old_history[0], new_history[0])
        self.assertEqual(f.handle_request(model, {"op": "info"})["historySize"], 71)
        for bad in (
            {**request, "history": None},
            {**request, "history": [[float("nan")] * 71]},
            {**request, "history": [[0] * 70]},
            {**request, "history": [[0] * 71] * 4097},
            {**request, "actions": [features()] * 513},
            {**request, "observation": [float("inf")] * 922},
        ):
            with self.assertRaises(ValueError):
                f.handle_request(model, bad)
        out = io.StringIO()
        f.serve(model, io.StringIO('{"op":"info"}\n{"op":"close"}\n'), out)
        responses = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(responses[0]["encodingVersion"], 2)
        self.assertEqual(responses[1], {"closed": True})

    def test_inference_cli_stdout_is_only_json_and_startup_failure_is_nonzero(self):
        checkpoint = self.checkpoint()
        argv = [sys.executable, str(f.ROOT / "foundation.py"), "infer",
                "--checkpoint", str(checkpoint), "--device", "cpu"]
        request = {"op": "score", "observation": observation(),
                   "actions": [features(), features(2)], "history": [event(3)]}
        result = subprocess.run(
            argv, input=json.dumps(request) + '\n{"op":"info"}\n{"op":"close"}\n',
            text=True, capture_output=True, timeout=30, check=True,
        )
        messages = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(messages[0]["scores"]), 2)
        self.assertEqual(messages[1]["encodingVersion"], 2)
        self.assertEqual(messages[2], {"closed": True})
        argv[argv.index(str(checkpoint))] = str(self.directory / "missing.pt")
        failed = subprocess.run(argv, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(failed.stdout, "")
        self.assertIn("failed", failed.stderr)

    def test_info_preserves_real_failure_but_does_not_raise_stale_resumed_error(self):
        model = f.ActionValue()
        model.metadata = {"games": 3, "updates": 0, "status": "failed", "error": "CuDNN failure"}
        failed = f.handle_request(model, {"op": "info"})
        self.assertEqual(failed["error"], "CuDNN failure")
        self.assertEqual(failed["updates"], 0)
        for status in ("running", "completed"):
            model.metadata["status"] = status
            self.assertNotIn("error", f.handle_request(model, {"op": "info"}))
        self.assertEqual(model.metadata["error"], "CuDNN failure")

    def test_training_records_learned_asks_and_returns_resume_inherits_capacity(self):
        checkpoint_dir = self.directory / "checkpoints"
        arguments = [
            "train", "--games", "6", "--device", "cpu", "--workers", "2",
            "--batch-size", "2", "--warmup", "2", "--replay-size", "10",
            "--save-every", "2", "--epsilon", "1",
            "--checkpoint-dir", str(checkpoint_dir),
            "--tensorboard-dir", str(self.directory / "tb"),
        ]
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment):
            f.train(f.arguments(arguments))
            initial_digest = f.sha256_file(checkpoint_dir / "initial.pt")
            _, data = f.load_checkpoint(checkpoint_dir / "latest.pt", "cpu")
            metadata = data["metadata"]
            self.assertEqual(metadata["games"], 6)
            self.assertEqual(metadata["updates"], 24)
            self.assertEqual(metadata["updatesPerGame"], 4)
            self.assertEqual(metadata["modeCounts"]["selfplay"], 4)
            self.assertGreaterEqual(metadata["decisionCounts"]["ask"], 2)
            self.assertGreaterEqual(metadata["decisionCounts"]["return"], 1)
            self.assertEqual(metadata["learnerAskDecisions"], metadata["decisionCounts"]["ask"])
            self.assertEqual(metadata["learnerReturnDecisions"], metadata["decisionCounts"]["return"])
            self.assertEqual(metadata["transitionsCollected"], sum(metadata["decisionCounts"].values()))
            self.assertEqual(metadata["maxObservedHistoryLength"], 2)
            self.assertEqual(metadata["continuedRounds"], 4)
            saved_kinds = Counter(ref[-1] for ref in data["replay"]["references"])
            self.assertGreater(saved_kinds["ask"], 0)
            self.assertGreater(saved_kinds["return"], 0)
            _, initial = f.load_checkpoint(checkpoint_dir / "initial.pt", "cpu")
            self.assertEqual(initial["metadata"]["updates"], 0)
            data["metadata"].update(status="failed", error="simulated GPU failure")
            f.atomic_torch_save(data, checkpoint_dir / "latest.pt")
            resume = [
                "train", "--games", "2", "--device", "cpu",
                "--resume", str(checkpoint_dir / "latest.pt"),
                "--checkpoint-dir", str(checkpoint_dir),
                "--tensorboard-dir", str(self.directory / "tb"),
            ]
            f.train(f.arguments(resume))
            _, continued = f.load_checkpoint(checkpoint_dir / "latest.pt", "cpu")
            self.assertEqual(continued["metadata"]["games"], 8)
            self.assertEqual(continued["metadata"]["updates"], 32)
            self.assertEqual(continued["metadata"]["status"], "completed")
            self.assertNotIn("error", continued["metadata"])
            self.assertEqual(continued["replay"]["capacity"], 10)
            self.assertEqual(continued["metadata"]["batchSize"], 2)
            self.assertEqual(f.sha256_file(checkpoint_dir / "initial.pt"), initial_digest)
            output = io.StringIO()
            with redirect_stdout(output):
                f.train(f.arguments(resume + [
                    "--games", "1", "--save-every", "1000", "--log-every", "7",
                ]))
            _, rescheduled = f.load_checkpoint(checkpoint_dir / "latest.pt", "cpu")
            self.assertEqual(rescheduled["metadata"]["saveEvery"], 1000)
            self.assertEqual(rescheduled["metadata"]["logEvery"], 7)
            override = rescheduled["metadata"]["resumeCadenceOverrides"][-1]
            self.assertEqual(override["changes"]["saveEvery"], {"from": 2, "to": 1000})
            self.assertEqual(override["changes"]["logEvery"], {"from": 100, "to": 7})
            self.assertTrue(override["snapshotRefreshChanged"])
            self.assertIn("saveEvery: 2 -> 1000", output.getvalue())
            self.assertIn("snapshot refresh follows saveEvery", output.getvalue())
            with redirect_stdout(io.StringIO()):
                f.train(f.arguments(resume + ["--games", "1"]))
            _, inherited = f.load_checkpoint(checkpoint_dir / "latest.pt", "cpu")
            self.assertEqual(inherited["metadata"]["saveEvery"], 1000)
            self.assertEqual(inherited["metadata"]["logEvery"], 7)
            for flag, value in (("--replay-size", "9"), ("--batch-size", "3"), ("--seed", "3"),
                                ("--updates-per-game", "2")):
                with self.assertRaisesRegex(ValueError, "differs"):
                    f.train(f.arguments(resume + [flag, value]))
            with self.assertRaisesRegex(ValueError, "already contains"):
                f.train(f.arguments(arguments))
        self.assertLessEqual(len(list(checkpoint_dir.glob("snapshot-*.pt"))), 3)

    def test_batch_warmup_bounds(self):
        for extra in (
            ["--batch-size", "3", "--warmup", "2"],
            ["--batch-size", "11", "--replay-size", "10"],
        ):
            args = f.arguments([
                "train", "--device", "cpu", "--games", "1",
                "--checkpoint-dir", str(self.directory / "checkpoints"), *extra,
            ])
            with self.assertRaises(ValueError):
                f.train(args)
        self.assertFalse((self.directory / "checkpoints" / "initial.pt").exists())
        with mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                f.arguments(["train", "--updates-per-game", "0"])

    def test_training_failure_never_rewards_partial_asking_episode(self):
        args = f.arguments([
            "train", "--games", "1", "--device", "cpu", "--max-actions", "1",
            "--checkpoint-dir", str(self.directory / "checkpoints"),
            "--tensorboard-dir", str(self.directory / "tb"),
        ])
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment):
            with self.assertRaisesRegex(RuntimeError, "no truncated reward"):
                f.train(args)
        _, data = f.load_checkpoint(args.checkpoint_dir / "latest.pt", "cpu")
        self.assertEqual(data["metadata"]["status"], "failed")
        self.assertEqual(data["metadata"]["games"], 0)
        self.assertEqual(data["replay"]["references"], [])

    def test_initial_save_cannot_overwrite_and_legacy_directory_is_rejected(self):
        path = self.checkpoint()
        digest = f.sha256_file(path)
        with self.assertRaises(FileExistsError):
            f.immutable_torch_save({"different": True}, path)
        self.assertEqual(f.sha256_file(path), digest)
        args = f.arguments(["train", "--checkpoint-dir", str(f.ROOT / "checkpoints")])
        with self.assertRaisesRegex(ValueError, "v1 checkpoints"):
            f.train(args)

    def test_v1_never_asks_and_uses_legacy_action_prefix(self):
        model, _, version = f.load_policy(self.v1())
        state = {
            "player": 0, "observation": observation(),
            "actions": [
                {"kind": "ask", "features": features(99)},
                {"kind": "play", "features": features(1)},
                {"kind": "pass", "features": features(2)},
            ],
        }
        with mock.patch.object(daguai_rl, "score_many", return_value=[np.array([0, 1])]) as scoring:
            self.assertEqual(f.policy_index(model, version, state, []), 2)
            received = scoring.call_args.args[1][0][1]
            self.assertEqual(len(received), 2)
            self.assertEqual(len(received[0]), 71)

    def test_evaluation_pairs_swapped_sides_real_terminal_outcomes(self):
        report = f.evaluate_checkpoint(self.checkpoint(), games=4, seed=10, env_factory=ToyEnvironment)
        self.assertEqual((report["wins"], report["losses"], report["draws"]), (2, 2, 0))
        self.assertEqual(report["score"], 0.5)
        self.assertEqual(report["confidence95"], [0.5, 0.5])
        self.assertEqual(report["pairedSeeds"][0]["learnerSeats"], [[0, 2, 4], [1, 3, 5]])
        single = f.evaluate_checkpoint(
            self.v1(), games=2, seed=10, mode="heuristic", env_factory=ToyEnvironment)
        self.assertEqual(single["pairedSeeds"][0]["learnerSeats"], [[2], [3]])
        self.assertEqual(single["policyEncoding"], 1)
        self.assertIn("tribute/return strategy is not measured", report["scope"])
        with self.assertRaises(RuntimeError):
            f.evaluate_checkpoint(self.checkpoint("limit.pt"), games=2, max_actions=1,
                                  env_factory=ToyEnvironment)

    def test_paired_difference_confidence_and_protocol_mismatch(self):
        report = f.evaluate_checkpoint(self.checkpoint(), games=4, seed=10, env_factory=ToyEnvironment)
        baseline = json.loads(json.dumps(report))
        for pair in report["pairedSeeds"]:
            pair["scores"] = [1, 1]
        for pair in baseline["pairedSeeds"]:
            pair["scores"] = [0, 0]
        result = f.compare_reports(report, baseline)
        self.assertEqual(result["confidence95"], [1, 1])
        self.assertEqual(result["assessment"], "improved")
        self.assertEqual(f.compare_reports(baseline, report)["assessment"], "regressed")
        self.assertEqual(f.compare_reports(report, report)["assessment"], "not established")
        baseline["protocol"]["maxActions"] += 1
        with self.assertRaisesRegex(ValueError, "protocol"):
            f.compare_reports(report, baseline)

    def test_freeze_is_atomic_immutable_and_leaves_no_partial(self):
        source = self.checkpoint()
        destination = self.directory / "frozen" / "model.pt"
        digest = f.freeze_checkpoint(source, destination)
        self.assertEqual(digest, f.sha256_file(source))
        self.assertEqual(f.freeze_checkpoint(source, destination), digest)
        replacement = self.checkpoint("replacement.pt", 1)
        with self.assertRaisesRegex(ValueError, "immutable"):
            f.freeze_checkpoint(replacement, destination)
        self.assertEqual(f.sha256_file(destination), digest)
        self.assertEqual(list(destination.parent.glob("*.partial")), [])
        with self.assertRaises(ValueError):
            f.freeze_checkpoint(source, source)
        with mock.patch.object(f.shutil, "copyfileobj", side_effect=OSError("copy failed")):
            with self.assertRaises(OSError):
                f.freeze_checkpoint(source, destination.parent / "failed.pt")
        self.assertFalse((destination.parent / "failed.pt").exists())

    def test_monitor_never_opens_final_holdout_and_reuses_paired_baselines(self):
        args = self.monitor_args()
        calls = []

        def evaluate(path, **kwargs):
            calls.append((Path(path).name, kwargs["seed"], kwargs["mode"], kwargs["device"]))
            return f.evaluate_checkpoint(path, **kwargs, env_factory=EvaluationToy)

        with mock.patch.object(f, "tactical_audit", return_value={"available": False}):
            self.assertEqual(f.monitor(args, evaluator=evaluate), 0)
        state = json.loads((args.output_dir / "latest.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["stage"], "validation complete; final holdout not opened")
        self.assertFalse(state["automaticFinalEvaluation"])
        self.assertEqual(len(calls), 6)
        self.assertEqual({call[1] for call in calls}, {args.seed})
        self.assertEqual({call[3] for call in calls}, {"cpu"})
        self.assertEqual(state["reports"]["model"]["comparisons"]["initial"]["assessment"], "not established")
        self.assertFalse((args.output_dir / "final-initial-model.json").exists())
        history = (args.output_dir / "history.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(row)["stage"] for row in history], ["validation"])
        self.assertEqual(f.monitor(args, evaluator=lambda *_: self.fail("duplicate validation")), 0)

    def test_monitor_failure_and_dead_trainer_are_durable(self):
        args = self.monitor_args()
        with mock.patch.object(f, "tactical_audit", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "evaluation broke"):
                f.monitor(args, evaluator=mock.Mock(side_effect=RuntimeError("evaluation broke")))
        state = json.loads((args.output_dir / "latest.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertIn("evaluation broke", state["error"])
        model, data = f.load_checkpoint(args.checkpoint, "cpu")
        data["metadata"].update(games=1, status="interrupted", error="test interruption")
        f.atomic_torch_save(data, args.checkpoint)
        with self.assertRaisesRegex(RuntimeError, "trainer interrupted"):
            f.monitor(args, evaluator=mock.Mock())
        state = json.loads((args.output_dir / "latest.json").read_text())
        self.assertEqual(state["status"], "failed")
        data["metadata"].update(status="running")
        f.atomic_torch_save(data, args.checkpoint)
        with self.assertRaisesRegex(RuntimeError, "trainer exited"):
            f.monitor(args, evaluator=mock.Mock(), alive=lambda _: False)
        args.checkpoint.unlink()
        with self.assertRaisesRegex(RuntimeError, "before any v2"):
            f.monitor(args, evaluator=mock.Mock(), alive=lambda _: False)

    def test_monitor_lock_rejects_duplicate_checkpoint_watcher(self):
        args = self.monitor_args()
        with (args.checkpoint.parent / ".foundation-monitor.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "another monitor"):
                f.monitor(args)
        self.assertFalse((args.output_dir / "latest.json").exists())

    def test_monitor_waits_for_live_resumed_trainer_with_stale_completed_checkpoint(self):
        args = self.monitor_args(target_games=4, trainer_pid=12345)
        evaluator = mock.Mock(side_effect=lambda path, **kw:
                              f.evaluate_checkpoint(path, **kw, env_factory=EvaluationToy))
        waits = []

        def advance_checkpoint(delay):
            state = json.loads((args.output_dir / "latest.json").read_text())
            self.assertEqual(state["status"], "waiting")
            self.assertEqual(state["stage"], "awaiting resumed checkpoint")
            self.assertTrue(state["waitingForResume"])
            self.assertEqual(state["currentGames"], 2)
            self.assertEqual(evaluator.call_count, 0)
            waits.append(delay)
            _, data = f.load_checkpoint(args.checkpoint, "cpu")
            data["metadata"].update(games=4, targetGames=4)
            f.atomic_torch_save(data, args.checkpoint)

        with mock.patch.object(f, "tactical_audit", return_value={"available": False}):
            f.monitor(args, evaluator=evaluator, sleep=advance_checkpoint, alive=lambda _: True)
        self.assertEqual(waits, [args.poll_seconds])
        state = json.loads((args.output_dir / "latest.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertFalse(state["waitingForResume"])

    def test_monitor_serial_catchup_uses_preserved_exact_milestone_models(self):
        args = self.monitor_args(target_games=20000)
        model, data = f.load_checkpoint(args.checkpoint, "cpu")
        data["metadata"].update(games=20000, targetGames=20000)
        f.atomic_torch_save(data, args.checkpoint)
        for game in (1000, 10000):
            f.atomic_torch_save(f.checkpoint_data(model, {
                "games": game, "updates": game, "status": "running",
            }), args.checkpoint.parent / f"milestone-{game:09d}.pt")
        with mock.patch.object(f, "tactical_audit", return_value={"available": False}):
            f.monitor(args, evaluator=lambda path, **kw:
                      f.evaluate_checkpoint(path, **kw, env_factory=EvaluationToy))
        rows = [json.loads(line) for line in (args.output_dir / "history.jsonl").read_text().splitlines()]
        self.assertEqual([row["checkpointGames"] for row in rows], [1000, 10000, 20000])
        state = json.loads((args.output_dir / "latest.json").read_text())
        self.assertEqual(state["completedMilestones"], [1000, 10000, 20000])
        self.assertEqual(state["skippedMilestones"], [])

    def test_more_frequent_validation_starts_after_transferred_training(self):
        self.assertEqual(f.evaluation_milestones(11500, 1000, 8000), [9000, 10000, 11000, 11500])
        self.assertEqual(f.next_milestone([], 11500, 1000, 8000), 9000)
        self.assertEqual(f.next_milestone([9008], 11500, 1000, 8000), 10000)
        self.assertEqual(f.next_milestone([11008], 11500, 1000, 8000), 11500)
        with self.assertRaisesRegex(ValueError, "interval"):
            f.evaluation_milestones(10000, 0, 8000)

    def test_warm_start_monitor_does_not_call_transferred_weights_random(self):
        args = self.monitor_args(start_games=1, validation_every=1, initial_kind="warm-start")
        model, data = f.load_checkpoint(args.initial, "cpu")
        data["metadata"].update(games=1, updates=4)
        f.atomic_torch_save(data, args.initial)
        with mock.patch.object(f, "tactical_audit", return_value={"available": False}):
            f.monitor(args, evaluator=lambda path, **kw:
                      f.evaluate_checkpoint(path, **kw, env_factory=EvaluationToy))
        state = json.loads((args.output_dir / "latest.json").read_text())
        self.assertEqual(state["initialGames"], 1)
        self.assertEqual(state["initialKind"], "warm-start")
        self.assertIn("not random", state["initialDescription"])
        self.assertEqual(state["completedMilestones"], [2])

    @unittest.skipUnless((f.ROOT / "foundation-scenarios.js").exists(), "parent fixtures not available")
    def test_tactical_audit_is_reference_only_and_reports_top_three(self):
        report = f.tactical_audit({"initial": self.checkpoint(), "v1": self.v1()})
        self.assertTrue(report["available"])
        self.assertIn("not proven optimal", report["notice"])
        self.assertGreaterEqual(len(report["records"]), 2)
        self.assertTrue(all(1 <= len(record["top3"]) <= 3 for record in report["records"]))

    @unittest.skipUnless((f.ROOT / "foundation-env.js").exists(), "parent v2 bridge not available")
    def test_real_bridge_two_rounds_public_delta_and_learnable_ask(self):
        env = f.NodeEnvironment(max_actions=6000)
        worker = {"events": []}
        asked = False
        try:
            for round_index in range(2):
                state = env.state("reset", seed=417 + round_index,
                                  **({"continue": True} if round_index else {}))
                self.assertEqual(state["events"], [])
                worker["events"] = []
                steps = 0
                while not state["terminal"]:
                    options = [i for i, action in enumerate(state["actions"]) if action["kind"] == "ask"]
                    if options and not asked:
                        player = state["player"]
                        next_state = env.state("step", index=options[0])
                        self.assertEqual(next_state["player"], player)
                        self.assertTrue(any(token[68] == 1 for token in next_state["events"]))
                        asked = True
                    else:
                        next_state = env.state("heuristic")
                    f.append_events(worker, next_state)
                    self.assertLessEqual(len(worker["events"]), 4096)
                    state = next_state
                    steps += 1
                    self.assertLess(steps, 6000)
                self.assertIn(state["result"]["winner"], (0, 1, None))
                self.assertGreater(len(worker["events"]), 0)
            self.assertTrue(asked)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
