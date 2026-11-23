"""CPU-only PPO protocol/objective checks; toy outcomes are not strength evidence."""

from collections import Counter
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import shutil
import unittest
from unittest import mock
import uuid

import numpy as np
import torch

import foundation as f
import team_ppo as p

torch.set_num_threads(1)


def observation(marker=0, finished=()):
    values = np.zeros(f.OBSERVATION_SIZE, np.float32)
    values[0] = marker
    for seat in finished:
        values[390 + seat] = .5
    return values


def actions(count=2):
    values = np.zeros((count, f.ACTION_SIZE), np.float32)
    values[:, -1] = np.arange(count)
    return values


def event(actor, kind="play"):
    values = np.zeros(f.HISTORY_SIZE, np.float32)
    values[54 + actor] = 1
    values[66 + ("play", "pass", "ask").index(kind)] = 1
    return values.tolist()


def decision(player=0, value=0, prefix=0, obs=None, index=0, old_lp=-.693147):
    return p.Decision(observation() if obs is None else obs, actions(), player,
                      prefix, "play", index, old_lp, value)


class ToyEnvironment:
    """Real protocol-shaped ask/same-seat/continued-return, six public seats."""

    def __init__(self, **kwargs):
        self.finished = False
        self.closed = False

    def state(self, op, **kwargs):
        events = []
        if op == "reset":
            continued = kwargs.get("continue", False)
            if continued and not self.finished:
                raise ValueError("continue requires completed previous round")
            self.seed = kwargs["seed"]
            self.turn = 0
            self.stage = "return" if continued else "ask"
            self.finished = False
        elif op in ("step", "heuristic"):
            if op == "step" and not 0 <= kwargs["index"] < 2:
                raise ValueError("bad legal index")
            if self.stage == "return":
                self.stage = "ask"
            elif self.stage == "ask":
                events = [event(0, "ask")]
                self.stage = "play"
            else:
                events = [event(self.turn)]
                self.turn += 1
                self.finished = self.turn == 6
        else:
            raise ValueError("unknown operation")
        player = min(self.turn, 5)
        kind = self.stage
        legal = []
        if not self.finished:
            for feature in actions():
                action = {"kind": kind, "ids": [] if kind == "ask" else [0],
                          "features": feature.tolist()}
                if kind == "ask":
                    action["target"] = 1
                legal.append(action)
        result = {
            "player": player, "observation": observation(player / 6).tolist(),
            "actions": legal, "events": events, "terminal": self.finished,
            "phase": "over" if self.finished else "returning" if kind == "return" else "playing",
            "result": {"winner": self.seed % 2} if self.finished else None,
        }
        f.validate_state(result)
        return result

    def close(self):
        self.closed = True


class PPOTests(unittest.TestCase):
    def setUp(self):
        self.directory = f.ROOT / f".ppo-tests-{uuid.uuid4().hex}"
        self.directory.mkdir()
        torch.manual_seed(17)
        self.source = self.directory / "v2.pt"
        self.source_model = f.ActionValue()
        f.atomic_torch_save(f.checkpoint_data(
            self.source_model, {"games": 8000, "updates": 20}), self.source)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def model(self):
        return p.warm_start(self.source, "cpu")

    def config(self, **changes):
        return {**p.CORE_DEFAULTS, "workers": 1, "rollout_games": 2, "epochs": 1,
                "batch_size": 32, **changes}

    def train_args(self, extra=(), checkpoint="run"):
        return p.arguments([
            "train", "--warm-start", str(self.source), "--games", "2",
            "--checkpoint-dir", str(self.directory / checkpoint),
            "--tensorboard-dir", str(self.directory / "tb"),
            "--device", "cpu", "--workers", "1", "--rollout-games", "2",
            "--epochs", "1", "--batch-size", "32", "--save-every", "1",
            "--log-every", "1", *extra,
        ])

    def test_warm_start_exact_scores_and_rankings_and_zero_value(self):
        model, metadata = self.model()
        states = [
            (observation(.2), actions(5), np.asarray([event(2)], np.float32)),
            (observation(-1), actions(3), np.empty((0, 71), np.float32)),
        ]
        before = f.score_many(self.source_model, states)
        after = f.score_many(model, states)
        for expected, actual in zip(before, after):
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(np.argsort(expected), np.argsort(actual))
        logits, values = model.policy_value(*p._tensor_states(model, states))
        torch.testing.assert_close(values, torch.zeros(2))
        for key, tensor in self.source_model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
        self.assertEqual(metadata["games"], 8000)
        self.assertEqual(metadata["ppoGames"], 0)
        self.assertEqual(metadata["lineage"]["actorTransfer"], "exact")

    def test_critic_does_not_receive_chosen_action_or_private_other_hands(self):
        model, _ = self.model()
        with torch.no_grad():
            model.value_head[-1].weight.fill_(.1)
        worker = {"env": ToyEnvironment(), "events": []}
        worker["state"] = worker["env"].state("reset", seed=1)
        worker["state"]["opponentPrivateHands"] = [[1, 2], [99, 100]]
        first = f.learner_job(worker)
        worker["state"]["opponentPrivateHands"] = [[500], [600]]
        second = f.learner_job(worker)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[2], second[2])
        states = [(np.asarray(first[0], np.float32), actions(2), first[2]),
                  (np.asarray(second[0], np.float32), actions(7) * 100, second[2])]
        _, values = model.policy_value(*p._tensor_states(model, states))
        torch.testing.assert_close(values[0], values[1], rtol=1e-6, atol=1e-7)

    def test_actor_and_critic_both_use_own_observation_and_seat_relative_public_history(self):
        model, _ = self.model()
        with torch.no_grad():
            model.value_head[-1].weight.fill_(.1)
        relative_zero = f.relative_history([event(0)], 0)
        relative_five = f.relative_history([event(5)], 5)
        states = [
            (observation(), actions(), relative_zero),
            (observation(1), actions(), relative_zero),
            (observation(), actions(), f.relative_history([event(1, "ask")], 0)),
            (observation(), actions(), relative_five),
        ]
        with torch.no_grad():
            logits, values = model.policy_value(*p._tensor_states(model, states))
        for changed in (1, 2):
            self.assertFalse(torch.allclose(logits[0], logits[changed]))
            self.assertFalse(torch.allclose(values[0], values[changed]))
        torch.testing.assert_close(logits[0], logits[3])
        torch.testing.assert_close(values[0], values[3])

    def test_team_potential_symmetry_and_telescoping_terminal_zero(self):
        absolute = np.array([.2, 0, .6, 0, 0, .8], np.float32)
        own = []
        for actor in (0, 2, 4):
            obs = observation()
            obs[390:396] = np.roll(absolute, -actor)
            own.append(p.potential(obs))
        self.assertAlmostEqual(own[0], own[1])
        self.assertAlmostEqual(own[1], own[2])
        enemy = observation()
        enemy[390:396] = np.roll(absolute, -1)
        self.assertAlmostEqual(p.potential(enemy), -own[0])
        trajectory = [observation(finished=(1,)), observation(finished=(0, 1)),
                      observation(finished=(0, 1, 2))]
        ep = p.Episode(np.empty((0, 71), np.float32),
                       [decision(obs=obs) for obs in trajectory], 0, 0)
        advantage, returns = p.seat_gae(ep, gae_lambda=1)
        self.assertAlmostEqual(float(returns[0]), 1 - p.potential(trajectory[0]), places=6)
        self.assertAlmostEqual(float(returns[-1]), 1 - p.potential(trajectory[-1]), places=6)
        # Terminal ranks are NOT treated as a final potential, even for trapped players.
        self.assertAlmostEqual(float(advantage[-1]), .9, places=6)
        unshaped = p.seat_gae(ep, gae_lambda=1, potential_scale=0)[1]
        self.assertAlmostEqual(float(unshaped[0]), 1)

    def test_per_seat_gae_no_enemy_or_episode_bootstrap(self):
        ep = p.Episode(np.empty((0, 71), np.float32),
                       [decision(0, .4), decision(1, 500), decision(0, .2),
                        decision(2, .7), decision(1, -50)], 0, 0)
        advantage, _ = p.seat_gae(ep, gae_lambda=1, potential_scale=0)
        self.assertAlmostEqual(float(advantage[0]), .6, places=6)
        self.assertAlmostEqual(float(advantage[2]), .8, places=6)
        self.assertAlmostEqual(float(advantage[3]), .3, places=6)
        self.assertAlmostEqual(float(advantage[4]), 49, places=6)
        enemy_changed = copy.deepcopy(ep)
        enemy_changed.decisions[1].old_value = -700
        changed = p.seat_gae(enemy_changed, gae_lambda=1, potential_scale=0)[0]
        np.testing.assert_array_equal(advantage[[0, 2, 3]], changed[[0, 2, 3]])
        unrelated = p.Episode(np.empty((0, 71), np.float32), [decision(0, 1000)], 1, 0)
        p.seat_gae(unrelated)
        np.testing.assert_array_equal(advantage, p.seat_gae(ep, gae_lambda=1, potential_scale=0)[0])

    def test_winning_episode_can_have_negative_earlier_gae_and_probability_update(self):
        ep = p.Episode(np.empty((0, 71), np.float32),
                       [decision(value=.9)] + [decision(value=.1) for _ in range(32)], 0, 0)
        advantage, _ = p.seat_gae(ep, potential_scale=0)
        self.assertLess(advantage[0], 0)
        self.assertGreater(advantage[-1], 0)
        logits = torch.zeros(2, requires_grad=True)
        optimizer = torch.optim.SGD([logits], lr=.1)
        old = logits.detach().log_softmax(0)[0]
        loss, _ = p.clipped_policy_loss(logits.log_softmax(0)[0:1], old.reshape(1),
                                       torch.tensor([advantage[0]]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertLess(float(logits.detach().softmax(0)[0]), .5)

    def test_ppo_clip_signs_detached_old_probabilities_and_zero_variance(self):
        new = torch.tensor([np.log(1.5), np.log(.5)], dtype=torch.float32, requires_grad=True)
        old = torch.zeros(2, requires_grad=True)
        loss, ratio = p.clipped_policy_loss(new, old, torch.tensor([1., -1.]))
        self.assertAlmostEqual(float(loss.detach()), -.2, places=6)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.zeros(2))
        self.assertIsNone(old.grad)
        np.testing.assert_allclose(ratio.detach().numpy(), [1.5, .5])
        normalized = p.normalize_advantages(torch.tensor([2., 2.]))
        self.assertTrue(torch.isfinite(normalized).all())

    def test_complete_collection_frozen_behavior_public_prefix_ask_and_return(self):
        model, _ = self.model()
        weights = copy.deepcopy(model.state_dict())
        workers = [{"env": ToyEnvironment(), "state": None, "finished": False} for _ in range(2)]
        config = self.config(workers=2, rollout_games=4)
        # Force full selfplay to cover every seat, including three team replicas.
        rng = mock.Mock(wraps=random.Random(1))
        rng.random.return_value = .1
        batch, coverage, number = p.collect_batch(model, workers, 4, rng, config, 3)
        self.assertEqual(number, 4)
        self.assertEqual(len(batch.episodes), 4)
        self.assertTrue(all(w["state"] is None for w in workers))
        self.assertGreater(coverage["decision:ask"], 0)
        self.assertGreater(coverage["decision:return"], 0)
        self.assertEqual(coverage["continuedRounds"], 2)
        self.assertEqual(set(d.player for d in batch.episodes[0].decisions), set(range(6)))
        for key, value in weights.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        for episode in batch.episodes:
            for d in episode.decisions:
                self.assertEqual(len(episode.history(d)), d.prefix)
                states = [(d.observation, d.actions, episode.history(d))]
                logits, values = model.policy_value(*p._tensor_states(model, states))
                lp = torch.distributions.Categorical(logits=logits[0] / config["temperature"]).log_prob(
                    torch.tensor(d.index))
                self.assertAlmostEqual(float(lp.detach()), d.old_logprob, places=6)
                self.assertAlmostEqual(float(values[0].detach()), d.old_value, places=6)
        first = batch.episodes[0]
        self.assertEqual(first.decisions[0].prefix, 0)
        self.assertEqual(first.decisions[1].player, first.decisions[0].player)
        self.assertEqual(first.decisions[1].prefix, 1)
        self.assertEqual(batch.episodes[1].decisions[0].prefix, 0)

    def test_consumed_batch_no_stale_replay_and_training_mode_preserved(self):
        model, _ = self.model()
        worker = {"env": ToyEnvironment(), "state": None, "finished": False}
        config = self.config(epochs=2, batch_size=8)
        rng = random.Random(2)
        batch, _, _ = p.collect_batch(model, [worker], 2, rng, config, 0, league=[model])
        old = [d.old_logprob for ep in batch.episodes for d in ep.decisions]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        modes = []
        handle = model.history.register_forward_hook(lambda module, args, out: modes.append(module.training))
        result = p.update_batch(model, optimizer, batch, 0, config, rng)
        handle.remove()
        self.assertGreater(result["optimizerSteps"], 0)
        self.assertTrue(all(modes))
        self.assertFalse(model.training)
        self.assertEqual(old, [d.old_logprob for ep in batch.episodes for d in ep.decisions])
        self.assertTrue(batch.consumed)
        with self.assertRaisesRegex(ValueError, "consumed"):
            p.update_batch(model, optimizer, batch, 0, config, rng)
        with self.assertRaisesRegex(ValueError, "stale"):
            p.RolloutBatch(batch.episodes, 0).validate(1)
        self.assertTrue(all(torch.isfinite(value).all() for value in model.parameters()))

    def test_checkpoint_rng_optimizer_metadata_and_operational_resume(self):
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            result = p.train(self.train_args())
        self.assertEqual(result["games"], 8002)
        self.assertEqual(result["ppoGames"], 2)
        metric_rows = [json.loads(line) for line in
                       (self.directory / "run" / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(metric_rows), 1)
        self.assertEqual(metric_rows[0]["sourceGames"], 8000)
        self.assertEqual(metric_rows[0]["ppoGames"], 2)
        self.assertTrue(metric_rows[0]["traceAudit"]["passed"])
        self.assertEqual(metric_rows[0]["advantageNormalization"], "once_per_complete_batch")
        latest = self.directory / "run" / "latest.pt"
        model, data = p.load_checkpoint(latest)
        self.assertEqual(data["metadata"]["algorithm"], p.ALGORITHM)
        self.assertEqual(data["metadata"]["methods"]["algorithm"], p.ALGORITHM)
        self.assertNotIn("replay", data)
        optimizer = torch.optim.Adam(model.parameters())
        rng = random.Random()
        p.restore_training(data, optimizer, rng, torch.device("cpu"))
        expected = (rng.random(), random.random(), np.random.random(), torch.rand(3))
        p.restore_training(data, optimizer, rng, torch.device("cpu"))
        actual = (rng.random(), random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[:3], actual[:3])
        torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertTrue(optimizer.state)
        for state in optimizer.state.values():
            self.assertTrue(all(torch.isfinite(t).all() for t in state.values() if isinstance(t, torch.Tensor)))
        args = p.arguments([
            "train", "--resume", str(latest), "--target-games", "8004",
            "--checkpoint-dir", str(self.directory / "run"),
            "--tensorboard-dir", str(self.directory / "tb-resume"),
            "--device", "cpu", "--save-every", "3", "--log-every", "4",
        ])
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            result = p.train(args)
        self.assertEqual(result["games"], 8004)
        self.assertEqual(result["ppoGames"], 4)
        self.assertEqual(result["config"]["workers"], 1)
        self.assertEqual(result["saveEvery"], 3)
        self.assertEqual(result["logEvery"], 4)
        self.assertEqual(len(result["resumeCadenceOverrides"]), 1)
        self.assertEqual(len((self.directory / "run" / "metrics.jsonl").read_text().splitlines()), 2)
        initial, initial_data = p.load_checkpoint(self.directory / "run" / "initial.pt")
        self.assertEqual(initial_data["metadata"]["ppoGames"], 0)
        self.assertEqual(initial_data["metadata"]["games"], 8000)
        self.assertEqual(result["initialSha256"], f.sha256_file(self.directory / "run" / "initial.pt"))
        bad = p.arguments(["train", "--resume", str(latest), "--potential-scale", "0"])
        with self.assertRaisesRegex(ValueError, "potential-scale"):
            p.resolve_config(bad, result)
        corrupt = copy.deepcopy(data)
        corrupt["optimizer"]["state"][next(iter(corrupt["optimizer"]["state"]))]["exp_avg"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "optimizer"):
            p.restore_training(corrupt, optimizer, rng, torch.device("cpu"))

    def test_from_scratch_has_no_inherited_games_and_preserves_initial_on_resume(self):
        directory = self.directory / "scratch-run"
        args = p.arguments([
            "train", "--from-scratch", "--games", "2", "--seed", "1234",
            "--checkpoint-dir", str(directory), "--tensorboard-dir", str(self.directory / "scratch-tb"),
            "--device", "cpu", "--workers", "1", "--rollout-games", "2", "--epochs", "1",
            "--save-every", "2", "--log-every", "2"])
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            result = p.train(args)
        self.assertEqual(result["sourceGames"], 0)
        self.assertEqual(result["games"], 2)
        self.assertIsNone(result["sourceSha256"])
        initial, initial_data = p.load_checkpoint(directory / "initial.pt")
        self.assertEqual(initial_data["metadata"]["games"], 0)
        self.assertEqual(initial_data["metadata"]["lineage"]["actorTransfer"], "random")
        torch.manual_seed(1234)
        expected = p.ActorCritic()
        for key, tensor in initial.state_dict().items():
            torch.testing.assert_close(tensor, expected.state_dict()[key], rtol=0, atol=0)
        digest = f.sha256_file(directory / "initial.pt")
        latest = directory / "latest.pt"
        resumed_args = p.arguments([
            "train", "--resume", str(latest), "--games", "2", "--device", "cpu",
            "--checkpoint-dir", str(directory), "--tensorboard-dir", str(self.directory / "scratch-tb")])
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            resumed = p.train(resumed_args)
        self.assertEqual(resumed["games"], 4)
        self.assertEqual(resumed["ppoGames"], 4)
        self.assertEqual(f.sha256_file(directory / "initial.pt"), digest)
        corrupt = copy.deepcopy(initial_data)
        corrupt["metadata"]["sourceSha256"] = "a" * 64
        corrupt["metadata"]["lineage"]["sourceSha256"] = "a" * 64
        f.atomic_torch_save(corrupt, self.directory / "false-scratch.pt")
        with self.assertRaisesRegex(ValueError, "inherited"):
            p.load_checkpoint(self.directory / "false-scratch.pt")

    def test_prepared_distilled_start_trains_without_importing_optimizer_and_resumes(self):
        model, metadata = self.model()
        metadata.update(initialization="source-behavior-distilled v2 actor",
                        config=dict(p.CORE_DEFAULTS),
                        behaviorDistillation={
                            "sourceSha256": metadata["sourceSha256"], "behaviorPreserved": True,
                            "temperature": .2,
                            "confirmation": {"passed": True, "meanKL": .01, "topAgreement": 1.0}})
        metadata["lineage"]["actorTransfer"] = "source-behavior-distilled"
        prepared = self.directory / "prepared.pt"
        f.atomic_torch_save(p.checkpoint_data(model, metadata), prepared)
        args = p.arguments([
            "train", "--prepared-start", str(prepared), "--games", "2",
            "--checkpoint-dir", str(self.directory / "prepared-run"),
            "--tensorboard-dir", str(self.directory / "prepared-tb"),
            "--device", "cpu", "--workers", "1", "--rollout-games", "2",
            "--epochs", "1", "--save-every", "2", "--log-every", "2"])
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            result = p.train(args)
        self.assertEqual(result["ppoGames"], 2)
        self.assertEqual(result["preparedSha256"], f.sha256_file(prepared))
        latest = self.directory / "prepared-run" / "latest.pt"
        _, data = p.load_checkpoint(latest)
        self.assertIn("distillation", data["metadata"]["methods"]["initialization"])
        args = p.arguments([
            "train", "--resume", str(latest), "--games", "2",
            "--checkpoint-dir", str(latest.parent), "--tensorboard-dir", str(self.directory / "prepared-tb"),
            "--device", "cpu"])
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            resumed = p.train(args)
        self.assertEqual(resumed["ppoGames"], 4)
        wrong = p.arguments([
            "train", "--prepared-start", str(prepared), "--games", "0",
            "--temperature", ".1", "--device", "cpu",
            "--checkpoint-dir", str(self.directory / "wrong-temperature"),
            "--tensorboard-dir", str(self.directory / "wrong-tb")])
        with self.assertRaisesRegex(ValueError, "temperature"):
            p.train(wrong)

    def test_algorithm_guards_lineage_and_foundation_inference(self):
        model, metadata = self.model()
        path = self.directory / "ppo.pt"
        f.atomic_torch_save(p.checkpoint_data(model, metadata), path)
        loaded, data = p.load_checkpoint(path)
        request = {"op": "score", "observation": observation().tolist(),
                   "actions": actions(3).tolist(), "history": []}
        result = f.handle_request(loaded, request)
        np.testing.assert_array_equal(result["scores"], f.score_many(model, [
            (request["observation"], request["actions"], [])])[0])
        self.assertEqual(f.handle_request(loaded, {"op": "info"})["algorithm"], p.ALGORITHM)
        with self.assertRaisesRegex(ValueError, "team-ppo"):
            p.load_checkpoint(self.source)
        with self.assertRaisesRegex(ValueError, "warm-start"):
            p.warm_start(path)
        data["metadata"]["ppoGames"] += 1
        f.atomic_torch_save(data, path)
        with self.assertRaisesRegex(ValueError, "cumulative"):
            p.load_checkpoint(path)
        args = self.train_args()
        args.checkpoint_dir = f.ROOT / "foundation" / "checkpoints"
        with self.assertRaisesRegex(ValueError, "v1/v2"):
            p._protect_outputs(args)

    def test_interrupt_discards_entire_incomplete_batch_without_fake_games(self):
        class InterruptedToy(ToyEnvironment):
            decisions = 0

            def state(self, op, **kwargs):
                if op == "step":
                    type(self).decisions += 1
                    if type(self).decisions == 3:
                        raise KeyboardInterrupt
                return super().state(op, **kwargs)

        with mock.patch.object(f, "NodeEnvironment", InterruptedToy), redirect_stdout(io.StringIO()):
            result = p.train(self.train_args())
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["games"], 8000)
        self.assertEqual(result["ppoGames"], 0)
        self.assertEqual(result["updates"], 0)
        self.assertEqual(result["discardedBatchDecisions"], 3)
        self.assertEqual(result["discardedInFlightEpisodes"], 1)
        initial, _ = p.load_checkpoint(self.directory / "run" / "initial.pt")
        latest, data = p.load_checkpoint(self.directory / "run" / "latest.pt")
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(value, latest.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(data["optimizer"]["state"])

    def test_bounded_league_retains_initial_and_immutable_milestone_actual_count(self):
        f.atomic_torch_save(f.checkpoint_data(
            self.source_model, {"games": 997, "updates": 20}), self.source)
        args = self.train_args(["--games", "14"])
        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment), redirect_stdout(io.StringIO()):
            result = p.train(args)
        self.assertEqual(result["games"], 1011)
        self.assertLessEqual(len(result["snapshotPool"]), 5)
        self.assertTrue((self.directory / "run" / "initial.pt").exists())
        self.assertLessEqual(len(list((self.directory / "run").glob("snapshot-*.pt"))), 5)
        milestone = self.directory / "run" / "milestone-000001001.pt"
        self.assertTrue(milestone.exists())
        _, data = p.load_checkpoint(milestone)
        self.assertEqual(data["metadata"]["games"], 1001)
        self.assertEqual(data["metadata"]["ppoGames"], 4)

    def test_interrupted_optimizer_transaction_rolls_back_weights_and_rng(self):
        real_update = p.update_batch

        def interrupt_after_update(*args, **kwargs):
            real_update(*args, **kwargs)
            raise KeyboardInterrupt

        with mock.patch.object(f, "NodeEnvironment", ToyEnvironment), \
                mock.patch.object(p, "update_batch", interrupt_after_update), \
                redirect_stdout(io.StringIO()):
            result = p.train(self.train_args())
        self.assertEqual(result["games"], 8000)
        self.assertEqual(result["updates"], 0)
        self.assertEqual(result["discardedCompleteEpisodes"], 2)
        self.assertGreater(result["discardedBatchDecisions"], 0)
        initial, _ = p.load_checkpoint(self.directory / "run" / "initial.pt")
        latest, data = p.load_checkpoint(self.directory / "run" / "latest.pt")
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(value, latest.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(data["optimizer"]["state"])


if __name__ == "__main__":
    unittest.main()
