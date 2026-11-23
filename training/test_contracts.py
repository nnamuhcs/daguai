"""Production trace/math/cache tests. Toy preference is plumbing, not expertise."""

import copy
from contextlib import redirect_stderr
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

import contracts as c
import foundation as f
import team_ppo as p
from test_team_ppo import ToyEnvironment, actions, decision, event, observation
from trace_audit import audit_batch, replay_predictions
import trace_audit as t

torch.set_num_threads(1)


def config(**changes):
    return {**p.CORE_DEFAULTS, "workers": 1, "rollout_games": 2,
            "epochs": 2, "batch_size": 64, **changes}


def episode(ds, winner=0, model=None, number=0, events=None):
    digest = c.model_digest(model) if model is not None else "a" * 64
    events = np.asarray(events, np.float32) if events is not None else np.empty((0, 71), np.float32)
    for i, d in enumerate(ds):
        d.contract_version = c.CONTRACT_VERSION
        d.episode_id = number
        d.global_step = i
        d.action_identities = tuple(c.action_identity({"kind": d.kind, "ids": [j]})
                                    for j in range(len(d.actions)))
        d.behavior_digest = digest
        d.legal_support_digest = c.legal_support_digest(d.action_identities, d.actions)
    return c.finalize_episode(c.Episode(
        events, ds, winner, 0, contract_version=c.CONTRACT_VERSION, episode_id=number,
        behavior_digest=digest, controlled_seats=tuple(range(6)), global_steps=len(ds) + 3),
        p.potential)


def batch(es):
    digest = es[0].behavior_digest
    return c.RolloutBatch(es, 0, behavior_digest_before=digest, behavior_digest_after=digest)


class ContractTests(unittest.TestCase):
    def test_persistent_private_jsonl_complete_contract_roundtrip_and_bounds(self):
        directory = f.ROOT / f".trace-export-tests-{uuid.uuid4().hex}"
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory)
        path = directory / "decisions.jsonl"
        seed = 20_000_031
        cfg = config(seed=seed)
        model = p.ActorCritic().eval()
        rng = mock.Mock(wraps=random.Random(seed))
        rng.random.return_value = .1
        rollout, _, _ = p.collect_batch(
            model, [{"env": ToyEnvironment(), "state": None}], 2, rng, cfg, 0)
        rollout.episodes[0].simulator_snapshot = {"hidden_hands": [[999]]}
        rollout.episodes[0].decisions[0].other_actors_private = [[999]]
        summary = t.export_trace_jsonl(path, rollout, cfg, seed)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        original = path.read_bytes()
        rows = [json.loads(line) for line in original.splitlines()]
        manifest = rows[0]
        self.assertEqual(manifest["trace_schema_version"], 1)
        self.assertEqual(manifest["classification"], "training-private-own-public-diagnostics")
        self.assertTrue(manifest["not_public_or_inference_input"])
        self.assertIn("[:decision.prefix]", manifest["history_rule"])
        self.assertEqual(summary["records"], 1 + 2 + manifest["decisions"])
        self.assertEqual(summary["bytes"], len(original))
        self.assertNotIn("hidden_hands", original.decode())
        self.assertNotIn("other_actors_private", original.decode())
        restored = []
        for row in rows[1:]:
            self.assertEqual(row["trace_schema_version"], 1)
            if row["record_type"] == "episode":
                values = row["episode"]
                values["events"] = np.asarray(values["events"], np.float32).reshape(-1, f.HISTORY_SIZE)
                values["controlled_seats"] = tuple(values["controlled_seats"])
                restored.append(c.Episode(decisions=[], **values))
            else:
                values = row["decision"]
                self.assertEqual(values.pop("team"), values["player"] % 2)
                self.assertEqual(values.pop("chosen_action_identity"),
                                 values["action_identities"][values["index"]])
                values["observation"] = np.asarray(values["observation"], np.float32)
                values["actions"] = np.asarray(values["actions"], np.float32)
                values["action_identities"] = tuple(values["action_identities"])
                self.assertEqual(row["decision_index"], len(restored[-1].decisions))
                restored[-1].decisions.append(c.Decision(**values))
        replay = c.RolloutBatch(
            restored, manifest["behavior_version"],
            behavior_digest_before=manifest["behavior_digest_before"],
            behavior_digest_after=manifest["behavior_digest_after"])
        self.assertTrue(audit_batch(replay, model, cfg)["passed"])
        for expected, actual in zip(rollout.episodes, restored):
            np.testing.assert_array_equal(expected.events, actual.events)
            for before, after in zip(expected.decisions, actual.decisions):
                np.testing.assert_array_equal(before.observation, after.observation)
                np.testing.assert_array_equal(before.actions, after.actions)
                np.testing.assert_array_equal(expected.history(before), actual.history(after))
        self.assertLess(restored[0].decisions[0].prefix, len(restored[0].events))
        with self.assertRaises(FileExistsError):
            t.export_trace_jsonl(path, rollout, cfg, seed)
        self.assertEqual(path.read_bytes(), original)
        for invalid in (1_000_000_000, 2_000_000_000, 20_999_999):
            with self.assertRaisesRegex(ValueError, "20m debug"):
                t.export_trace_jsonl(path, rollout, {**cfg, "seed": invalid}, invalid)
        limited = directory / "limited.jsonl"
        with mock.patch.object(t, "MAX_TRACE_JSONL_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "size limit"):
                t.export_trace_jsonl(limited, rollout, cfg, seed)
        self.assertFalse(limited.exists())
        with mock.patch("sys.argv", ["trace_audit.py", "--output", str(path)]), \
                mock.patch.object(f, "NodeEnvironment") as environment, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            t.main()
        environment.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_all_actors_win_loss_draw_terminal_assignment_and_mc(self):
        for winner in (0, 1, None):
            with self.subTest(winner=winner):
                ds = [decision(actor, value=.1 * actor) for actor in range(6)]
                ds.extend(decision(actor, value=-.2 * actor) for actor in range(1, 6))
                ep = episode(ds, winner)
                report = audit_batch(batch([ep]))
                self.assertTrue(report["passed"])
                self.assertEqual(report["terminalAssignmentsByActor"], {str(i): 1 for i in range(6)})
                advantage, targets = p.seat_gae(ep, gae_lambda=1)
                for d, a, r in zip(ds, advantage, targets):
                    self.assertAlmostEqual(float(r), f.team_reward(d.player, winner), places=6)
                    self.assertAlmostEqual(float(a), f.team_reward(d.player, winner) - d.old_value, places=6)

    def test_other_actors_finish_between_own_turns_early_exit_ask_return(self):
        ds = [decision(0, .2, prefix=0), decision(1, .4, prefix=1),
              decision(0, .1, prefix=2, obs=observation(finished=(0, 2, 3))),
              decision(0, -.3, prefix=3, obs=observation(finished=(0, 2, 3))),
              decision(2, 20, prefix=4), decision(3, -9, prefix=5)]
        ds[2].kind = "ask"
        ds[3].kind = "return"
        ep = episode(ds, events=[event(i % 6) for i in range(8)])
        self.assertEqual(ds[0].next_actor_index, 2)
        self.assertEqual(ds[2].next_actor_index, 3)
        self.assertEqual(ds[1].next_actor_index, None)
        self.assertEqual(ds[1].terminal_reward, -1)
        self.assertEqual(ds[0].elapsed_events, 2)
        self.assertAlmostEqual(ds[0].potential_successor, .1)
        self.assertTrue(audit_batch(batch([ep]))["passed"])
        shaped = p.seat_gae(ep, gae_lambda=1)[1]
        unshaped = p.seat_gae(ep, gae_lambda=1, potential_scale=0)[1]
        for d, a, b in zip(ds, shaped, unshaped):
            self.assertAlmostEqual(float(a - b), -p.potential(d.observation), places=6)

    def test_trapped_rank_not_used_as_terminal_potential(self):
        ep = episode([decision(obs=observation(finished=(0, 2)))])
        self.assertEqual(ep.decisions[0].potential_successor, 0)
        self.assertAlmostEqual(float(p.seat_gae(ep, gae_lambda=1)[1][0]), .8, places=6)

    def test_enforceable_corruption_checks(self):
        ep = episode([decision(), decision(1), decision(0)])
        edits = [
            lambda e: setattr(e.decisions[0], "next_actor_index", 1),
            lambda e: setattr(e.decisions[0], "terminal_reward", 1),
            lambda e: setattr(e.decisions[1], "terminal_reward", 1),
            lambda e: setattr(e.decisions[0], "prefix", 100),
            lambda e: setattr(e.decisions[0], "old_value", float("nan")),
            lambda e: setattr(e, "controlled_seats", (0,)),
            lambda e: setattr(e, "terminal", False),
            lambda e: setattr(e.decisions[0], "elapsed_steps", 500),
            lambda e: setattr(e.decisions[0], "potential_successor", .4),
            lambda e: setattr(e.decisions[0], "actions", e.decisions[0].actions[::-1].copy()),
            lambda e: setattr(e, "temperature", .1),
        ]
        for edit in edits:
            corrupt = copy.deepcopy(ep)
            edit(corrupt)
            with self.assertRaisesRegex(ValueError, "trace audit failed"):
                audit_batch(batch([corrupt]))
            self.assertFalse(audit_batch(batch([corrupt]), enforce=False)["passed"])

    def test_exact_action_identity_preserves_cards_targets_and_order(self):
        first = {"kind": "ask", "ids": [], "target": 2, "features": [5]}
        second = {"target": 2, "kind": "ask", "features": [7], "ids": []}
        self.assertEqual(c.action_identity(first), c.action_identity(second))
        self.assertNotEqual(c.action_identity(first), c.action_identity({**first, "target": 4}))
        self.assertNotEqual(c.action_identity({"kind": "play", "ids": [3, 4]}),
                            c.action_identity({"kind": "play", "ids": [4, 3]}))

    def test_gamma_rejected_math_cli_and_legacy_resume(self):
        with self.assertRaisesRegex(ValueError, "gamma must equal 1"):
            p.seat_gae(episode([decision()]), gamma=.99)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p.arguments(["train", "--warm-start", "unused.pt", "--gamma", ".99"])
        saved = {"config": {**p.CORE_DEFAULTS, "gamma": .99}, "saveEvery": 1000, "logEvery": 100}
        args = p.arguments(["train", "--resume", "unused.pt"])
        with self.assertRaisesRegex(ValueError, "gamma must equal 1"):
            p.resolve_config(args, saved)

    def test_legacy_config_migration_records_guard_not_temperature_change(self):
        saved_config = dict(p.CORE_DEFAULTS)
        saved_config.pop("target_kl")
        saved = {"config": saved_config, "saveEvery": 1000, "logEvery": 100}
        resolved, changes = p.resolve_config(p.arguments(["train", "--resume", "unused.pt"]), saved)
        self.assertEqual(changes["migrationDefaults"], {"target_kl": .015})
        self.assertEqual(resolved["temperature"], .2)

    def test_global_normalization_zero_variance_has_no_fabricated_signal(self):
        x = torch.tensor([1., 1., 10., 10.])
        normalized = p.normalize_advantages(x)
        torch.testing.assert_close(normalized, torch.tensor([-1., -1., 1., 1.]))
        torch.testing.assert_close(p.normalize_advantages(torch.tensor([2., 2.])), torch.zeros(2))

    def test_legal_support_replay_ratios_one_sizes_1_2_100_1000(self):
        torch.manual_seed(23)
        model = p.ActorCritic().eval()
        states = [(observation(), actions(n), np.asarray([event(2)], np.float32))
                  for n in (1, 2, 100, 1000)]
        choices = p.sample_states(model, states)
        ds = [p.Decision(o, a, 0, 1, "play", i, lp, value)
              for (o, a, _), (i, lp, value) in zip(states, choices)]
        ep = episode(ds, model=model, events=[event(2)])
        report = audit_batch(batch([ep]), model, config())
        self.assertLessEqual(report["maximumRatioReplayError"], 1e-5)
        self.assertEqual(report["errors"], [])

    def test_collection_only_actual_controlled_seats_and_behavior_immutability(self):
        model = p.ActorCritic().eval()
        frozen = p.ActorCritic().eval()
        for mode_draw in (.7, .9):
            rng = mock.Mock(wraps=random.Random(23))
            rng.random.return_value = mode_draw
            rng.randrange.return_value = 0
            result, coverage, _ = p.collect_batch(
                model, [{"env": ToyEnvironment(), "state": None}], 1, rng,
                config(rollout_games=1, mixed_teammate_probability=1), 0, league=[frozen])
            ep = result.episodes[0]
            self.assertEqual(set(d.player for d in ep.decisions), set(ep.controlled_seats))
            self.assertTrue(set(ep.controlled_seats) < {0, 2, 4})
            self.assertEqual(result.behavior_digest_before, result.behavior_digest_after)
            self.assertEqual(coverage["mixedTeamGames"], 1)
            self.assertTrue(audit_batch(result, model, config())["passed"])

    def test_changed_frozen_behavior_is_explicit_failure(self):
        model = p.ActorCritic().eval()

        class MutatingToy(ToyEnvironment):
            def state(self, op, **kw):
                if op == "step":
                    with torch.no_grad():
                        next(model.parameters()).add_(.01)
                return super().state(op, **kw)

        rng = mock.Mock(wraps=random.Random(2))
        rng.random.return_value = .1
        with self.assertRaisesRegex(ValueError, "behavior digest changed"):
            p.collect_batch(model, [{"env": MutatingToy(), "state": None}], 1, rng,
                            config(rollout_games=1), 0)

    def test_cache_full_prefix_equivalence_seats_ask_noevent_return_resets(self):
        torch.manual_seed(19)
        model = p.ActorCritic().eval()
        with torch.no_grad():
            model.value_head[-1].weight.fill_(.2)
        cache = p.SeatHistoryCache(model)
        public = []
        round_id, version = 0, 0
        for step, seats in enumerate(((0, 1), (2, 0), (0, 3), (4, 5), (1, 2), (0, 5))):
            if step == 2:  # same actor ask; zero public tokens in private return phase.
                pass
            else:
                public.append(event(step % 6, "ask" if step == 1 else "play"))
            requests = [p.CachedState(observation(seat / 6), actions(3 + seat),
                                     public, seat, "worker", round_id) for seat in seats]
            actual, values = cache.policy_value(requests, version)
            states = [(s.observation, s.actions, f.relative_history(public, s.player))
                      for s in requests]
            with torch.no_grad():
                expected, full_values = model.policy_value(*p._tensor_states(model, states))
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(values, full_values, atol=1e-6, rtol=1e-5)
        self.assertLess(cache.tokens, cache.full_prefix_tokens)
        for round_id, version in ((1, 0), (1, 1)):
            with torch.no_grad():
                if version:
                    model.history.weight_hh_l0.add_(.1)
            public = [event(4)]
            request = p.CachedState(observation(), actions(), public, 0, "worker", round_id)
            actual, values = cache.policy_value([request], version)
            with torch.no_grad():
                expected, full = model.policy_value(*p._tensor_states(
                    model, [(request.observation, request.actions, f.relative_history(public, 0))]))
            torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(values, full, atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_cache_matches_heterogeneous_full_prefix_replay_with_tf32_initially_enabled(self):
        torch.manual_seed(701)
        model = p.ActorCritic().cuda().eval()
        with torch.no_grad():
            # Amplify one recurrent unit to expose reduced-precision drift.
            model.value_head[0].weight.zero_()
            model.value_head[0].weight[0, 256] = 1
            model.value_head[0].bias.zero_()
            model.value_head[0].bias[0] = 1
            model.value_head[-1].weight.zero_()
            model.value_head[-1].weight[0, 0] = 1000
        rng = np.random.default_rng(701)
        public = np.zeros((160, f.HISTORY_SIZE), np.float32)
        for index in range(len(public)):
            public[index, rng.integers(54)] = 1 / 3
            public[index, 54 + index % 6] = 1
            public[index, 66 + index % 3] = 1
        old_recurrent = torch.backends.cudnn.allow_tf32
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = True
            cache = p.SeatHistoryCache(model)
            cached, states = [], []
            for prefix in range(8, 161, 8):
                obs, legal = observation(.2), actions(3)
                _, value = cache.policy_value(
                    [p.CachedState(obs, legal, public[:prefix], 0, 0, 0)], 0)
                cached.append(value[0])
                states.append((obs, legal, f.relative_history(public[:prefix], 0)))
            with torch.no_grad():
                _, full = model.policy_value(*p._tensor_states(model, states))
            torch.testing.assert_close(torch.stack(cached), full, rtol=0, atol=3e-4)
        finally:
            torch.backends.cudnn.allow_tf32 = old_recurrent
            torch.backends.cuda.matmul.allow_tf32 = old_matmul

    def test_cache_no_global_absolute_state_and_no_future_event(self):
        model = p.ActorCritic().eval()
        cache = p.SeatHistoryCache(model)
        public = [event(0), event(1)]
        requests = [p.CachedState(observation(), actions(), public, i, "worker", 0) for i in range(6)]
        logits, _ = cache.policy_value(requests, 0)
        self.assertEqual(cache.tokens, 12)  # Six relative views, not <2 tokens/decision.
        self.assertFalse(torch.allclose(logits[0], logits[1]))
        with self.assertRaisesRegex(ValueError, "shrank"):
            cache.policy_value([p.CachedState(observation(), actions(), [], 0, "worker", 0)], 0)

    def test_cache_generic_foundation_frozen_opponent_scores_unchanged(self):
        model = f.ActionValue().eval()
        cache = p.SeatHistoryCache(model)
        public = [event(1), event(4, "ask")]
        requests = [p.CachedState(observation(), actions(5 + seat), public, seat, seat, 0)
                    for seat in (0, 2, 5)]
        scores, values = cache.policy_value(requests, "frozen-snapshot")
        expected = f.score_many(model, [
            (s.observation, s.actions, f.relative_history(public, s.player)) for s in requests])
        for actual, full in zip(scores, expected):
            np.testing.assert_allclose(actual.detach().numpy(), full, rtol=1e-5, atol=1e-6)
            self.assertEqual(int(actual.argmax()), int(np.argmax(full)))
        torch.testing.assert_close(values, torch.zeros(3))

    def test_kl_guard_halts_remaining_epochs_and_metrics_are_real(self):
        torch.manual_seed(41)
        model = p.ActorCritic().eval()
        rng = mock.Mock(wraps=random.Random(1))
        rng.random.return_value = .1
        cfg = config(epochs=9, target_kl=1e-12)
        rollout, _, _ = p.collect_batch(
            model, [{"env": ToyEnvironment(), "state": None}], 2, rng, cfg, 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        with mock.patch.object(p, "normalize_advantages", wraps=p.normalize_advantages) as normalize:
            metrics = p.update_batch(model, optimizer, rollout, 0, cfg, random.Random(7))
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual(normalize.call_args[0][0].shape[0], metrics["transitions"])
        self.assertEqual(metrics["epochsRun"], 1)
        self.assertEqual(metrics["stopReason"], "target_kl_exceeded")
        self.assertGreater(metrics["approxKL"], metrics["targetKL"])
        self.assertEqual(metrics["optimizerSteps"], 1)
        for key in ("policyLoss", "valueLoss", "totalLoss", "entropy", "normalizedEntropy",
                    "pmax", "effectiveSupport", "gradNorm", "degenerateMinibatchFraction"):
            self.assertTrue(np.isfinite(metrics[key]))
        self.assertEqual(metrics["advantageNormalization"], "once_per_complete_batch")
        json.dumps(metrics, allow_nan=False)

    def test_zero_target_variance_explained_variance_explicitly_undefined(self):
        model = p.ActorCritic().eval()
        d = decision()
        state = (d.observation, d.actions, np.empty((0, 71), np.float32))
        d.index, d.old_logprob, d.old_value = p.sample_states(model, [state])[0]
        ep = episode([d], winner=None, model=model)
        result = p.update_batch(model, torch.optim.Adam(model.parameters(), lr=1e-4),
                                batch([ep]), 0, config(epochs=1), random.Random(3))
        self.assertIsNone(result["explainedVariance"])
        self.assertEqual(result["explainedVarianceUndefinedReason"], "zero_target_variance")
        self.assertEqual(result["degenerateMinibatchFraction"], 1)

    def test_every_epoch_shuffles_all_actors_and_episodes_not_separate_chains(self):
        model = p.ActorCritic().eval()
        rng = mock.Mock(wraps=random.Random(33))
        rng.random.return_value = .1
        cfg = config(epochs=3, target_kl=100, batch_size=3)
        rollout, _, _ = p.collect_batch(
            model, [{"env": ToyEnvironment(), "state": None}], 2, rng, cfg, 0)
        shuffle_input = []

        class RecordingRandom(random.Random):
            def shuffle(self, order):
                shuffle_input.append(tuple(sorted(order)))
                super().shuffle(order)

        result = p.update_batch(model, torch.optim.Adam(model.parameters(), lr=1e-5),
                                rollout, 0, cfg, RecordingRandom(9))
        self.assertEqual(result["epochsRun"], 3)
        self.assertEqual(shuffle_input, [tuple(range(result["transitions"]))] * 3)

    def test_existing_40_game_checkpoint_stays_loadable_and_rng_untouched(self):
        path = f.ROOT / "reports" / "blueprint-evidence" / "ppo-40.pt"
        if not path.is_file():
            self.skipTest("immutable verification checkpoint not available")
        torch.manual_seed(71)
        before = torch.get_rng_state().clone()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        model, data = p.load_checkpoint(path)
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(np.random.get_state()[0], numpy_before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.assertEqual(np.random.get_state()[2:], numpy_before[2:])
        self.assertEqual(data["metadata"]["ppoGames"], 40)
        self.assertEqual(data["metadata"]["games"], 1040)
        self.assertEqual(data["metadata"]["config"]["temperature"], .2)
        state = (observation(), actions(4), np.empty((0, 71), np.float32))
        logits, _ = model.policy_value(*p._tensor_states(model, [state]))
        np.testing.assert_array_equal(logits[0].detach().numpy(), f.score_many(model, [state])[0])

    def test_cpu_toy_preference_improves_actual_ppo_three_fixed_seeds(self):
        """Balanced on-policy one-decision trials with terminal +/- team rewards."""
        for seed in (7, 17, 27):
            torch.manual_seed(seed)
            model = p.ActorCritic().eval()
            cfg = config(rollout_games=64, batch_size=64, epochs=2, entropy_coef=0,
                         potential_scale=0, target_kl=.015)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            rng = random.Random(seed)
            state = (observation(), actions(), np.empty((0, 71), np.float32))

            def preference():
                with torch.no_grad():
                    score, _ = model.policy_value(*p._tensor_states(model, [state]))
                    return float((score[0] / cfg["temperature"]).softmax(0)[1])

            before = preference()
            for version in range(6):
                selected = p.sample_states(model, [state] * 64, cfg["temperature"])
                es = []
                for i, (index, lp, value) in enumerate(selected):
                    d = decision(index=index, old_lp=lp, value=value)
                    ep = episode([d], winner=0 if index == 1 else 1, model=model, number=i)
                    ep.behavior_version = version
                    ep.potential_scale = 0
                    es.append(ep)
                rollout = batch(es)
                rollout.behavior_version = version
                p.update_batch(model, optimizer, rollout, version, cfg, rng)
            self.assertGreater(preference(), before + .15, msg=f"seed {seed}")


if __name__ == "__main__":
    unittest.main()
