import copy
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import extend_ppo
import foundation as f
import tournament


class ExtensionTests(unittest.TestCase):
    def recovery_fixture(self, directory):
        root = Path(directory)
        checkpoints = root / "checkpoints"
        checkpoints.mkdir()
        args = Namespace(
            resume=checkpoints / "latest.pt", checkpoint_dir=checkpoints,
            tensorboard_dir=root / "tensorboard", output_dir=root, baseline_reports=None,
            games=16, every=8, evaluation_games=2, evaluation_seed=20_701_001,
            device="cpu", continue_run=True)
        config = {"temperature": .2}
        metadata = {"status": "failed", "ppoGames": 20, "nextEpisode": 20,
                    "initialSha256": "0" * 64, "config": config}
        baseline = {"metadata": {**metadata, "ppoGames": 8}}
        data = {"metadata": metadata, "model": {"weight": torch.tensor([3.0])},
                "optimizer": {"step": 7}, "rng": {"local": (1, 2, 3)}}
        (root / "baseline.pt").write_bytes(b"frozen baseline")
        stage_dir = root / "stage-000000016"
        stage_dir.mkdir()
        (stage_dir / "frozen.pt").write_bytes(b"frozen milestone")
        stage = {"totalPpoGames": 16, "checkpointSha256": f.sha256_file(stage_dir / "frozen.pt")}
        (stage_dir / "summary.json").write_text(json.dumps(stage))
        plan = {
            "kind": "ppo-extension", "startPpoGames": 8, "ppoGames": 16,
            "targetTotalPpoGames": 24, "milestoneGames": 8, "evaluationGames": 2,
            "evaluationSeed": args.evaluation_seed, "rules": f.rules_fingerprint(),
            "checkpointDirectory": str(checkpoints), "replayDirectory": str(root / "replays"),
            "baselineSha256": f.sha256_file(root / "baseline.pt"),
            "promotion": False, "finalHoldoutOpened": False}
        state = {"status": "failed", "error": "original audit failure", "stages": [stage],
                 "completedEvaluationReports": 6}
        (root / "plan.json").write_text(json.dumps(plan))
        (root / "series-progress.json").write_text(json.dumps(state))
        (root / "replays").mkdir()
        for episode in range(8, 20):
            name = f"game-{episode:09d}"
            (root / "replays" / f"{name}.json").write_text(
                json.dumps({"episodeId": episode, "terminal": True}))
            (root / "replays" / f"{name}.jsonl.gz").write_bytes(b"replay fixture")
        return args, data, baseline, plan, state

    def test_recovery_archives_optimizer_rng_and_preserves_original_plan_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            args, data, baseline, plan, state = self.recovery_fixture(directory)
            with mock.patch.object(extend_ppo.p, "load_checkpoint", return_value=(None, baseline)), \
                    mock.patch.object(extend_ppo, "reusable_baseline"):
                restored_plan, restored_state = extend_ppo.restore_extension(args, data)
            self.assertEqual(restored_plan, plan)
            self.assertEqual(restored_state["stages"], state["stages"])
            self.assertEqual(restored_state["recoveries"][0]["remainingPpoGames"], 4)
            self.assertEqual(restored_state["recoveries"][0]["committedReplays"], 12)
            self.assertEqual(restored_state["evaluationModelSeats"], [3])
            self.assertEqual(restored_state["completedEvaluationReports"], 3)
            self.assertEqual(restored_state["totalEvaluationReports"], 4)
            self.assertEqual(restored_state["comparisonScopeAmendments"][0]["previousEvaluationModelSeats"], [1, 3])
            archived = torch.load(restored_state["recoveries"][0]["sourceCheckpoint"], weights_only=True)
            self.assertEqual(archived["optimizer"], data["optimizer"])
            self.assertEqual(archived["rng"], data["rng"])
            torch.testing.assert_close(archived["model"]["weight"], data["model"]["weight"])
            self.assertEqual(json.loads((args.output_dir / "plan.json").read_text()), plan)
            self.assertEqual(json.loads((args.output_dir / "series-progress.json").read_text()), state)

    def test_recovery_refuses_changed_protocol_or_gaps_in_committed_replays(self):
        for corruption in ("protocol", "replay"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                args, data, baseline, _, _ = self.recovery_fixture(directory)
                if corruption == "protocol":
                    args.evaluation_seed += 1
                else:
                    (args.output_dir / "replays" / "game-000000008.json").unlink()
                with mock.patch.object(extend_ppo.p, "load_checkpoint", return_value=(None, baseline)), \
                        mock.patch.object(extend_ppo, "reusable_baseline"):
                    with self.assertRaisesRegex(ValueError, "plan|replay IDs"):
                        extend_ppo.restore_extension(args, data)
                self.assertFalse(list(args.output_dir.glob("recovery-source-*.pt")))

    def test_resumed_orchestrator_trains_only_remaining_games_without_repeating_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            args, data, _, plan, state = self.recovery_fixture(directory)
            state.update(baseline={}, latestComparisons={})
            report = {"score": .5, "wins": 1, "losses": 1, "draws": 0, "confidence95": [0, 1]}
            with mock.patch.object(extend_ppo.p, "load_checkpoint", return_value=(None, data)), \
                    mock.patch.object(extend_ppo, "restore_extension", return_value=(plan, state)), \
                    mock.patch.object(extend_ppo, "reusable_baseline", return_value=report), \
                    mock.patch.object(extend_ppo.registry, "validate_split"), \
                    mock.patch.object(extend_ppo, "evaluate") as evaluate, \
                    mock.patch.object(extend_ppo.p, "train", side_effect=RuntimeError("dry-run boundary")) as train:
                with self.assertRaisesRegex(RuntimeError, "dry-run boundary"):
                    extend_ppo.run(args)
            self.assertEqual(train.call_args.args[0].games, 4)
            self.assertEqual(train.call_args.args[0].resume, args.resume)
            self.assertEqual(train.call_args.args[0].replay_dir, args.output_dir / "replays")
            evaluate.assert_not_called()

    def test_full_team_only_runner_never_requests_mixed_team_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            args, data, _, plan, state = self.recovery_fixture(directory)
            state.pop("error")
            state.update(baseline={}, latestComparisons={}, evaluationModelSeats=[3],
                         completedEvaluationReports=3, totalEvaluationReports=4)
            report = {"score": .5, "wins": 1, "losses": 1, "draws": 0, "confidence95": [0, 1]}
            with mock.patch.object(extend_ppo.p, "load_checkpoint", return_value=(None, data)), \
                    mock.patch.object(extend_ppo, "restore_extension", return_value=(plan, state)), \
                    mock.patch.object(extend_ppo.registry, "validate_split"), \
                    mock.patch.object(extend_ppo.registry, "record_evaluation"), \
                    mock.patch.object(extend_ppo.registry, "finish_evaluation"), \
                    mock.patch.object(extend_ppo, "reusable_baseline", return_value=report) as reused, \
                    mock.patch.object(extend_ppo.p, "train",
                                      return_value={"status": "completed", "ppoGames": 24}), \
                    mock.patch.object(extend_ppo.f, "freeze_checkpoint",
                                      side_effect=lambda source, target: target.write_bytes(b"final model")), \
                    mock.patch.object(extend_ppo.f, "compare_reports",
                                      return_value={"delta": 0, "confidence95": [-.1, .1],
                                                    "assessment": "not established"}), \
                    mock.patch.object(extend_ppo, "evaluate", return_value=copy.deepcopy(report)) as evaluate:
                result = extend_ppo.run(args)
            self.assertTrue(reused.call_args_list)
            self.assertTrue(all(call.args[2] == 3 for call in reused.call_args_list))
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(evaluate.call_args.kwargs["model_seats"], 3)
            self.assertEqual(set(result["stages"][-1]["comparisons"]), {"3"})
            self.assertEqual(result["completedEvaluationReports"], 4)

    def test_negative_mean_alone_does_not_trigger_regression_stop(self):
        comparison = {"confidence95": [-.1, .05]}
        result = {"3": {"versusBaseline": comparison, "versusPrevious": comparison}}
        self.assertFalse(extend_ppo.regression_detected(result))
        changed = copy.deepcopy(result)
        changed["3"]["versusPrevious"]["confidence95"] = [-.1, -.01]
        self.assertTrue(extend_ppo.regression_detected(changed))

    def test_evaluation_reports_actual_completed_pairs_and_cache_checks_protocol(self):
        updates = []
        report = tournament.evaluate(games=2, seed=20_701_001, progress_callback=updates.append)
        self.assertEqual(updates, [{"completedGames": 2, "totalGames": 2, "completedPairs": 1}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(report))
            reused = extend_ppo.reusable_baseline(
                path, "heuristic", 3, 2, report["seed"], report["protocol"]["rules"])
            self.assertEqual(reused["score"], .5)
            with self.assertRaisesRegex(ValueError, "cached baseline"):
                extend_ppo.reusable_baseline(
                    path, "heuristic", 1, 2, report["seed"], report["protocol"]["rules"])


if __name__ == "__main__":
    unittest.main()
