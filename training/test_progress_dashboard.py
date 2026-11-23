import fcntl
import json
from pathlib import Path
import tempfile
import unittest

import progress_dashboard as dashboard


class DashboardTests(unittest.TestCase):
    def fixture(self, root):
        (root / "checkpoints").mkdir()
        (root / "plan.json").write_text(json.dumps({"ppoGames": 256}))
        (root / "checkpoints" / "progress.json").write_text(json.dumps(
            {"ppoGames": 64, "updates": 400, "status": "running"}))
        return root

    def test_live_batch_is_newer_than_saved_checkpoint_without_claiming_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(Path(directory))
            row = {"ppoGames": 72, "updates": 450, "rolloutWallSeconds": 10, "updateWallSeconds": 20}
            (root / "checkpoints" / "metrics.jsonl").write_text(json.dumps(row) + "\n{\"unfinished\":")
            result = dashboard.progress(root, include_gpu=False)
            self.assertEqual(result["games"], 72)
            self.assertEqual(result["phase"], "Training")
            self.assertEqual(result["comparisons"][0]["assessment"], "pending paired comparison")
            self.assertIn("Not known", result["verdict"])
            self.assertEqual(len(result["errors"]), 1)

    def test_failed_pilot_is_not_presented_as_successful_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(Path(directory))
            (root / "result.json").write_text(json.dumps({"status": "failed", "error": "training failed"}))
            result = dashboard.progress(root, include_gpu=False)
            self.assertEqual(result["phase"], "Pilot failed")
            self.assertIn("training failed", result["errors"])

    def test_completed_comparison_reports_inconclusive_result_honestly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(Path(directory))
            comparisons = {str(seats): {"delta": .01, "confidence95": [-.03, .04],
                                       "assessment": "not established"} for seats in (1, 3)}
            (root / "result.json").write_text(json.dumps(
                {"status": "completed", "comparisons": comparisons, "fullTeamImprovementEstablished": False}))
            result = dashboard.progress(root, include_gpu=False)
            self.assertEqual(result["phase"], "Pilot complete")
            self.assertIn("not statistically established", result["verdict"])

    def test_extension_uses_additional_game_counts_and_live_evaluation_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(Path(directory))
            (root / "plan.json").write_text(json.dumps({"ppoGames": 1024, "startPpoGames": 256}))
            row = {"ppoGames": 272, "updates": 1800, "rolloutWallSeconds": 10, "updateWallSeconds": 20}
            (root / "checkpoints" / "metrics.jsonl").write_text(json.dumps(row) + "\n")
            state = {"phase": "Evaluating checkpoint", "completedEvaluationReports": 4,
                     "totalEvaluationReports": 12, "baseline": {}, "latestComparisons": {}, "stages": [],
                     "liveEvaluation": {"label": "checkpoint", "completedGames": 50, "totalGames": 200}}
            (root / "series-progress.json").write_text(json.dumps(state))
            result = dashboard.progress(root, include_gpu=False)
            self.assertEqual(result["games"], 16)
            self.assertEqual(result["targetGames"], 1024)
            self.assertEqual(result["liveEvaluation"]["completedGames"], 50)
            self.assertEqual(result["totalEvaluationReports"], 12)

    def test_team_only_scope_does_not_display_archived_mixed_team_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(Path(directory))
            comparison = {
                "score": .5, "evaluatedAtTotalGames": 768,
                "versusBaseline": {"delta": .1, "confidence95": [.02, .18], "assessment": "improved"},
                "versusWebHeuristic": {"assessment": "not established"}}
            state = {
                "phase": "Evaluating full team", "evaluationModelSeats": [3],
                "completedEvaluationReports": 5, "totalEvaluationReports": 6,
                "baseline": {"1": {"score": .47}, "3": {"score": .37}},
                "latestComparisons": {"1": comparison, "3": comparison},
                "stages": [], "liveEvaluation": None}
            (root / "series-progress.json").write_text(json.dumps(state))
            result = dashboard.progress(root, include_gpu=False)
            self.assertEqual([row["seats"] for row in result["comparisons"]], [3])
            self.assertEqual(result["completedEvaluationReports"], 5)
            self.assertEqual(result["totalEvaluationReports"], 6)

    def continuous_fixture(self, root):
        self.fixture(root)
        (root / "plan.json").write_text(json.dumps({
            "modelSeats": 3, "checkpointDir": str(root / "checkpoints"),
            "evaluationGames": 400, "confirmationGames": 1000,
        }))
        return {
            "status": "running", "phase": "training", "totalNewPpoGames": 0,
            "champion": {"checkpoint": "heuristic"}, "cycles": [],
            "pending": {"cycle": 1, "startGames": 800, "targetGames": 928, "phase": "training"},
        }

    def test_continuous_training_reports_saved_progress_without_implying_a_match_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.continuous_fixture(root)
            (root / "progress.json").write_text(json.dumps(state))
            (root / "checkpoints" / "progress.json").write_text(json.dumps({"games": 824}))
            with (root / ".controller.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                live = dashboard.continuous_progress(root)
            self.assertTrue(live["running"])
            self.assertEqual(live["training"], {"completedGames": 24, "totalGames": 128})
            self.assertEqual(live["newTrainingGames"], 24)
            self.assertEqual(live["history"], [])
            self.assertIsNone(live["championCycle"])
            self.assertFalse(dashboard.continuous_progress(root)["running"])

    def test_continuous_history_includes_both_completed_scores_during_next_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.continuous_fixture(root)
            state.update(totalNewPpoGames=128, phase="evaluating",
                         pending={"cycle": 2, "startGames": 928, "targetGames": 1056, "phase": "evaluating"},
                         liveEvaluation={"label": "confirmation-candidate", "rounds": "fresh",
                                         "completedGames": 60, "totalGames": 1000})
            state["cycles"] = [{
                "cycle": 1, "candidateScores": {"3": .55}, "incumbentScores": {"3": .50},
                "pairedComparisons": {"3": {"delta": .05, "confidence95": [.012, .08],
                                           "assessment": "improved"}},
                "confirmationComparisons": {"3": {"delta": -.012, "confidence95": [-.035, .0125]}},
                "promoted": False,
            }]
            cycle = root / "cycles" / "000001"
            cycle.mkdir(parents=True)
            validation = {"games": 400, "wins": 142, "losses": 102, "draws": 156}
            confirmation = {"games": 1000, "wins": 297, "losses": 321, "draws": 382}
            (cycle / "candidate-seats-3-fresh.json").write_text(json.dumps(validation))
            (cycle / "confirmation-candidate-seats-3-fresh.json").write_text(
                json.dumps({"score": .488, **confirmation}))
            (cycle / "confirmation-incumbent-seats-3-fresh.json").write_text(json.dumps({"score": .50}))
            (root / "progress.json").write_text(json.dumps(state))
            with (root / ".controller.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                live = dashboard.continuous_progress(root)
            self.assertEqual(live["evaluation"]["label"], "confirmation-candidate")
            self.assertEqual(live["cycle"], 2)
            self.assertEqual(live["newTrainingGames"], 256)
            row = live["history"][0]
            self.assertEqual((row["score"], row["results"]), (.55, validation))
            self.assertEqual(row["confirmation"], {
                "score": .488, "incumbentScore": .50, "results": confirmation,
                "delta": -.012, "confidence95": [-.035, .0125],
            })
            self.assertFalse(row["promoted"])
            self.assertIsNone(live["championCycle"])

if __name__ == "__main__":
    unittest.main()
