import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import action_audit
import check_release
import foundation as f
import seed_registry
import tournament


class AuditTests(unittest.TestCase):
    def test_distribution_exposes_near_uniform_nonforced_actions_and_canonical_kind_mass(self):
        actions = [{"kind": "play"}, {"kind": "ask"}, {"kind": "pass"}]
        stats = action_audit.distribution_stats([0.001, 0.0, -0.001], actions, 0.2)
        self.assertGreater(stats["normalizedEntropy"], 0.999)
        self.assertAlmostEqual(sum(stats["kindProbability"].values()), 1)
        self.assertLess(stats["topProbability"], 0.34)
        self.assertLess(action_audit.distribution_stats([0.001], actions[:1], 0.2)["normalizedEntropy"], 1)

    def test_calibration_excludes_forced_turns_from_policy_confidence(self):
        rows = []
        for seed, count in ((1, 1), (2, 3)):
            actions = [{"kind": "play"} for _ in range(count)]
            rows.append({"seed": seed, "phase": "fresh", "legalActions": count,
                         "epsilonTopProbability": 0.85 + 0.15 / count,
                         "temperatures": {
                             f"{temperature:.2f}": action_audit.distribution_stats(
                                 np.zeros(count), actions, temperature)
                             for temperature in action_audit.TEMPERATURES}})
        report = {"positions": 2, "positionsData": rows, "checkpointSha256": "fixture",
                  "samplingPolicy": "fixture"}
        result = action_audit.calibration_summary(report)["groups"]["all"]
        self.assertEqual(result["forcedPositions"], 1)
        self.assertAlmostEqual(result["temperatures"]["0.20"]["topProbability"], 1 / 3)
        self.assertAlmostEqual(result["temperatures"]["0.20"]["normalizedEntropy"], 1)

    def test_final_seed_access_is_denied_without_matching_preselection(self):
        with self.assertRaisesRegex(ValueError, "preselected"):
            seed_registry.validate_split("final", 3_500_000_000, 2)
        with self.assertRaisesRegex(ValueError, "outside"):
            seed_registry.validate_split("validation", 2_000_000_000, 2)
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text(json.dumps({"status": "sealed", "candidateSha256": "a",
                                        "seed": 3_500_000_000, "games": 2}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                seed_registry.validate_split("final", 3_500_000_000, 2, plan, "b")

    def test_release_checker_blocks_missing_or_small_evidence_without_opening_holdout(self):
        result = check_release.check_evidence({})
        self.assertFalse(result["preselectionGatesPassed"])
        self.assertFalse(result["finalHoldoutOpened"])
        self.assertFalse(result["deploymentAllowed"])

    def test_online_gate_distinguishes_complete_support_from_per_request_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "online.json"
            online = {"passed": True, "allLegalActionsScored": True, "maxLegalActions": 1836,
                      "maxBatchActions": 512, "measuredDecisions": 100, "p95DecisionLatencyMs": 25}
            path.write_text(json.dumps(online))
            result = check_release.check_evidence({"onlineEvidence": str(path)})
            self.assertTrue(result["gates"]["onlineLegalSupport"]["passed"])
            online["maxBatchActions"] = 513
            path.write_text(json.dumps(online))
            self.assertFalse(check_release.check_evidence({"onlineEvidence": str(path)})[
                "gates"]["onlineLegalSupport"]["passed"])

    def test_incremental_legacy_actor_matches_full_prefix_including_asking_and_rotations(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor.pt"
            f.atomic_torch_save(f.checkpoint_data(f.ActionValue(), {"games": 0, "updates": 0}), checkpoint)
            cached = tournament.Policy(checkpoint)
            full = tournament.Policy(checkpoint, cache=False)
            env = f.NodeEnvironment(script=f.ROOT / "audit-env.js")
            try:
                state = env.state("reset", seed=20_005_001)
                events = list(state["events"])
                for _ in range(25):
                    a = cached.scores(state, events)[1]
                    b = full.scores(state, events)[1]
                    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
                    ask = next((i for i, action in enumerate(state["actions"]) if action["kind"] == "ask"), None)
                    state = env.state("step", index=ask) if ask is not None else env.state("heuristic")
                    events.extend(state["events"])
                    if state["terminal"]:
                        break
            finally:
                env.close()
            self.assertLess(cached.gru_steps, full.gru_steps)

    def test_real_continuation_outcomes_and_score_tampering_are_checked(self):
        report = tournament.evaluate(games=2, seed=20_000_013, rounds="continued")
        self.assertEqual(report["score"], 0.5)
        self.assertEqual(check_release.validate_report(report), [0.5, 0.5])
        altered = copy.deepcopy(report)
        altered["score"] = 1.0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            check_release.validate_report(altered)
        altered = copy.deepcopy(report)
        altered["protocol"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "protocol and measured"):
            check_release.validate_report(altered)


if __name__ == "__main__":
    unittest.main()
