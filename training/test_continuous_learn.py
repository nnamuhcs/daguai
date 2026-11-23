"""Gates and persistent-state tests for unattended continuous training."""

from pathlib import Path
import tempfile
import unittest

import foundation as f
from continuous_learn import (Controller, decision_from_comparison, parse_args,
                              stage_seed, exclusive_lock, validate_team_report)
from tournament import evaluate


class ContinuousGatesTests(unittest.TestCase):
    def test_promotion_requires_a_positive_paired_lower_bound(self):
        plus = {'confidence95': [0.010, 0.050]}
        neutral = {'confidence95': [-0.019, 0.040]}
        self.assertTrue(decision_from_comparison(plus)[0])
        self.assertFalse(decision_from_comparison(neutral)[0])
        self.assertFalse(decision_from_comparison(plus, minimum_full_improvement=.02)[0])
        self.assertFalse(decision_from_comparison({'confidence95': [float('nan'), .3]})[0])

    def test_team_report_requires_actual_complete_side_swaps(self):
        seed = 20_891_070
        report = evaluate(games=2, seed=seed)
        protocol = ('heuristic', 2, seed, 'fresh', f.rules_fingerprint())
        validate_team_report(report, *protocol)
        report['pairedSeeds'][0]['matches'][0]['learnerSeats'] = [0]
        with self.assertRaisesRegex(ValueError, 'swap complete teams'):
            validate_team_report(report, *protocol)

    def test_different_cycles_use_disjoint_valid_seed_ranges(self):
        start = 1_100_300_000
        ranges = [(stage_seed(start, 200, i), stage_seed(start, 200, i) + 99)
                  for i in range(1, 100)]
        self.assertTrue(all(a[1] < b[0] for a, b in zip(ranges, ranges[1:])))
        self.assertTrue(all(1_100_000_000 <= low <= high <= 1_100_999_999 for low, high in ranges))

    def test_invalid_inputs_rejected(self):
        for item in (['--stage-games', '9'], ['--evaluation-games', '3'],
                     ['--rollout-games', '65'], ['--rollout-games', '0'], ['--strategic'],
                     ['--ablate-from', 'model.pt']):
            with self.subTest(item=item), self.assertRaises(SystemExit):
                parse_args(item)

    def test_immutable_plan_and_resume_are_not_silently_reset(self):
        with tempfile.TemporaryDirectory(dir=f.ROOT) as tmp:
            root = Path(tmp)
            path = root / 'run'
            checkpoints = root / 'train' / 'checkpoints'
            args = parse_args(['--root', str(path), '--checkpoint-dir', str(checkpoints),
                               '--from-scratch', '--cycles', '1'])
            controller = Controller(args)
            controller._load_or_initialize()
            self.assertEqual(controller.progress['champion']['checkpoint'], 'heuristic')
            self.assertTrue((path / 'plan.json').exists())
            resumed = parse_args(['--root', str(path), '--checkpoint-dir', str(checkpoints),
                                  '--cycles', '1'])
            controller2 = Controller(resumed)
            controller2._load_or_initialize()
            self.assertEqual(controller2.plan, controller.plan)
            changed = parse_args(['--root', str(path), '--checkpoint-dir', str(checkpoints),
                                  '--stage-games', '256'])
            with self.assertRaisesRegex(ValueError, 'existing plan changed'):
                Controller(changed)._load_or_initialize()
            with self.assertRaisesRegex(ValueError, 'initialization arguments'):
                Controller(args)._load_or_initialize()

    def test_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory(dir=f.ROOT) as tmp:
            target = Path(tmp) / 'lock'
            with exclusive_lock(target):
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    with exclusive_lock(target):
                        pass


if __name__ == '__main__':
    unittest.main()
