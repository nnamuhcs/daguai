import unittest
import numpy as np
from decision_quality import candidate_indices, interpret_audit

class DecisionQualityTests(unittest.TestCase):
    def test_include_argmax_and_pass(self):
        state = {'actions': [{'kind': 'play'}, {'kind': 'pass'}, {'kind': 'play'}, {'kind': 'ask'}]}
        self.assertEqual(candidate_indices(state, np.array([0., -5., 2., 1.]), 3), [2, 3, 1])

    def test_older_choice_does_not_displace_current_choice_or_pass(self):
        state = {'actions': [{'kind': kind} for kind in ('play', 'play', 'play', 'pass')]}
        scores = [4, 3, 2, 1]
        self.assertEqual(candidate_indices(state, scores, 3, previous=2, budget=3), [0, 3, 2])

    def test_explicit_uncertainty_before_calling_a_move_better(self):
        result = {'status': 'ok', 'estimates': [
            {'index': 3, 'score': .3, 'paired95Approx': [0, 0]},
            {'index': 8, 'score': .7, 'paired95Approx': [.1, .7]}]}
        view = interpret_audit(result)
        self.assertTrue(view['confidentDisagreement'])
        self.assertAlmostEqual(view['estimatedRegret'], .4)
        result['estimates'][1]['paired95Approx'] = [-.1, .9]
        self.assertFalse(interpret_audit(result)['confidentDisagreement'])

    def test_insufficient_worlds_have_no_estimated_regret(self):
        self.assertNotIn('estimatedRegret', interpret_audit({'status': 'insufficient-belief-worlds'}))

if __name__ == '__main__':
    unittest.main()
