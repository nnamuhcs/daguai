"""Focused, fast tests for proof-of-learning reporting; no full simulator needed."""
import json
import tempfile
from pathlib import Path
import unittest

import torch

from learning_evidence import checkpoint_delta, metric_comparison, audit_existing


class LearningEvidenceTests(unittest.TestCase):
    def test_state_changes_not_just_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / 'initial.pt'
            new = root / 'latest.pt'
            torch.save({'model': {'w': torch.tensor([1., 2., 3.])},
                        'metadata': {'updates': 2, 'ppoGames': 8}}, old)
            torch.save({'model': {'w': torch.tensor([1., 2.5, 3.])},
                        'metadata': {'updates': 20, 'ppoGames': 16}}, new)
            result = checkpoint_delta(old, new)
            self.assertTrue(result['comparable'])
            self.assertEqual(result['changedElements'], 1)
            self.assertEqual(result['totalElements'], 3)
            self.assertEqual(result['optimizerStepsAfter'], 20)

            # Metadata-only changes cannot be reported as learned weights.
            torch.save({'model': {'w': torch.tensor([1., 2., 3.])},
                        'metadata': {'updates': 200, 'ppoGames': 5000}}, new)
            self.assertEqual(checkpoint_delta(old, new)['changedElements'], 0)

    def test_metric_deltas_are_descriptive(self):
        one = {'metrics': {'headFinisher': .4, 'jokersUsed': 3}}
        two = {'metrics': {'headFinisher': .2, 'jokersUsed': 5}}
        result = metric_comparison(one, two)
        self.assertAlmostEqual(result['headFinisher']['delta'], .2)
        self.assertAlmostEqual(result['jokersUsed']['delta'], -2)

    def test_read_only_summarizes_saved_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chk = root / 'checkpoints'
            chk.mkdir()
            torch.save({'model': {'w': torch.tensor([1.])},
                        'metadata': {'updates': 0, 'ppoGames': 0}}, chk/'initial.pt')
            torch.save({'model': {'w': torch.tensor([2.])},
                        'metadata': {'updates': 1, 'ppoGames': 8}}, chk/'latest.pt')
            (root/'plan.json').write_text(json.dumps({'checkpointDir': str(chk)}))
            summary = {
                'cycle': 1, 'toGames': 8, 'candidateScores': {'3': .55}, 'promoted': False,
                'pairedComparisons': {'3': {'delta': .05}},
                'confirmationComparisons': {'3': {'delta': -.012}},
            }
            (root/'progress.json').write_text(json.dumps({
                'status': 'running', 'phase': 'cycle-completed', 'cycles': [summary],
                'champion': {'checkpoint': 'heuristic'}, 'researchBest': None}))
            cycle = root / 'cycles' / '000001'
            cycle.mkdir(parents=True)
            for label, score in (('candidate', .488), ('incumbent', .5)):
                (cycle / f'confirmation-{label}-seats-3-fresh.json').write_text(json.dumps({
                    'score': score, 'games': 1000, 'protocol': {'modelSeats': 3},
                    'opponentSha256': 'heuristic',
                }))
            (chk/'metrics.jsonl').write_text(json.dumps({'games': 8, 'optimizerSteps': 2})+'\n{"partial"')
            result = audit_existing(root)
            self.assertEqual(result['trainingWeights']['changedElements'], 1)
            self.assertEqual(result['trainerLastBatch']['optimizerSteps'], 2)
            self.assertEqual(result['recentCycles'][0]['confirmation']['score'], .488)
            self.assertEqual(result['recentCycles'][0]['confirmation']['games'], 1000)


if __name__ == '__main__':
    unittest.main()
