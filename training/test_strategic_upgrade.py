"""Transferred (not just random-origin) PPO -> v3 strategic upgrade remains score-exact."""

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import foundation as f
import strategic_protocol as strategic
import team_ppo as p


class StrategicUpgradeTests(unittest.TestCase):
    def test_transferred_v2_ppo_upgrade_preserves_scores_and_lineage(self):
        with tempfile.TemporaryDirectory(dir=f.ROOT) as directory:
            root = Path(directory)
            torch.manual_seed(13)
            source = p.ActorCritic().eval()
            parent_metadata = {
                **f.SIZES, 'algorithm': p.ALGORITHM, 'status': 'completed',
                'learnReturns': True, 'games': 1008, 'ppoGames': 8, 'updates': 31,
                'sourceGames': 1000, 'sourceUpdates': 3988,
                'sourceSha256': 'a' * 64, 'sourceCheckpoint': '/example/frozen-v2.pt',
                'lineage': {'sourceGames': 1000, 'sourceSha256': 'a' * 64,
                            'sourceAlgorithm': f.METHODS['algorithm'], 'actorTransfer': 'exact'},
                'initialization': 'transferred self-trained v2 actor weights',
                'methods': {'algorithm': p.ALGORITHM},
            }
            source_file = root / 'parent-v2.pt'
            torch.save({'metadata': parent_metadata, 'model': source.state_dict()}, source_file)
            parent, loaded = p.load_checkpoint(source_file)
            self.assertEqual(loaded['metadata']['ppoGames'], 8)
            args = p.arguments(['train', '--upgrade-to-strategic', str(source_file), '--games', '0',
                                '--checkpoint-dir', str(root / 'v3'),
                                '--tensorboard-dir', str(root / 'tb'), '--device', 'cpu'])
            with patch.dict(os.environ, {'DAGUAI_ALLOW_JSON_LOGGER': '1'}):
                created = p.train(args)
            self.assertEqual(created['games'], 1000)
            self.assertEqual(created['sourceGames'], 1000)
            self.assertEqual(created['ppoGames'], 0)
            self.assertEqual(created['ablationParent']['priorPpoGames'], 8)
            upgraded, data = p.load_checkpoint(root / 'v3' / 'latest.pt')
            self.assertEqual(data['metadata']['encodingVersion'], 3)
            observation = np.zeros(f.OBSERVATION_SIZE, np.float32)
            observation[396] = 1.0
            actions = np.zeros((4, f.ACTION_SIZE), np.float32)
            actions[1, 1] = 0.5
            actions[2, 8] = 1.0
            history = np.zeros((0, f.HISTORY_SIZE), np.float32)
            with torch.inference_mode():
                legacy, legacy_value = parent.policy_value(*p._tensor_states(parent, [
                    (observation, actions, history)]))
                v3, v3_value = upgraded.policy_value(*p._tensor_states(upgraded, [
                    (np.pad(observation, (0, strategic.SIZES['observationSize'] - f.OBSERVATION_SIZE)),
                     np.pad(actions, ((0, 0), (0, strategic.SIZES['actionSize'] - f.ACTION_SIZE))),
                     history)]))
            torch.testing.assert_close(legacy[0], v3[0], rtol=0, atol=1e-6)
            torch.testing.assert_close(legacy_value, v3_value, rtol=0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
