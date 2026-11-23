import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

import foundation as f
import strategic_protocol as strategic
import team_ppo as p
from test_team_ppo import event


class StrategicModelTests(unittest.TestCase):
    def test_v3_transfer_preserves_parent_scores_value_and_categorical_distribution(self):
        torch.manual_seed(101)
        parent = p.ActorCritic().eval()
        evolved = p.StrategicActorCritic().eval()
        evolved.load_state_dict({**evolved.state_dict(), **parent.state_dict()})
        env = f.NodeEnvironment(script=f.ROOT / "strategic-env.js", sizes=strategic.SIZES)
        try:
            states = []
            for seed in range(20_804_390, 20_804_396):
                state = env.state("reset", seed=seed)
                states.append((np.asarray(state["observation"], np.float32),
                               np.asarray([a["features"] for a in state["actions"]], np.float32),
                               [event(2)] * (seed % 8)))
        finally:
            env.close()
        legacy = [(obs[:f.OBSERVATION_SIZE], actions[:, :f.ACTION_SIZE], history)
                  for obs, actions, history in states]
        with torch.no_grad():
            old, old_value = parent.policy_value(*p._tensor_states(parent, legacy))
            new, new_value = evolved.policy_value(*p._tensor_states(evolved, states))
        for before, after in zip(old, new):
            torch.testing.assert_close(before, after, rtol=0, atol=1e-7)
            self.assertEqual(before.argmax().item(), after.argmax().item())
            torch.testing.assert_close(torch.softmax(before / .2, 0), torch.softmax(after / .2, 0))
        torch.testing.assert_close(old_value, new_value, rtol=0, atol=1e-7)

    def test_v3_scores_use_candidate_features_without_changing_action_independent_value(self):
        torch.manual_seed(102)
        model = p.StrategicActorCritic().eval()
        with torch.no_grad():
            model.actor_residual[-1].weight.fill_(.5)
        obs = np.zeros(strategic.SIZES["observationSize"], np.float32)
        actions = np.zeros((2, strategic.SIZES["actionSize"]), np.float32)
        actions[1, f.ACTION_SIZE:] = 1
        with torch.no_grad():
            scores, values = model.policy_value(*p._tensor_states(model, [(obs, actions, [])]))
            reversed_scores, reversed_values = model.policy_value(
                *p._tensor_states(model, [(obs, actions[::-1].copy(), [])]))
        self.assertGreater(abs(float(scores[0][0] - scores[0][1])), 1e-5)
        torch.testing.assert_close(reversed_scores[0].flip(0), scores[0])
        torch.testing.assert_close(values, reversed_values, rtol=0, atol=0)

    def test_strategic_checkpoint_recognized_by_tournament_but_not_legacy_inference(self):
        metadata = {
            "games": 0, "ppoGames": 0, "updates": 0, "sourceGames": 0, "sourceUpdates": 0,
            "initialization": "random actor from scratch",
            "lineage": {"sourceGames": 0, "actorTransfer": "random"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            data = p.checkpoint_data(p.StrategicActorCritic(), metadata)
            f.atomic_torch_save(data, path)
            self.assertEqual(f.load_policy(path, allow_strategic=True)[2], 3)
            with self.assertRaisesRegex(ValueError, "unsupported checkpoint encoding"):
                f.load_policy(path)
            data["metadata"]["strategicFeatureSha256"] = "not this manifest"
            f.atomic_torch_save(data, path)
            with self.assertRaisesRegex(ValueError, "incompatible strategic"):
                p.load_checkpoint(path)
