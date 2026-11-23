import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import distill_start as d
import foundation as f
import team_ppo as p


class DistillTests(unittest.TestCase):
    def test_source_target_handles_ties_full_support_and_single_actions(self):
        target = d.epsilon_target(torch.tensor([1.0, 1.0, -3.0]))
        torch.testing.assert_close(target, torch.tensor([.9, .05, .05]))
        torch.testing.assert_close(d.epsilon_target(torch.tensor([0.])), torch.tensor([1.]))
        with self.assertRaises(ValueError):
            d.epsilon_target(torch.tensor([float("nan")]))

    def test_retention_measures_distribution_not_just_matching_greedy(self):
        target = d.epsilon_target(torch.tensor([1., 0.]))
        result = d.retention([target.log() * .2], [target], .2)
        self.assertTrue(result["passed"])
        json.dumps(result, allow_nan=False)
        almost_uniform = d.retention([torch.tensor([.001, 0.])], [target], .2)
        self.assertEqual(almost_uniform["topAgreement"], 1)
        self.assertFalse(almost_uniform["passed"])

    def test_distilled_checkpoint_rejects_unverified_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            checkpoint = Path(directory) / "prepared.pt"
            f.atomic_torch_save(f.checkpoint_data(f.ActionValue(), {"games": 1, "updates": 1}), source)
            model, metadata = p.warm_start(source)
            metadata["initialization"] = "source-behavior-distilled v2 actor"
            metadata["lineage"]["actorTransfer"] = "source-behavior-distilled"
            metadata["config"] = dict(p.CORE_DEFAULTS)
            metadata["behaviorDistillation"] = {
                "sourceSha256": metadata["sourceSha256"], "behaviorPreserved": True,
                "temperature": .2, "confirmation": {"passed": True, "meanKL": .01, "topAgreement": 1.0}}
            f.atomic_torch_save(p.checkpoint_data(model, metadata), checkpoint)
            p.load_checkpoint(checkpoint)
            failed = copy.deepcopy(metadata)
            failed["behaviorDistillation"]["confirmation"]["meanKL"] = .5
            f.atomic_torch_save(p.checkpoint_data(model, failed), checkpoint)
            with self.assertRaisesRegex(ValueError, "retention"):
                p.load_checkpoint(checkpoint)

    def test_distillation_reduces_kl_without_training_shared_history_or_value(self):
        model = p.ActorCritic()
        model.head = torch.nn.Linear(465, 1, bias=False)
        torch.nn.init.zeros_(model.head.weight)
        before_base = model.base[0].weight.detach().clone()
        features = torch.zeros(2, 465)
        features[0, 0], features[1, 0] = 1, -1
        target = d.epsilon_target(torch.tensor([1., 0.]))
        entries = [(features, target)]
        result = d.fit_head(model, entries, entries, epochs=40, learning_rate=.05,
                            temperature=.2, seed=7)
        self.assertLess(result["after"]["meanKL"], result["before"]["meanKL"])
        self.assertTrue(result["after"]["passed"])
        torch.testing.assert_close(model.base[0].weight, before_base, rtol=0, atol=0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
