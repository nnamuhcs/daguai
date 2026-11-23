import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

import foundation as f
from game_replay import GameReplay, compact_game, list_replays, read_replay
import team_ppo as p


class ReplayTests(unittest.TestCase):
    def test_full_hand_recording_does_not_change_actor_inputs_or_sampled_actions(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        model = p.ActorCritic().eval()
        config = {**p.CORE_DEFAULTS, "workers": 1, "rollout_games": 2, "seed": 20_700_000}

        def collect(directory=None):
            env = f.NodeEnvironment(script=f.ROOT / "audit-env.js" if directory else None)
            worker = {"env": env, "state": None, "finished": False}
            recorders = []
            try:
                torch.manual_seed(7)
                batch, _, _ = p.collect_batch(
                    model, [worker], 2, random.Random(17), config, 0,
                    replay_dir=directory, replay_recorders=recorders)
                return batch, recorders
            finally:
                env.close()

        with tempfile.TemporaryDirectory() as directory:
            plain, _ = collect()
            recorded, recorders = collect(Path(directory))
            self.assertEqual(list_replays(directory), [])
            for first, second in zip(plain.episodes, recorded.episodes):
                self.assertEqual(first.winner, second.winner)
                np.testing.assert_array_equal(first.events, second.events)
                self.assertEqual(len(first.decisions), len(second.decisions))
                for a, b in zip(first.decisions, second.decisions):
                    np.testing.assert_array_equal(a.observation, b.observation)
                    np.testing.assert_array_equal(a.actions, b.actions)
                    self.assertEqual(a.index, b.index)
                    self.assertEqual(a.old_logprob, b.old_logprob)
                    self.assertEqual(a.old_value, b.old_value)
            for recorder in recorders:
                recorder.commit()
            self.assertEqual(len(list_replays(directory)), 2)
            for metadata in list_replays(directory):
                name = f"game-{metadata['episodeId']:09d}"
                replay = read_replay(directory, name)
                self.assertEqual(replay["steps"][-1]["state"]["result"]["winner"], metadata["winner"])
                for step in replay["steps"]:
                    ids = sum(step["state"]["hands"], []) + [card[0] for card in step["state"]["played"]]
                    self.assertEqual(len(ids), 162)
                    self.assertEqual(len(set(ids)), 162)
                self.assertEqual((Path(directory) / f"{name}.jsonl.gz").stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, "invalid"):
                read_replay(directory, "../../secret")

    def test_incomplete_match_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            env = f.NodeEnvironment(script=f.ROOT / "audit-env.js")
            try:
                env.state("reset", seed=20_700_002)
                recorder = GameReplay(directory, 3, 20_700_002, env.request("snapshot")["snapshot"], {})
                with self.assertRaisesRegex(ValueError, "complete"):
                    recorder.commit()
                recorder.abort()
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
