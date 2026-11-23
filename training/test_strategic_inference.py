"""Integration assertions for the opt-in strategic browser/model boundary.

A synthetic checkpoint is useful for protocol testing only; it is not evidence
of card strength, training, or release eligibility.
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch

import foundation as f
import strategic_protocol as strategic
import team_ppo as p
from serve_strategic import handle_request


class StrategicInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name) / "synthetic-v3.pt"
        metadata = {
            "games": 1, "ppoGames": 1, "updates": 0,
            "sourceGames": 0, "sourceUpdates": 0,
            "initialization": "random actor from scratch",
            "lineage": {
                "sourceGames": 0, "sourceSha256": None,
                "sourceAlgorithm": None, "actorTransfer": "random",
            },
            "status": "synthetic-test-only",
        }
        torch.manual_seed(4002)
        cls.model = p.StrategicActorCritic().eval()
        f.atomic_torch_save(p.checkpoint_data(cls.model, metadata), cls.path)
        cls.model, _ = p.load_checkpoint(cls.path, "cpu")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fixture(self, phase="mid"):
        env = f.NodeEnvironment(script=f.ROOT / "strategic-env.js", sizes=strategic.SIZES)
        try:
            return env.state("sample", seed=20_516_004, phase=phase)
        finally:
            env.close()

    def payload(self, state, limit=32):
        return {
            "op": "score",
            "observation": state["observation"],
            "actions": [a["features"] for a in state["actions"][:limit]],
            "history": f.relative_history(state["events"], state["player"]).tolist(),
        }

    def test_v3_rpc_matches_direct_policy_and_has_strict_input_shape(self):
        state = self.fixture("mid")
        request = self.payload(state)
        result = handle_request(self.model, request)
        states = [(np.asarray(request["observation"], np.float32),
                   np.asarray(request["actions"], np.float32),
                   np.asarray(request["history"], np.float32))]
        with torch.inference_mode():
            expected, _ = self.model.policy_value(*p._tensor_states(self.model, states))
        np.testing.assert_allclose(result["scores"], expected[0].numpy(), atol=1e-7, rtol=1e-6)
        self.assertEqual(len(result["scores"]), len(request["actions"]))
        self.assertTrue(np.isfinite(result["scores"]).all())
        metadata = handle_request(self.model, {"op": "info"})
        self.assertTrue(metadata["experimental"])
        self.assertEqual(metadata["encodingVersion"], 3)
        self.assertEqual(metadata["strategicFeatureSha256"], strategic.fingerprint())
        for mutation in (
                lambda data: data["observation"].pop(),
                lambda data: data["actions"][0].pop(),
                lambda data: data["history"].append([0] * 70),
                lambda data: data["observation"].__setitem__(0, float("nan")),
                lambda data: data["actions"][0].__setitem__(0, -4),
                lambda data: data["observation"].__setitem__(922, -4)):
            malformed = json.loads(json.dumps(request))
            mutation(malformed)
            with self.assertRaises(ValueError):
                handle_request(self.model, malformed)

    def test_http_scores_reload_local_origin_and_static_feature_modules(self):
        state = self.fixture("mid")
        payload = self.payload(state)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        environment = {
            **os.environ, "PORT": str(port), "AI_WORKER": "strategic",
            "AI_MODEL": str(self.path), "AI_PYTHON": sys.executable,
            "AI_DEVICE": "cpu", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        }
        server = subprocess.Popen(
            ["node", "server.js"], cwd=f.ROOT.parent, env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True,
        )

        def call(path, data=None, origin=None):
            headers = {"Content-Type": "application/json"}
            if origin:
                headers["Origin"] = origin
            request = Request(
                f"http://127.0.0.1:{port}{path}",
                data=None if data is None else json.dumps(data).encode(), headers=headers,
                method="GET" if data is None else "POST",
            )
            try:
                with urlopen(request, timeout=30) as response:
                    return response.status, response.read().decode("utf-8")
            except HTTPError as error:
                return error.code, error.read().decode("utf-8")

        try:
            until = time.monotonic() + 25
            while True:
                try:
                    status, body = call("/api/ai/status")
                    break
                except (URLError, ConnectionError):
                    if time.monotonic() >= until or server.poll() is not None:
                        self.fail("test game server failed to become ready")
                    time.sleep(.1)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["encodingVersion"], 3)
            score_status, scores = call("/api/ai/score", payload)
            self.assertEqual(score_status, 200, scores)
            np.testing.assert_allclose(json.loads(scores)["scores"],
                                       handle_request(self.model, payload)["scores"], rtol=1e-5, atol=1e-7)
            self.assertEqual(call("/api/ai/score", {"observation": [], "actions": [[]]})[0], 400)
            self.assertEqual(call("/api/ai/status", origin="http://foreign.invalid")[0], 403)
            self.assertEqual(call("/strategic-policy.js")[0], 200)
            self.assertEqual(call("/training/manifests/strategic-v3.json")[0], 200)
            bad = json.loads(json.dumps(payload))
            bad["observation"][924] = -4
            self.assertEqual(call("/api/ai/score", bad)[0], 400)
            self.assertEqual(call("/api/ai/reload", {})[0], 200)
            self.assertEqual(call("/api/ai/score", payload)[0], 200)
        finally:
            server.terminate()
            try:
                server.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
