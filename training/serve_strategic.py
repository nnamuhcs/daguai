"""Explicit opt-in, localhost-only v3 inference worker.

This worker never chooses actions and never sees simulator hands.  The caller
supplies ONLY the current actor observation, ordered legal features, and
seat-relative public history.  The engine validates the selected action.
Unlike the legacy inference path, dimensions and the strategic producer SHA
are checked against the saved checkpoint on load.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

import foundation as f
import strategic_protocol as strategic
import team_ppo as p

MAX_REQUEST = 8 * 1024 * 1024


def handle_request(model, request):
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    op = request.get("op")
    if op == "info":
        metadata = model.metadata
        return {
            "available": True, "experimental": True, "learnReturns": True,
            "games": metadata["games"], "updates": metadata["updates"],
            "device": str(next(model.parameters()).device),
            "checkpoint": model.checkpoint,
            **strategic.SIZES,
            "trainingStatus": metadata.get("status", "unknown"),
            "strategicFeatureSha256": metadata["strategicFeatureSha256"],
        }
    if op == "score":
        observation = f.finite_vector(request.get("observation"),
                                      strategic.SIZES["observationSize"], "observation")
        actions = f.matrix(request.get("actions"), strategic.SIZES["actionSize"],
                           "actions", 512)
        history = f.matrix(request.get("history"), strategic.SIZES["historySize"],
                           "history", f.MAX_HISTORY, empty=True)
        if not len(actions):
            raise ValueError("at least one legal action is required")
        # The eight strategic facts include a documented unknown = -1 sentinel;
        # all other base and card/action channels are nonnegative bounded values.
        base, extra = observation[:f.OBSERVATION_SIZE], observation[f.OBSERVATION_SIZE:]
        if (not np.all((base >= 0) & (base <= 2))
                or not np.all((extra >= -1) & (extra <= 2))
                or not np.all((actions >= 0) & (actions <= 2))
                or not np.all((history >= -1) & (history <= 2))):
            raise ValueError("out-of-range v3 actor/public feature")
        states = [(observation, actions, history)]
        with torch.inference_mode():
            logits, _ = model.policy_value(*p._tensor_states(model, states))
        values = logits[0].cpu().numpy()
        if len(values) != len(actions) or not np.isfinite(values).all():
            raise RuntimeError("invalid strategic policy scores")
        return {"scores": [float(value) for value in values]}
    if op == "close":
        return {"closed": True}
    raise ValueError("unknown op; expected info, score, or close")


def serve(model, incoming=None, outgoing=None):
    incoming, outgoing = incoming or sys.stdin, outgoing or sys.stdout
    while True:
        line = incoming.readline(MAX_REQUEST + 1)
        if not line:
            return
        request = None
        try:
            if len(line) > MAX_REQUEST:
                while line and not line.endswith("\n"):
                    line = incoming.readline(MAX_REQUEST + 1)
                raise ValueError("model request exceeds 8 MiB")
            request = json.loads(line)
            result = handle_request(model, request)
        except Exception as error:
            result = {"error": str(error)}
        outgoing.write(json.dumps(result, allow_nan=False) + "\n")
        outgoing.flush()
        if isinstance(request, dict) and request.get("op") == "close" and "error" not in result:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _ = p.load_checkpoint(args.checkpoint, f.device_for(args.device))
    if model.sizes != strategic.SIZES:
        raise ValueError("strategic inference requires a v3 team-PPO checkpoint")
    model.eval()
    serve(model)


if __name__ == "__main__":
    main()
