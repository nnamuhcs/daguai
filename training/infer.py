"""Load one immutable checkpoint and serve action scores over JSON lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def handle_request(request, model, metadata, device, checkpoint):
    from daguai_rl import OBSERVATION_SIZE, SIZES, finite_vector, feature_matrix, score_many

    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    op = request.get("op")
    if op == "info":
        return {
            "available": True, "games": metadata["games"], "updates": metadata["updates"],
            "learnReturns": metadata.get("learnReturns", False),
            **SIZES, "device": str(device), "checkpoint": str(checkpoint),
        }
    if op == "score":
        observation = finite_vector(request.get("observation"), OBSERVATION_SIZE, "observation")
        actions = feature_matrix(request.get("actions"), maximum=512)
        scores = score_many(model, [(observation, actions)])[0]
        return {"scores": scores.tolist()}
    if op == "close":
        return {"closed": True}
    raise ValueError("unknown op; expected info, score, or close")


def serve(model, metadata, device, checkpoint, input_stream=None, output_stream=None):
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    # Read bounded lines; invalid oversized requests are drained, never scored.
    maximum_bytes = 4 * 1024 * 1024
    while True:
        line = input_stream.readline(maximum_bytes + 1)
        if not line:
            return
        request = None
        try:
            if len(line) > maximum_bytes:
                while line and not line.endswith("\n"):
                    line = input_stream.readline(maximum_bytes + 1)
                raise ValueError("request exceeds 4 MiB")
            request = json.loads(line)
            response = handle_request(request, model, metadata, device, checkpoint)
        except Exception as error:
            response = {"error": str(error)}
        output_stream.write(json.dumps(response, allow_nan=False) + "\n")
        output_stream.flush()
        if isinstance(request, dict) and request.get("op") == "close" and "error" not in response:
            return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--stdio", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        from daguai_rl import device_for, load_checkpoint
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        device = device_for(args.device)
        model, data = load_checkpoint(args.checkpoint, device)
        metadata = data["metadata"]
        del data
        serve(model, metadata, device, args.checkpoint.resolve())
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"inference startup/runtime error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
