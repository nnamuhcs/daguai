"""Reference diagnostics, not an oracle or evidence of increased winning strength."""

import argparse
from pathlib import Path

import torch

import foundation as f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists")
    checkpoints = {}
    for value in args.checkpoint:
        label, separator, path = value.partition("=")
        if not separator or not label or not Path(path).is_file() or label in checkpoints:
            parser.error("checkpoints must have unique LABEL=existing_PATH entries")
        checkpoints[label] = Path(path)
    torch.set_num_threads(1)
    result = f.tactical_audit(checkpoints)
    result["checkpointSha256"] = {label: f.sha256_file(path) for label, path in checkpoints.items()}
    result["strengthClaim"] = False
    f.atomic_json_save(result, args.report)
    print(f"Reference diagnostic decisions: {len(result.get('records', []))}; not a promotion gate by themselves")


if __name__ == "__main__":
    main()
