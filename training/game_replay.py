"""Private, offline match recordings; privileged snapshots never enter the policy."""

import gzip
import json
import os
from pathlib import Path
import uuid


def compact_game(game):
    return {
        "hands": [[card["id"] for card in hand] for hand in game["hands"]],
        "played": [[card["id"], card["player"]] for card in game["played"]],
        "target": {"ids": [card["id"] for card in game["target"]["cards"]],
                   "type": game["target"]["type"]} if game["target"] else None,
        **{key: game.get(key) for key in ("turn", "phase", "level", "levels", "dealer", "round",
                                         "lastPlayer", "lead", "passed", "order", "over", "result",
                                         "returnIndex", "logs", "questions")},
        "exchanges": [{**exchange, "tribute": exchange["tribute"]["id"],
                       "returned": exchange["returned"]["id"] if exchange["returned"] else None}
                      for exchange in game["exchanges"]],
    }


class GameReplay:
    def __init__(self, directory, episode_id, seed, snapshot, metadata):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = f"game-{episode_id:09d}"
        self.temporary = self.directory / f".{self.name}-{uuid.uuid4().hex}.partial"
        self.final = self.directory / f"{self.name}.jsonl.gz"
        self.index = self.directory / f"{self.name}.json"
        if self.final.exists() or self.index.exists():
            raise ValueError("replay already exists; never overwrite an actual match")
        self.raw = os.fdopen(os.open(self.temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
        self.stream = gzip.GzipFile(fileobj=self.raw, mode="wb", compresslevel=3, mtime=0)
        self.steps = 0
        self.played = len(snapshot["played"])
        self.sealed = False
        self.committed = False
        self.bytes = 0
        self.metadata = {
            "schemaVersion": 1, "episodeId": episode_id, "seed": seed, **metadata,
            "round": snapshot["round"], "level": snapshot["level"], "dealer": snapshot["dealer"],
            "scope": "omniscient offline replay only; never actor/critic input",
        }
        self.write({"record": "initial", "metadata": self.metadata, "game": snapshot})

    def write(self, value):
        encoded = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.bytes += len(encoded)
        if self.bytes > 128 * 1024 * 1024:
            raise RuntimeError("single game replay exceeds the 128 MiB uncompressed bound")
        self.stream.write(encoded)

    def step(self, snapshot, actor, op, action, previous_phase):
        if self.sealed:
            raise ValueError("cannot append to a finished replay")
        kind = (action["kind"] if action else "return" if previous_phase == "returning"
                else "play" if len(snapshot["played"]) > self.played else "pass")
        self.steps += 1
        self.write({"record": "step", "step": self.steps, "actor": actor, "op": op,
                    "action": {key: value for key, value in action.items() if key != "features"}
                    if action else None, "kind": kind, "state": compact_game(snapshot)})
        self.played = len(snapshot["played"])
        if snapshot["over"]:
            self.metadata.update(
                steps=self.steps, winner=snapshot["result"]["winner"], result=snapshot["result"],
                order=snapshot["order"], terminal=True)
            self.write({"record": "terminal", "metadata": self.metadata})
            self.stream.close()
            self.raw.flush()
            os.fsync(self.raw.fileno())
            self.raw.close()
            self.sealed = True

    def commit(self):
        from foundation import atomic_json_save
        if not self.sealed:
            raise ValueError("only complete real matches can be published")
        os.link(self.temporary, self.final)
        self.temporary.unlink()
        self.committed = True
        atomic_json_save(self.metadata, self.index)
        os.chmod(self.index, 0o600)

    def abort(self):
        if not self.sealed:
            self.stream.close()
            self.raw.close()
            self.sealed = True
        self.temporary.unlink(missing_ok=True)


def list_replays(directory):
    directory = Path(directory)
    return [json.loads(path.read_text()) for path in sorted(directory.glob("game-*.json"))]


def read_replay(directory, name):
    if not isinstance(name, str) or not name.startswith("game-") or len(name) != 14 or not name[5:].isdigit():
        raise ValueError("invalid replay identifier")
    path = Path(directory) / f"{name}.jsonl.gz"
    records = []
    total = 0
    with gzip.open(path, "rb") as stream:
        for line in stream:
            total += len(line)
            if total > 128 * 1024 * 1024:
                raise ValueError("replay exceeds size limit")
            records.append(json.loads(line))
    if not records or records[0]["record"] != "initial" or records[-1]["record"] != "terminal":
        raise ValueError("replay is incomplete")
    return {"initial": records[0], "steps": records[1:-1], "metadata": records[-1]["metadata"]}
