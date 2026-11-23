"""Train shared team action values with terminal-reward self-play."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import sys
import time


def arguments(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100000, help="additional completed games")
    parser.add_argument("--workers", type=int, default=4, help="persistent Node environments")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--mixed-teammate-probability", type=float, default=0.25,
                        help="chance to replace one learner-side teammate with the heuristic in one-team episodes")
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100, help="completed games between progress logs")
    parser.add_argument("--replay-size", type=int,
                        help="new run: 50000; resume: reuse the checkpoint capacity")
    parser.add_argument("--warmup", type=int, default=1024)
    parser.add_argument("--max-actions", type=int, default=1500)
    parser.add_argument("--tensorboard-dir", type=Path, default=root / "runs")
    parser.add_argument("--learn-returns", action="store_true",
                        help="continue completed rounds and learn tribute returns instead of using the heuristic")
    args = parser.parse_args(argv)
    for name in ("games", "workers", "batch_size", "save_every", "log_every",
                 "replay_size", "warmup", "max_actions"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers > 128:
        parser.error("--workers must be at most 128")
    if args.replay_size is not None and (args.batch_size > args.replay_size or args.warmup > args.replay_size):
        parser.error("--batch-size and --warmup must not exceed --replay-size")
    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed must be a uint32")
    if not 0 <= args.epsilon <= 1:
        parser.error("--epsilon must be between zero and one")
    if not 0 <= args.mixed_teammate_probability <= 1:
        parser.error("--mixed-teammate-probability must be between zero and one")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be finite and positive")
    return args


def run(args):
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from daguai_rl import (
        ActionValue, NodeEnvironment, ReplayBuffer, atomic_torch_save, checkpoint_data,
        choose_action, device_for, load_checkpoint, restore_training, score_many,
        seed_everything, team_reward, update_model,
    )

    device = device_for(args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    metadata = {
        "games": 0, "updates": 0, "seed": args.seed, "nextEpisode": 0,
        "modeCounts": {}, "snapshotPool": [], "reward": "terminal team parity: +1/-1/0",
        "algorithm": "deep Monte Carlo action value",
        "resumeRestartsInFlight": True, "mixedTeamGames": 0,
    }
    data = None
    if args.resume:
        model, data = load_checkpoint(args.resume, device)
        metadata.update(data["metadata"])
    else:
        model = ActionValue().to(device)
    if args.replay_size is None:
        if data is None:
            args.replay_size = 50000
        else:
            saved_replay = data.get("replay")
            if not isinstance(saved_replay, dict) or type(saved_replay.get("capacity")) is not int or saved_replay["capacity"] < 1:
                raise ValueError("resume requires a checkpoint with valid saved replay capacity")
            args.replay_size = saved_replay["capacity"]
    if args.batch_size > args.replay_size or args.warmup > args.replay_size:
        raise ValueError(f"batch-size and warmup must not exceed replay capacity {args.replay_size}")
    replay = ReplayBuffer(args.replay_size)
    metadata["continuationProbability"] = 1.0 if args.learn_returns else 0.0
    metadata["roundResetMode"] = "continue" if args.learn_returns else "fresh"
    metadata["learnReturns"] = metadata.get("learnReturns", False) or args.learn_returns
    metadata["mixedTeammateProbability"] = args.mixed_teammate_probability
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if data is not None:
        restore_training(data, optimizer, rng, replay, device)
        # A resumed optimizer retains its saved learning rate.
        del data
    model.eval()
    modes = Counter(metadata["modeCounts"])
    snapshots = []
    last_snapshot_games = None
    for path in metadata.get("snapshotPool", [])[-3:]:
        snapshot, snapshot_data = load_checkpoint(path, torch.device("cpu"))
        snapshots.append((str(Path(path).resolve()), snapshot))
        last_snapshot_games = snapshot_data["metadata"]["games"]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    workers = []
    starting_games = metadata["games"]
    goal = starting_games + args.games
    started = time.monotonic()
    loss = None
    next_save = (metadata["games"] // args.save_every + 1) * args.save_every
    next_log = (metadata["games"] // args.log_every + 1) * args.log_every
    status = "running"

    def save(*, snapshot=False):
        nonlocal last_snapshot_games
        metadata["modeCounts"] = dict(modes)
        metadata["status"] = status
        metadata["workers"] = len(workers)
        metadata["epsilon"] = args.epsilon
        metadata["batchSize"] = args.batch_size
        metadata["learningRate"] = optimizer.param_groups[0]["lr"]
        metadata["warmup"] = args.warmup
        metadata["maxActions"] = args.max_actions
        metadata["logEvery"] = args.log_every
        metadata["inFlightEpisodesDiscardedOnResume"] = sum(
            worker["state"] is not None for worker in workers
        )
        if snapshot and last_snapshot_games != metadata["games"]:
            path = (args.checkpoint_dir / f"snapshot-{metadata['games']:09d}.pt").resolve()
            atomic_torch_save(checkpoint_data(model, metadata), path)
            older, _ = load_checkpoint(path, torch.device("cpu"))
            snapshots.append((str(path), older))
            if len(snapshots) > 3:
                discarded, _ = snapshots.pop(0)
                # Never delete snapshots owned by another checkpoint directory.
                discarded_path = Path(discarded)
                if discarded_path.parent == args.checkpoint_dir.resolve():
                    discarded_path.unlink(missing_ok=True)
            last_snapshot_games = metadata["games"]
        metadata["snapshotPool"] = [path for path, _ in snapshots]
        path = args.checkpoint_dir / "latest.pt"
        atomic_torch_save(
            checkpoint_data(model, metadata, optimizer=optimizer, rng=rng, replay=replay), path
        )
        print(f"checkpoint={path.resolve()}", flush=True)

    def launch(worker):
        episode = metadata["nextEpisode"]
        metadata["nextEpisode"] += 1
        slot = episode % 4
        mode = "selfplay" if slot < 2 else "heuristic" if slot == 2 else "snapshot"
        old = None
        if mode == "snapshot":
            if snapshots:
                old = rng.choice(snapshots)[1]
            else:
                mode = "heuristic_fallback"
        learner_team = None if mode == "selfplay" else rng.randrange(2)
        heuristic_seats = set()
        if learner_team is not None and rng.random() < args.mixed_teammate_probability:
            heuristic_seats.add(rng.choice([learner_team, learner_team + 2, learner_team + 4]))
        recorded_mode = f"mixed_{mode}" if heuristic_seats else mode
        continuation = args.learn_returns and worker["finished"]
        seed = (metadata["seed"] + episode) & 0xFFFFFFFF
        state = worker["env"].state("reset", seed=seed, **({"continue": True} if continuation else {}))
        worker.update(state=state, trajectory=[], mode=recorded_mode, opponent_mode=mode,
                      learner_team=learner_team, heuristic_seats=heuristic_seats,
                      snapshot=old, actions_count=0, seed=seed, finished=False)
        if state["terminal"]:
            raise RuntimeError("reset returned a terminal episode without any decisions")

    def finish(worker):
        nonlocal loss, next_log
        winner = worker["state"]["result"]["winner"]
        for observation, features, player in worker["trajectory"]:
            replay.add(observation, features, team_reward(player, winner))
        metadata["games"] += 1
        modes[worker["mode"]] += 1
        if worker["heuristic_seats"]:
            metadata["mixedTeamGames"] += 1
        if len(replay) >= max(args.warmup, args.batch_size):
            loss = update_model(model, optimizer, replay, args.batch_size, rng, device)
            metadata["updates"] += 1
            writer.add_scalar("training/loss", loss, metadata["updates"])
        writer.add_scalar("training/replay_size", len(replay), metadata["games"])
        writer.add_scalar("training/games_per_second",
                          (metadata["games"] - starting_games) / max(time.monotonic() - started, 1e-9),
                          metadata["games"])
        worker.update(state=None, trajectory=[], snapshot=None, finished=True)
        if (metadata["games"] == starting_games + 1 or metadata["games"] >= next_log
                or metadata["games"] == goal):
            speed = (metadata["games"] - starting_games) / max(time.monotonic() - started, 1e-9)
            loss_text = "none" if loss is None else f"{loss:.6f}"
            print(f"games={metadata['games']} updates={metadata['updates']} loss={loss_text} "
                  f"games/sec={speed:.3f} modes={dict(modes)}", flush=True)
            next_log = (metadata["games"] // args.log_every + 1) * args.log_every

    try:
        writer = SummaryWriter(str(args.tensorboard_dir))
        print(f"device={device} seed={metadata['seed']} target_games={goal}", flush=True)
        if device.type == "cuda":
            print(f"CUDA GPU={torch.cuda.get_device_name(device)} torch={torch.__version__}", flush=True)
        for _ in range(min(args.workers, args.games)):
            worker = {"env": NodeEnvironment(max_actions=args.max_actions), "state": None, "finished": False}
            workers.append(worker)
            launch(worker)
        while metadata["games"] < goal:
            current_jobs = []
            older_jobs = {}
            heuristic_jobs = []
            for worker in workers:
                state = worker["state"]
                if state is None:
                    continue
                if worker["actions_count"] >= args.max_actions:
                    raise RuntimeError(
                        f"episode seed={worker['seed']} exceeded --max-actions={args.max_actions}; "
                        "not a rewarded terminal truncation"
                    )
                is_teammate_heuristic = state["player"] in worker["heuristic_seats"]
                is_current = (not is_teammate_heuristic and
                              (worker["learner_team"] is None or state["player"] % 2 == worker["learner_team"]))
                if state["phase"] == "returning" and not args.learn_returns:
                    heuristic_jobs.append(worker)
                elif is_current:
                    current_jobs.append(worker)
                elif not is_teammate_heuristic and worker["opponent_mode"] == "snapshot":
                    old = worker["snapshot"]
                    older_jobs.setdefault(id(old), (old, []))[1].append(worker)
                else:
                    heuristic_jobs.append(worker)

            def inputs(jobs):
                return [(worker["state"]["observation"],
                         [action["features"] for action in worker["state"]["actions"]])
                        for worker in jobs]

            decisions = []
            if current_jobs:
                predictions = score_many(model, inputs(current_jobs))
                for worker, scores in zip(current_jobs, predictions):
                    index = choose_action(scores, args.epsilon, rng)
                    state = worker["state"]
                    chosen = state["actions"][index]
                    worker["trajectory"].append((
                        np.asarray(state["observation"], dtype=np.float32),
                        np.asarray(chosen["features"], dtype=np.float32), state["player"],
                    ))
                    decisions.append((worker, "step", {"index": index}))
            for old, jobs in older_jobs.values():
                for worker, scores in zip(jobs, score_many(old, inputs(jobs))):
                    decisions.append((worker, "step", {"index": int(np.argmax(scores))}))
            decisions.extend((worker, "heuristic", {}) for worker in heuristic_jobs)
            for worker, op, kwargs in decisions:
                worker["state"] = worker["env"].state(op, **kwargs)
                worker["actions_count"] += 1
                if worker["state"]["terminal"]:
                    finish(worker)
            if metadata["games"] >= next_save:
                save(snapshot=True)
                next_save = (metadata["games"] // args.save_every + 1) * args.save_every
            for worker in workers:
                in_flight = sum(other["state"] is not None for other in workers)
                if worker["state"] is None and metadata["games"] + in_flight < goal:
                    launch(worker)
        status = "completed"
        save(snapshot=True)
        print(f"completed games={metadata['games']} updates={metadata['updates']}", flush=True)
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        print("Interrupted; saving completed-game replay and optimizer.", file=sys.stderr)
        save()
        return 130
    except Exception:
        status = "failed"
        save()
        raise
    finally:
        for worker in workers:
            worker["env"].close()
        if writer is not None:
            writer.close()


def main(argv=None):
    args = arguments(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"training error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
