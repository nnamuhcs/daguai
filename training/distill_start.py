"""Fit PPO's categorical actor to its own frozen source's epsilon policy.

This is initialization, not reinforcement learning or evidence of better play.
Independent debug seeds measure retention; final evaluation seeds stay sealed.
"""

import argparse
import copy
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

import foundation as f
import team_ppo as p


def epsilon_target(scores, epsilon=0.15):
    if scores.ndim != 1 or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError("source scores must be a finite nonempty vector")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between zero and one")
    target = torch.full_like(scores, epsilon / len(scores))
    target[int(torch.argmax(scores))] += 1 - epsilon
    return target


def retention(logits, targets, temperature):
    rows = []
    for scores, target in zip(logits, targets):
        if len(scores) == 1:
            continue
        log_prob = F.log_softmax(scores / temperature, dim=0)
        rows.append((float(torch.sum(target * (target.log() - log_prob))),
                     int(scores.argmax() == target.argmax()),
                     float(log_prob[target.argmax()].exp())))
    if not rows:
        raise ValueError("retention requires nonforced decisions")
    return {"nonforcedPositions": len(rows), "meanKL": float(np.mean([r[0] for r in rows])),
            "topAgreement": float(np.mean([r[1] for r in rows])),
            "meanPreferredProbability": float(np.mean([r[2] for r in rows])),
            "passed": bool(np.mean([r[0] for r in rows]) < 0.05
                           and np.mean([r[1] for r in rows]) > 0.98)}


@torch.no_grad()
def collect_contexts(teacher, count, seed, epsilon):
    device = next(teacher.parameters()).device
    env = f.NodeEnvironment(script=f.ROOT / "audit-env.js")
    entries = []
    try:
        for index in range(count):
            state = env.state("sample", seed=seed + index * 64,
                              phase=("fresh", "mid", "late", "returning")[index % 4])
            if len(state["actions"]) == 1:
                continue
            obs = torch.tensor([state["observation"]], device=device, dtype=torch.float32)
            history = torch.from_numpy(f.relative_history(state["events"], state["player"])).to(device)
            context = torch.cat((teacher.base(obs), teacher.encode_history([history])), dim=1)
            actions = torch.tensor([a["features"] for a in state["actions"]],
                                   device=device, dtype=torch.float32)
            combined = torch.cat((context.expand(len(actions), -1), actions), dim=1)
            scores = teacher.head(combined).squeeze(-1)
            entries.append((combined.cpu(), epsilon_target(scores, epsilon).cpu()))
    finally:
        env.close()
    if not entries:
        raise ValueError("sampled no nonforced positions")
    return entries


@torch.no_grad()
def evaluate_head(model, entries, temperature):
    device = next(model.parameters()).device
    scores = [model.head(features.to(device)).squeeze(-1).cpu() for features, _ in entries]
    return retention(scores, [target for _, target in entries], temperature)


def fit_head(model, train, validation, epochs, learning_rate, temperature, seed):
    device = next(model.parameters()).device
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    before = evaluate_head(model, validation, temperature)
    best_metric = before
    best = copy.deepcopy(model.head.state_dict())
    history = []
    for epoch in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), 8):
            losses = []
            for index in order[start:start + 8]:
                features, target = train[index]
                scores = model.head(features.to(device)).squeeze(-1)
                log_prob = F.log_softmax(scores / temperature, dim=0)
                target = target.to(device)
                losses.append(torch.sum(target * (target.log() - log_prob)))
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite behavior-distillation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        measured = evaluate_head(model, validation, temperature)
        history.append({"epoch": epoch + 1, **measured})
        print(f"epoch={epoch + 1} KL={measured['meanKL']:.5f} "
              f"topAgreement={measured['topAgreement']:.4f}", flush=True)
        if measured["meanKL"] < best_metric["meanKL"]:
            best_metric, best = measured, copy.deepcopy(model.head.state_dict())
        if measured["passed"]:
            break
    model.head.load_state_dict(best)
    model.requires_grad_(True)
    model.eval()
    return {"before": before, "after": best_metric, "epochs": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=512)
    parser.add_argument("--validation-positions", type=int, default=256)
    parser.add_argument("--confirmation-positions", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20_200_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    for size in (args.positions, args.validation_positions, args.confirmation_positions):
        if not 4 <= size <= 4096:
            parser.error("position counts must be 4..4096")
    if not 1 <= args.epochs <= 100 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("invalid bounded epochs or learning rate")
    if not np.isfinite(args.temperature) or args.temperature <= 0 or not 0 < args.epsilon < 1:
        parser.error("invalid categorical temperature or source epsilon")
    total = args.positions + args.validation_positions + args.confirmation_positions
    if not 20_000_000 <= args.seed <= args.seed + total * 64 < 21_000_000:
        parser.error("distillation may only use the debug seed family")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    teacher, source_data = f.load_checkpoint(args.source, args.device)
    if source_data["metadata"].get("epsilon") != args.epsilon:
        raise ValueError("distillation epsilon must match the source training policy")
    model, metadata = p.warm_start(args.source, args.device)
    digest = f.sha256_file(args.source)
    train = collect_contexts(teacher, args.positions, args.seed, args.epsilon)
    validation = collect_contexts(teacher, args.validation_positions,
                                  args.seed + args.positions * 64, args.epsilon)
    fitting = fit_head(model, train, validation, args.epochs, args.learning_rate,
                       args.temperature, args.seed)
    f.atomic_json_save(fitting, args.output_dir / "fitting.json")
    confirmation = collect_contexts(
        teacher, args.confirmation_positions,
        args.seed + (args.positions + args.validation_positions) * 64, args.epsilon)
    confirmed = evaluate_head(model, confirmation, args.temperature)
    if f.sha256_file(args.source) != digest:
        raise RuntimeError("source checkpoint changed during distillation")
    passed = bool(fitting["after"]["passed"] and confirmed["passed"])
    report = {
        "schemaVersion": 1, "sourceSha256": digest, "sourceEpsilon": args.epsilon,
        "temperature": args.temperature, "seed": args.seed, "trainingPositions": len(train),
        "requestedTrainingPositions": args.positions,
        "requestedValidationPositions": args.validation_positions,
        "requestedConfirmationPositions": args.confirmation_positions,
        "trainable": "actor head only; own/public base and GRU preserved",
        "fitting": fitting, "confirmation": confirmed, "behaviorPreserved": passed,
        "selection": "best validation KL; confirmation never used to select or update",
        "strengthClaim": False, "reward": "no RL reward in behavior distillation",
    }
    if passed:
        metadata["lineage"]["actorTransfer"] = "source-behavior-distilled"
        metadata["initialization"] = "source-behavior-distilled v2 actor"
        metadata["behaviorDistillation"] = report
        metadata["config"] = {**p.CORE_DEFAULTS, "temperature": args.temperature}
        f.immutable_torch_save(p.checkpoint_data(model, metadata), args.output_dir / "prepared.pt")
    f.atomic_json_save(report, args.output_dir / "report.json")
    print(f"Independent behavior-retention gate passed={passed}; "
          f"confirmation KL={confirmed['meanKL']:.5f}; no strength claim")


if __name__ == "__main__":
    main()
