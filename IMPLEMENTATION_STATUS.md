# Implementation and verification status — 2026-11-22

This file describes **verified behavior of the source archive**, not a claim that the neural network has reached expert playing strength.

## Implemented in the local integration

- Existing six-player game engine, legal-action masks, and v2/v3 private-observation schema retained.
- Earlier opt-in strategic v3 local inference path (`serve_strategic.py` and browser/server routing) retained; fingerprint-checked.
- New `training/continuous_learn.py`: repeated bounded PPO training, immutable cycle checkpoints, paired **3-vs-3 only** tournaments against the unchanged original heuristic, fresh validation seeds per cycle, separate confirmation seeds for promising checkpoints, guarded local champion promotion, atomic progress/evidence files, and restart from a training boundary. New training games are privately recorded under the run's `replays/`.
- New `training/learner_status.py`: human-readable or JSON progress, per-cycle score and paired confidence intervals.
- New `start-champion.js`: deliberate serving of an already-promoted champion with verified SHA256/checkpoint location/encoding. No implicit browser rollout.
- New `--upgrade-to-strategic`: score-preserving v2 PPO -> strategic v3 transfer, including checkpoints whose original weights came from v2 Monte Carlo; new optimizer state, explicit PPO-parent provenance.
- New `training/training_writer.py`: TensorBoard when installed; JSONL metrics fallback only with `DAGUAI_ALLOW_JSON_LOGGER=1` for offline environments.
- New tests for continuous controller, transfer lineage, score retention, and plan immutability. WSL2 service template and runbook.
- `learn:evidence` reports parameter changes and separately labeled validation and confirmation results, with an optional 3-vs-3 matched-deal comparison of frozen checkpoints.
- `learn:decisions` runs bounded counterfactual action comparisons over sampled hidden hands. It is an offline diagnostic, not part of PPO, browser play, or promotion. See [audit design and limits](COUNTERFACTUAL_DECISION_AUDIT.md).

## Verification reported by the earlier continuous-learning archive

| Check | Result |
| --- | --- |
| `node --test` | 58 passed, 0 failed |
| `test_continuous_learn` | 5 passed |
| `test_strategic_model` | 3 passed |
| `test_strategic_inference` | 2 passed |
| Targeted PPO GAE/trace/rollout/update tests | 4 passed |
| `test_strategic_upgrade` | 1 passed, including transferred v2 lineage |
| CPU-only complete PPO cycle (8 games) | Trained, saved, evaluated, rejected weak checkpoint. The local integration does not run the archive's mixed-team comparison. |
| Direct v2→v3 action-logit equivalence | Absolute difference = 0.0 on constructed test state |

The archive did not complete a CPU resume cycle or its full Python suite. Its checks are not a substitute for local verification.

## Local WSL2 verification

- 58 JavaScript tests, 119 Python tests, the default browser smoke test, and opt-in v3 browser/HTTP inference passed.
- The RTX 5090 completed two eight-game cycles in `training/ppo/continuous-import-smoke-001/`, restarting the controller between cycles. Both checkpoints passed the frozen-policy trace audit, and all 16 training replays are complete.
- In 12 paired 3-vs-3 games per cycle against the original web AI, the candidates scored 41.7% and 33.3%. This checks the pipeline, not playing strength. No candidate reached confirmation or promotion, no final split was opened, and the browser default stayed unchanged.
- The service was not installed. Recovery was verified between committed cycles, not after a crash inside a training batch.
- The counterfactual update passed the JavaScript suite and targeted Python tests. Both audit commands were exercised against local frozen v3 checkpoints; the matched-deal smoke test used only full teams on debug seeds.

## What remains empirically unproven

- Reliable improvement over the original heuristic on the sealed final release split. Selection on separate confirmation deals is not a release result.
- Calibrated *learned* opponent-hand posteriors; the existing belief sampler and tactical-goal ablations are not equivalent to a proven learned belief model.
- Game-theoretically sound full-depth information-set search; the existing bounded strategic planner is an experimental approximation, not exhaustive search.
- Adaptive team coordination in expert-level competitive play.
- Integration with a real human-playing platform and strong-human ability.
- Long-run CUDA throughput and continuously running WSL service.

No program can truthfully mark those as passed based on code/tests alone. What this implementation provides is a **continuously operating experiment and selection system** that can gather the evidence needed to pursue them without shipping regressions as successes.
