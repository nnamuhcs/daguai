# Counterfactual Decision Audit — Design and Verification

This project now supports a bounded, paired action evaluator. Its purpose is to
answer: *which card decisions changed, and do simulations suggest that the new
decisions improve expected team outcomes?*

## Procedure
1. Choose a frozen policy checkpoint and sample a legal decision state.
2. Use the acting player's own cards, public played cards, card counts, ask
   signals and actor-known transfers to sample public-consistent hidden worlds.
3. Preserve original actor card IDs and public played IDs; assign only remaining
   physical deck cards to the five unknown hands.
4. For each sampled world, apply each alternative legal first action. Finish
   the rest of the round using the same engine heuristic continuation policy.
5. Score each completed game as 1 team win, 0.5 draw, 0 team loss.
6. Compare actions using paired outcomes and approximate confidence intervals.
7. Save a JSON report with the policy's decision, candidate rankings, uncertainty,
   estimated regret, and optional comparison with an older checkpoint.

## Boundaries and known limitations
- The sampler is approximate, especially for complex tribute constraints.
- A single determinization can let the continuation heuristic react to perfect
  simulator knowledge; it is NOT a sound information-set equilibrium solver.
- The rollout continuation is not the learned policy. It evaluates alternatives
  under a fixed heuristic future, and that can misrank moves.
- Inferences from the same sampled worlds are paired, not independent. The
  confidence interval is approximate, and the max-of-candidates estimate has
  selection bias. A future bootstrap on independent world batches may help.
- Passes are soft evidence and are not treated as proof the seat cannot beat.
- The code refuses to estimate when fewer than four valid worlds are available.
- No results from this audit are used as PPO advantages, returns, targets, or
  automatic champion-promotion signals. These require independent validation.
- Actual expert-level strength is only established via independent evaluation,
  ideally including games against skilled human teams.

## Verification
- `node --test training/counterfactual-env.test.js`
- `PYTHONPATH=training python -m unittest test_decision_quality -q`
- `npm test`
- For an end-to-end trained-checkpoint test, run `npm run learn:decisions` using
  *your own local frozen checkpoint files*, which are intentionally not shipped.

## Run it

```bash
npm run learn:decisions -- \
  --checkpoint /path/to/frozen-new.pt \
  --compare-old /path/to/frozen-old.pt \
  --positions 12 --worlds 8 --top-k 4 \
  --output training/reports/decision-audit.json
```

`--compare-old` is optional. The audit evaluates the new model's choice, the old
model's choice when supplied, a legal pass when available, and other highly
scored candidates within the rollout budget. Reports distinguish the played
cards or ask target, estimated outcomes, and approximate paired intervals.
Use a new output filename for each audit. Position seeds are restricted to the
debug split; the tool never opens validation, confirmation, or final deals.

`npm run learn:evidence` reads existing weight and match evidence without
playing any games. See [the runbook](CONTINUOUS_LEARNING.md#inspect-learning-evidence)
for an optional full-team comparison on matching debug deals.

### Privacy invariant
Replace all true opponent hidden hands while keeping the acting player's hand
and public history unchanged. For identical seed and selected actions, the
counterfactual audit must return identical outcome estimates.

### Promotion invariant
This is diagnostic-only. A favorable decision-quality score is never sufficient
to promote a checkpoint. Continue to require independent matched-deal
championship evaluations.
