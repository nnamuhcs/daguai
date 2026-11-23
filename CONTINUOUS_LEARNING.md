# Daguai — continuous learning, evidence, and safe model selection

**Status:** The supplied continuous PPO train/evaluate/select controller is integrated with full-team-only evaluation. **This is not proof of strong play or a guarantee of monotonically increasing skill.** Training can improve, plateau, or regress. The controller protects the *selected champion* from unproven replacement.

## What is actually continuous

1. The PPO trainer plays another bounded set of **complete** self-play games. It uses the trained policy in a mixture with rule and historical opponents and saves model, optimizer, and random-number state (no stale off-policy PPO replay).
2. The controller freezes the new checkpoint under `cycles/NNNNNN/candidate.pt`.
3. It plays **3-vs-3 only**, on identical seeded deals with sides swapped. The candidate and incumbent (initially the heuristic) each face the unchanged original heuristic team. This measures their difference against the same opponent, not a direct candidate–incumbent match.
4. If the validation gain over the incumbent exceeds the configured minimum, it repeats the comparison on a **separate confirmation seed family**. Promotion requires the lower bound of the paired confirmation interval to exceed the same minimum.
5. It promotes successful checkpoints to the immutable `champions/` directory. Unsuccessful checkpoints are retained for analysis, not deployed. It also preserves an *exploratory* research-best neural model, clearly not a statistically validated champion.
6. It saves `progress.json`, `plan.json`, immutable evaluation reports, and private training replays under `replays/`. Restarting the same command **without source/init flags** resumes the existing PPO checkpoint and incomplete comparison instead of starting again.
7. Repeat until Ctrl+C or `--cycles` is reached. Training doesn't stop just because one candidate fails promotion. A training exception **does** stop the loop so a mathematical or privacy bug cannot silently repeat forever.

This is **continuous learning with guarded selection**, not an automated proof of expert-level play.

## First-time setup — WSL2 RTX 5090

From your WSL2 machine, where the real project and its checkpoints live:

```bash
cd ~/daguai-web
python3 -m venv .venv  # skip if already created
.venv/bin/pip install -r training/requirements.txt
node --test
.venv/bin/python -m unittest discover -s training -p 'test*.py' -q
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

**Preferred existing-checkpoint path:** If you already have an operational PPO run, specify its **directory containing `latest.pt` and `initial.pt`** and reuse it (never recreate these files):

```bash
.venv/bin/python training/continuous_learn.py \
  --checkpoint-dir training/ppo/checkpoints/strategic \
  --root training/ppo/my-continuous-run \
  --stage-games 128 --evaluation-games 400 --confirmation-games 1000 --device cuda
```

Adjust the checkpoint directory to match your actual machine; **do not assume the example path exists**. For an existing run, pass no `--from-scratch`, `--warm-start`, or `--prepared-start` flag.

**If starting a new run from the saved best v2 Monte Carlo actor**, point at the real local v2 checkpoint:

```bash
.venv/bin/python training/continuous_learn.py \
  --warm-start training/foundation/checkpoints/milestone-000001000.pt \
  --stage-games 128 --evaluation-games 400 --confirmation-games 1000 --device cuda
```

**If you already distilled the v2 behavior and verified the policy transfer**, prefer:

```bash
.venv/bin/python training/continuous_learn.py \
  --prepared-start training/ppo/reports/pre-ppo/distilled.pt \
  --stage-games 128 --evaluation-games 400 --confirmation-games 1000 --device cuda
```

Alternatively, `--from-scratch` starts a random v2 PPO actor; this generally requires a substantially larger training budget. Starting a **new v3 strategic residual** requires a *completed v2 PPO checkpoint with the random-origin lineage accepted by `--ablate-from`*:

```bash
.venv/bin/python training/continuous_learn.py \
  --ablate-from training/ppo/checkpoints/latest.pt --strategic \
  --stage-games 128 --evaluation-games 400 --confirmation-games 1000 --device cuda
```

You **must** use a new `--root` and `--checkpoint-dir` if you want to run separate experiments. The existing game on port 3000 stays unchanged.

## Continue indefinitely

After the first cycle (or once you have a complete controller plan), you **resume without the initialization flag**:

```bash
cd ~/daguai-web
.venv/bin/python training/continuous_learn.py --device cuda
```

Repeat the **same** training and evaluation plan flags as the original run (including `--root`, any custom `--checkpoint-dir`, game counts, worker/rollout settings, seeds, and thresholds). `--cycles 1` is a bounded verification mode. The default `--cycles 0` continues until interrupted or a safety check stops it.

Logs and status:

```bash
.venv/bin/python training/learner_status.py
.venv/bin/python training/learner_status.py --json
.venv/bin/tensorboard --logdir training/ppo/continuous/tensorboard --host 127.0.0.1 --port 6006
```

Check `training/ppo/continuous/progress.json`, `cycles/NNNNNN/summary.json`, and `checkpoints/metrics.jsonl`. The player shown by the browser is **not** automatically replaced by the trainer.

### Maintainer's local run

The maintainer's continuous run in `training/ppo/continuous/` starts from the verified
1,280-game v2 PPO checkpoint with the strategic v3 upgrade. Its plan uses 128
new training games and 400 validation games per matchup per cycle; a promising
result triggers 1,000 confirmation games per matchup on different deals.
Its checkpoint, reports, and replays are local artifacts, not part of a fresh clone.

Open the live [learning dashboard](http://localhost:8071) for plain-language
results, live cycle progress, and training replays. To start the dashboard from
the project directory:

```bash
.venv/bin/python training/progress_dashboard.py --continuous \
  --run-dir training/ppo/continuous --port 8071
```

Or run `watch -n 5 .venv/bin/python training/learner_status.py`
to see the live training/evaluation count, completed 3-vs-3 scores, paired
confidence intervals, and champion. TensorBoard at `http://localhost:6007`
shows training metrics, not whether the player is stronger.

If this run is stopped, resume it from the project directory without the
initialization flag:

```bash
.venv/bin/python training/continuous_learn.py \
  --stage-games 128 --evaluation-games 400 --confirmation-games 1000 \
  --train-seed 3332000 --evaluation-seed 1100340000 --confirmation-seed 1200340000 \
  --device cuda
```

## Keep running as a WSL2 user service (optional)

First initialize the model/controller and complete one bounded cycle. Then use `deploy/daguai-learner.service` as a template:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/daguai-learner.service ~/.config/systemd/user/daguai-learner.service
systemctl --user daemon-reload
systemctl --user enable --now daguai-learner.service
systemctl --user status daguai-learner.service
journalctl --user -u daguai-learner.service -f
```

The example assumes `~/daguai-web` and the default controller arguments. Adjust the paths and flags to match your run plan before enabling it. WSL2 must remain running and CUDA must be available; a service does **not** continue training while Windows is shut down or sleeping. Do not run a manual second trainer on the same checkpoint directory.

## Inspect or serve a champion deliberately

The training loop **never updates an already-running Python model worker** or the browser default. When a checkpoint has cleared validation **and** separate confirmation, you can explicitly launch the selected champion for personal testing:

```bash
npm run learn:status
npm run start:champion
```

`start:champion` verifies that the chosen path is in this run's `champions/` directory, checks its SHA256 digest, and selects the v2 or v3 inference worker. This starts a **new** local server on port 3000, so stop an existing server first. Winning against the heuristic is *not* evidence of expert-human strength and does not authorize automatic deployment to a third-party platform.

## Verification and limits

See `IMPLEMENTATION_STATUS.md` for the local results. A bounded test checks that the pipeline works, not that its candidate is stronger. The optional systemd service is not installed by the controller.

## Safety rules for an implementing agent

- Never change the simulator's hidden-hand boundary to improve training scores; actor observations may use only its own cards and observable public information.
- Never call training loss, played game count, or an action-value estimate a win rate. The true score is `(wins + 0.5*draws) / games` on frozen, paired evaluations.
- Do not change the recorded run plan or game rules without creating a new experimental lineage.
- Do not train on final test seeds or repeatedly tune hyperparameters on the confirmation or final seeds. The confirmation family used here is for conservative local promotion; independent final/human evaluation remains necessary.
- Do not automatically deploy a policy that failed evaluation. The browser stays heuristic until a separate, deliberate choice.
- Do not claim training will always improve. Data, optimization, stochastic deals, model capacity, and opponent diversity can all limit progress.

### Recommended v3 upgrade from your existing learned PPO

If your best PPO checkpoint is still **v2**, the `--upgrade-to-strategic` option creates a new v3 actor with **the exact inherited v2 score path** and a zero-output strategic residual. It accepts a completed v2 model even when it was initialized from the earlier Monte Carlo learner, and resets the optimizer without pretending to inherit the old optimizer's moments. This is preferable to starting the v3 policy from random weights:

```bash
.venv/bin/python training/continuous_learn.py \
  --upgrade-to-strategic training/ppo/checkpoints/latest.pt \
  --root training/ppo/v3-continuous \
  --checkpoint-dir training/ppo/v3-continuous/checkpoints \
  --stage-games 128 --evaluation-games 400 --confirmation-games 1000 --device cuda
```

The source **must be completed v2 PPO**. Do not pass a v1/v2 Monte Carlo file or an already-v3 PPO file. If `latest.pt` is paused, use the *last verified completed* checkpoint instead. On resuming this new experiment, use the same flags but **omit `--upgrade-to-strategic`**.

## Inspect learning evidence

`npm run learn:evidence` reads the existing run's weight changes, optimizer
updates, and recent full-team results, showing the first test and separate
confirmation when available. Use `-- --root path/to/run` for a different run.
Changed weights don't by themselves establish stronger play.

To compare two frozen checkpoints against the original web heuristic on the
same side-swapped 3-vs-3 debug deals:

```bash
npm run learn:evidence -- \
  --compare-old /path/to/frozen-old.pt --compare-new /path/to/frozen-new.pt \
  --games 400 --seed 20050000 --decision-samples 100 \
  --output training/reports/learning-evidence.json
```

This comparison is exploratory and never promotes either checkpoint. Use
`--rounds continued` and a separate output file to include tribute and returns.
To examine *why a specific move may have helped or hurt* under a fixed
heuristic continuation, see the [counterfactual decision audit](COUNTERFACTUAL_DECISION_AUDIT.md).
