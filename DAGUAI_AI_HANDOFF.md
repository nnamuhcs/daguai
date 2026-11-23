# Daguai Web: Project History, AI Architecture, Results, and Review Handoff

> Subsequent blueprint implementation and new measurements are documented in
> [the implementation status](IMPLEMENTATION_STATUS.md). Paths to checkpoints,
> report bundles and replays in this historical handoff are local artifacts.
> This document preserves the earlier handoff; its automatic-final-monitor behavior
> has been superseded by validation-only monitoring.
> A separately authorized random-initialized PPO pilot completed in
> `training/reports/fresh-ppo-pilot-001/`; it inherits no old model weights.
> After 256 PPO games, paired full-team score improved from .1000 to .3725
> (difference 95% CI [.2175, .3275]), but remained below the heuristic benchmark.
> A subsequent 1,024-additional-game extension is tracked in
> `training/reports/ppo-extension-001/`, with original-web-heuristic comparisons
> every 256 games and complete new-match replays at http://localhost:8070/replays.
> Completed full-team scores: .3900 at 512 games, .5275 at 768, .4675 at 1,024.
> The latter two improved over the 256-game model but did not establish superiority
> over the original web AI. A CUDA TF32 recurrence-consistency failure stopped
> training safely at 1,128 games; FP32 fixed the exactly reproduced failure.
> The bounded extension completed at 1,280 total games, preserving the original
> evidence and all 1,024 new-game replays. Final direct three-versus-three
> evaluation: 49 wins, 49 losses, 102 draws; .5000 score, no established advantage
> over the original web AI.
> See the implementation status for measured errors and archived recovery state.
> The current comparison scope is now exclusively three learned AI versus three
> original web AI; previous mixed-team reports are archived, not active results.
> The core PPO model does not learn explicit beliefs or goal heads. An opt-in
> sampled-belief and adaptive-goal experiment is now available.

## 1. Purpose of this document

This is a detailed handoff for another model or engineer to review the project.
It covers the game implementation, the user's actual requirements, successive
AI approaches, measured failures, the latest team-PPO implementation, and what
has **not** been demonstrated.

**Please distinguish these three things throughout the review:**

1. The game and training software run correctly.
2. A neural network's parameters actually change through training.
3. Those changes produce better card play.

The first two have been verified. Strong card play has not.

## 2. Current status at handoff

| Component | Actual status |
| --- | --- |
| Six-player web game | Implemented and served on port 3000 |
| Default browser AI | Heuristic/rule AI, not automatically replaced by a neural model |
| Original neural learner, v1 | Implemented; its long run was stopped and preserved |
| Public-history Monte Carlo learner, v2 | Implemented; later weights measurably regressed |
| Old v2 long trainer | Intentionally paused with SIGSTOP, not terminated |
| New shared-team PPO learner | Implemented and verified on the actual RTX 5090 |
| PPO verification training | 40 new PPO games, 468 optimizer steps, saved successfully |
| PPO cumulative lineage counter | 1,040: 1,000 source-model games plus 40 PPO games |
| PPO HTTP/browser inference | Verified using the 1,032-game verification checkpoint |
| PPO CUDA resume | Verified from 1,032 to 1,040 cumulative games |
| PPO warm-start monitoring | End-to-end integration verified with tiny evaluations |
| Meaningful PPO strength assessment | **Not performed yet** |
| Proposed 2,000-game PPO trial | **Not started**; this handoff was requested for feedback |
| Previous automatic reporting schedule | Cancelled during the transition; no new PPO schedule created yet |

The old v2 trainer was paused because an immediate, fair assessment of its
8,000-game checkpoint showed regression against its 1,000-game checkpoint.
It is not honest to describe the current situation as "more training is
definitely making it stronger."

The latest work deliberately uses the **best measured v2 checkpoint** as a PPO
warm start, rather than blindly transferring the newest, weaker weights.
"Best measured" here means best among the assessed v2 checkpoints, not a claim
that it beats human experts or every possible baseline.

## 3. What the user actually wants

The original request was a web version of Shanghai 大怪路子, not a simplified
card-shedding demo.

The user subsequently demanded:

- Complete tribute, return, asking, settlement, and turn-order rules.
- Readable cards and an interface that makes the last play and finished players obvious.
- Teammates that accept useful feeds instead of passing mechanically.
- Actual hand decomposition and recoverable routes, not formulas based only on card count.
- Sensible use of Jokers and trump cards instead of wasting them on weak combinations.
- A local model that genuinely learns, with visible progress and evidence of improvement.
- Learning from fundamentals, rather than merely imitating the existing weak heuristic.
- Applicable lessons from AlphaGo Zero and OpenAI Five, especially team coordination.

The user's most recent strategic clarification was:

> They should first evaluate their own cards, decide whether to push to finish
> or support others, account for a bad deal, remember played cards, infer the
> strength and roles of other players, and choose how to make their team win.

The intended behavior is **dynamic, belief-based team planning**:

- Evaluate combinations, control, flexibility, and possible exit routes in one's own hand.
- Estimate the whole team's position from public evidence.
- Decide whether to lead, receive a feed, feed a teammate, preserve control, or defend.
- Revise roles as the round develops; do not assign a permanent "leader" by seat.
- Treat a bad hand as lower expected success, not a reason to change the objective or give up.
- Optimize team success, not just personal finishing speed.

A small hand is not necessarily a good hand. Two stranded weak cards may be
worse than eight cards arranged into two controllable legal plays.

Other players' unplayed cards must remain unknown. The AI should infer beliefs
from evidence, not secretly read the simulator's complete state.

## 4. Game rules currently implemented

This section describes this implementation and the user's selected variants.
It should not be treated as a claim that every regional ruleset is identical.

### Players, teams, and direction

- Six players, three standard decks including Jokers.
- 162 physical cards; each player initially receives 27.
- Three small Jokers and three big Jokers.
- Alternating teams by seat parity.
- Team 0: seats 0, 2, 4. Team 1: seats 1, 3, 5.
- Seat names: `0 你`, `1 阿诚`, `2 小满`, `3 老周`, `4 阿岚`, `5 小林`.
- The user explicitly selected counterclockwise rotation:
  `0 -> 5 -> 4 -> 3 -> 2 -> 1 -> 0`.
- Finished players are skipped.
- An unbeaten finished holder passes the next lead to the next active player
  in that direction, regardless of team.
- There is no invented "always give the lead to a teammate" rule.

### Plays and comparisons

- Legal play sizes are 1, 2, 3, or 5 cards.
- A following play must have the same card count as the target.
- Five-card hierarchy, strongest first:
  five-of-a-kind, straight flush, four-plus-one, full house, flush, straight.
- The active level rank is trump and beats ordinary ranks.
- Small and big Jokers are stronger still.
- Jokers act as wildcards in combinations, not in single-card comparisons.
- The strongest legal category is used automatically.
- Within a category, wildcard interpretations use the minimum valid comparison value.
- Mixed all-Joker combinations are valued as small Jokers; pure big-Joker groups are big.
- A2345 and 10JQKA are valid straight endpoints; arbitrary wraparound is not.
- Trump ranks retain their ordinary sequence position inside straights.
- No cross-count "bomb" rule was invented.

### Round ending and progression

- A round ends immediately when one team has all three members out.
- Players still holding cards are trapped.
- Display ordering can append trapped players, but they must not be confused
  with players who actually emptied their hands.
- The head finisher's team wins only if that team also completes all three players.
  Otherwise the round is a draw under the current rules.
- A dealer win raises its level by one.
- A challenger head finisher can cause a dealer change.
- Passing all active non-holder players ends the trick.
- The current holder leads again if still active; otherwise the next active player leads.

### Tribute and return

- Trapped opponents tribute to the previous winning team in the next round.
- The highest card is transferred automatically, evaluated using the new level.
- Receiver selection and donor ordering respect the chosen counterclockwise direction.
- Receivers choose one card to return.
- They may not simply return the same newly received tribute card.
- Human returns are manual; both heuristic and neural return paths exist.
- Tribute temporarily changes hand counts to 26/28; after returns, players have 27 again.
- Actual continued rounds are used in training, not just fake return-phase fixtures.

### Asking

- Following: only on one's turn, with the ability to legally beat the current
  play, may the player ask the previous actual player.
- Free lead: may ask an unfinished teammate.
- At most 10 cards: report the count. Above 10: do not report it.
- Asking does not advance the turn.
- The user selected a capability prerequisite, **not** an additional forced-follow rule.
- Neural asking is an explicit learnable action.
- Repeating the same ask to the same recipient at unchanged public play count is masked.

Asking can publicly signal that the asker has a legal stronger play. It does
not reveal hand composition. A pass also does not prove inability to beat:
players may deliberately preserve control or support a teammate.

An important review question is whether reported counts add information beyond
the implementation's already public remaining counts, and how much strategic
value comes specifically from the public capability signal.

## 5. Web implementation and usability work

The application uses vanilla HTML/CSS/JavaScript and a small Node server.
There are no JavaScript runtime framework dependencies. Playwright is a dev dependency.

Main files:

| File | Responsibility |
| --- | --- |
| `engine.js` | Rules, progression, asking, tribute/return, heuristic AI, persistence validation |
| `engine.test.js` | Rule and tactical regression tests |
| `index.html` | Application structure |
| `style.css` | Desktop/mobile layout and card presentation |
| `app.js` | Human interaction, animations, AI selection, local scoring requests |
| `server.js` | Static routes and bounded local model API |
| `joker-big.svg`, `joker-small.svg` | Original local Joker artwork |
| `browser-check.js` | Browser interaction and privacy/inference checks |
| `benchmark.js` | Paired heuristic comparisons |
| `soak.js` | Consecutive-round stability testing |

Completed usability fixes:

- Replaced the pale decorative layout with a high-contrast table.
- Enlarged card ranks and made player-associated played-card rows readable.
- Preserved each player's last actual play even when they later pass.
- Added chronological actual-card history and current-holder highlighting.
- Clearly marked cards belonging to the previous trick.
- Added pause and selectable play speeds.
- Added mouse/touch range selection, backwards contraction, and deselection.
- Added horizontal hand scrolling controls and edge autoscroll.
- Preserved vertical native touch scrolling.
- Added conspicuous finished-player badges, finishing rank, head-finisher styling,
  team completion progress, and a table-level finishing-order strip.
- Distinguished trapped players from genuine finishers.
- Improved the mobile layout instead of compressing everything around an ellipse.
- Added original big/small Joker pictures with accessible labels.

Production dealing uses `crypto.getRandomValues()` with rejection sampling for
unbiased Fisher-Yates indices. It is cryptographically secure pseudorandomness,
not a physical true-random source. Tests and benchmarks use seeded randomness.

Persistence uses LocalStorage and validates physical cards and public history.
Old directionless saves can complete an existing clockwise trick before
switching safely to the new counterclockwise rule.

## 6. Heuristic AI work before neural training

The earlier AI made exactly the mistakes the user objected to:

- Treated seven/eight remaining cards as if the correct decomposition followed from count.
- Burned both Jokers on weak five-card combinations.
- Refused useful teammate feeds.
- Sometimes overtook teammates without a real tactical reason.

Changes included:

- Bounded hand-partition reasoning.
- Public card counting, unseen-card ceilings, and sampled unseen hands.
- Actual two-play exit detection:
  the first play must preserve recoverable control and the actual remainder must be legal.
- Avoiding low multiple-Joker combinations when not necessary.
- Receiving a teammate's feed when it enables a controlled exit.
- Receiving natural plays along real singles/pairs/triples routes while keeping stronger recaptures.
- Avoiding feeds that strand a weaker group.
- Allowing necessary defensive overrides and whole-hand finishing.
- Recording reasons for teammate overtakes and feed reception.

Concrete regression example:

```text
Teammate plays: 33
Own hand:      44 + 77 + AA
Useful route:  receive with 44, preserve 77 and AA
```

This is not a universal rule that every feed must be accepted.
Control, opponent threats, subsequent lead direction, and the actual remainder matter.

Historical paired heuristic comparison after cooperation fixes:
18 advanced wins, 15 baseline wins, 27 draws across 60 games.
It was not proof of expert strength.

An older 200-game soak run completed 39,639 state/action checks, 232 tributes,
232 returns, 2,122 save/restore checks, and 2,827 legal questions without errors.
Its team outcomes were 57 wins, 64 opposing wins, and 79 draws.
Those results predate later rule/strategy changes and are **stability evidence**,
not directly comparable current strength measurements.

## 7. Local ML environment

- Linux/WSL2.
- Python 3.12 in the project's `.venv`.
- PyTorch `2.11.0+cu128`.
- NumPy and TensorBoard.
- Actual NVIDIA RTX 5090, approximately 32 GB VRAM.
- Actual CUDA computation, training, and resume have been verified.
- Node observed version: v24.13.1.
- Online inference defaults to CPU.
- Web game: `http://localhost:3000`.
- TensorBoard: `http://localhost:6006`.
- Windows-to-WSL localhost model API access was verified.

This is a custom card-game neural policy, **not** Llama, Qwen, or an LLM.
Playing a game repeatedly does not automatically update an LLM.
DouZero/RLCard were investigated as relevant infrastructure/research references,
but existing DouZero weights do not implement these six-player rules.

## 8. Neural learner v1

Files include:

- `policy.js`, `policy.test.js`
- `training/env.js`
- `training/daguai_rl.py`
- `training/train.py`
- `training/evaluate.py`
- `training/test_rl.py`
- `model-host.js`
- `model-check.js`

### Model

Action-conditioned Monte Carlo value network:

```text
observation: 922 floats
action:       71 floats
combined:    993
MLP:         993 -> 256 -> 128 -> 1
parameters:  287,489
```

It regresses chosen-action values toward terminal team reward:
win +1, loss -1, draw 0.

Observation features contain own card types and public information, not other
players' actual unplayed card identities. Legal moves are generated canonically;
equivalent deck copies are represented by canonical physical IDs.

The legal action generator was checked against exhaustive enumeration on
small hands. Neural actions are still validated by the game engine.

The original v1 policy did not have explicit learned asking or the full
recurrent public-history representation introduced later.
The pipeline supports return learning, but the preserved original 80-game
baseline does not declare trained returns. A separate integration run exercised returns.

### Initial measured result

After 80 games and 79 optimizer updates, the model was evaluated against the
heuristic over 200 paired games:

```text
wins:     5
losses:   147
draws:    48
score:    0.145
paired 95% CI: [0.1125, 0.18]
```

It was not promoted to default browser AI.

A later v1 long run reached roughly 5,100 games before replacement by v2.
Its stopped checkpoint was preserved. That later v1 checkpoint has not been
the baseline in the reported v2 comparisons; those use the reproducible 80-game baseline.

## 9. TensorBoard failures and fixes

The user repeatedly saw "No dashboards are active."
This was a real setup/run failure, not something to dismiss as a browser issue.

Initially:

- TensorBoard watched `training/runs/main`.
- Existing event files were elsewhere, from earlier smoke runs.
- The attempted long trainer exited before writing main-run events.
- Its default replay capacity was 50,000 while the resumed checkpoint used 10,000.
- A failed detached execution handle also caused misleading "already running" behavior.

Fixes:

- Successfully relaunched with the saved replay capacity.
- Changed v1 resume defaults so omitted replay capacity inherits the checkpoint.
- Kept explicit incompatible replay-size overrides as clear errors.
- Verified real scalar tags and rendered TensorBoard charts.
- Verified access from Windows as well as Linux.

Later v2 fixes included:

- Explicit save/log cadence overrides may change on resume.
- Core incompatible training settings still fail.
- Cadence changes are printed and persisted.
- Successful resume clears stale failure metadata.
- Actual CUDA recurrent backpropagation was tested, not merely CPU forward passes.
- Fixed the CuDNN requirement that RNN backward must run in training mode.

Training curves and throughput are useful diagnostics.
They do not establish better gameplay.

## 10. Fundamental learner v2

The user explicitly requested fundamental self-play, including learned asking
and better history, rather than just more training of the original network.

Main files:

- `foundation-policy.js`
- `foundation.test.js`
- `training/foundation-env.js`
- `training/foundation.py`
- `training/test_foundation.py`
- `training/foundation-scenarios.js`
- `training/test-bridge.js`

### Model and protocol

```text
encoding version: 2
observation:      922
action features:   81
public event:      71

own/public observation -> 256-dimensional base encoding
full public history   -> 128-dimensional GRU encoding
base + history + action = 465
action-value head:      465 -> 256 -> 128 -> 1
parameters:             465,793
```

The model started with random weights.
There were no human imitation labels or hidden-hand strategic teacher.
It still used terminal team action-value regression, **not PPO**.

Actions became explicit:

```text
play, pass, return, ask
```

The 81 action features are the legacy 71 plus four action-kind flags and six
relative recipient indicators.

The 71 public-history features contain:

- 54 card-type counts.
- Six actor indicators.
- Six recipient indicators.
- Three event kinds: play/pass/ask.
- One reported-answer field.
- One lead flag.

Unknown above-threshold asking answers encode as -1.
Private returned cards are not emitted as public history tokens.

### Information and history

- Each actor sees its own hand and public game information.
- Complete observed current-round play/pass/ask history is maintained.
- Actors and recipients are rotated into the acting seat's coordinates.
- Round history resets when a new round begins.
- Current-round public history persists across tricks and saved reloads.
- Legacy saves can reconstruct public plays/questions, but missing old passes cannot be recreated.
- Replay rows refer to the history prefix before the decision.
- Future public events may be stored in an episode's shared array, but are not actor inputs.
- Opponent private-card identities are excluded from the model payload.

The simulator naturally stores all hands to run the game.
That does not mean they are allowed into actor/critic observations.

### Training

- Terminal TEAM reward +1/-1/0.
- Four gradient minibatches after each completed game.
- Batch size 64, replay capacity 10,000, warmup 256.
- Adam learning rate 0.0001.
- Epsilon exploration 0.15.
- Mixture of current self-play, heuristic opponents, and frozen snapshots.
- Mixed heuristic teammates for some one-team episodes.
- Real continued rounds, including tribute and return.
- Bounded decision replay and public-history storage.
- Safe checkpoint loading with `weights_only=True`.
- Optimizer, RNG, replay, and metadata resume.
- In-flight episodes can be discarded on resume; resume is not bit-identical
  to uninterrupted execution.

The verified initial CUDA run completed 32 games, 116 gradient updates,
1,459 learner asks, 20 learner returns, and 5,885 transitions.
The user-approved long run then resumed toward 100,000 cumulative games.

## 11. Evaluation design and caveats

Current meaningful evaluation:

- Frozen checkpoints, not mutable live learner files.
- Identical seeded deals with learner sides swapped.
- 200 games means 100 paired seeds, two sides each.
- Opponent is the current engine heuristic.
- Two separate teammate modes:
  - `model`: all three teammates use the model.
  - `heuristic`: one rotating model seat plus two heuristic teammates.
- Score is:

```text
score = (wins + 0.5 * draws) / games
```

This is **not** pure win rate.

Confidence intervals use paired-seed bootstrap resampling.
Comparisons use paired score differences, not overlapping-CI guesses.

Validation seed: `1000000000`.
Separate final seed: `2000000000`.
Rules/protocol fingerprints are checked before paired comparison.

Important limitations:

- These assessments start fresh rounds.
- They do **not** measure tribute/return-round strength.
- The heuristic is not a certified human-expert opponent.
- Repeated fixed validation checks are exploratory and can be overfit.
- The separate held-out final evaluation is needed for confirmation.
- Mixed-teammate success cannot be substituted for full-model-team competence.
- Integration tests, low loss, and coverage counts are not strength evidence.
- No automatic browser/default-model promotion takes place.

There are seven reference card-choice scenarios covering:

- Seven/eight-card controlled exits.
- Mixed-Joker preservation.
- Receiving `33` with `44` while retaining `77 + AA`.
- Receiving a weak pair while retaining strict recapture.
- Avoiding a feed that strands a lower pair.
- Avoiding both Jokers on a nonurgent low five-card exchange.

They inspect card choices without optional asking.
Their suggested actions are useful diagnostics, **not proven optimal strategy**.

## 12. Actual strength results

All rows below use 200 games against the heuristic unless stated otherwise.

| Policy / teammate mode | Wins | Losses | Draws | Score | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| Original v1, all-model team | 5 | 147 | 48 | 0.1450 | [0.1125, 0.1800] |
| v2 at 1,000, all-model team | 3 | 151 | 46 | 0.1300 | [0.1000, 0.1625] |
| v2 at 1,000, two heuristic teammates | 33 | 82 | 85 | 0.3775 | [0.3300, 0.4250] |
| v2 at 8,000, all-model team | 2 | 171 | 27 | 0.0775 | [0.0525, 0.1025] |
| v2 at 8,000, two heuristic teammates | 35 | 105 | 60 | 0.3250 | [0.2725, 0.3800] |

### At 1,000 v2 games

| Mode | Difference vs random initial v2 | Difference vs original v1 |
| --- | --- | --- |
| Full model team | +0.1300, CI [0.1000, 0.1625] | -0.0150, CI [-0.0575, 0.0275] |
| Two heuristic teammates | +0.2000, CI [0.1450, 0.2550] | +0.0100, CI [-0.0550, 0.0750] |

Conclusion: better than random initialization, but improvement over v1 was not established.

### At 8,000 v2 games

| Mode | Difference vs v2 at 1,000 | Difference vs original v1 |
| --- | --- | --- |
| Full model team | **-0.0525**, CI **[-0.0900, -0.0150]** | -0.0675, CI [-0.1075, -0.0275] |
| Two heuristic teammates | -0.0525, CI [-0.1250, 0.0200] | -0.0425, CI [-0.1150, 0.0275] |

Conclusion: the full model team measurably regressed on this fixed validation
protocol. Mixed-teammate regression was not statistically established.
Both remained better than random initialization, which is a very weak bar.

### Concrete tactical observations

- The 1,000-game model receives `33` with `44` in the `44 + 77 + AA` fixture.
- The 8,000-game model instead prefers passing; its third choice is `44`.
- The 8,000-game model still burns both Jokers on a low five-card exchange.
- It improved the stranded-lower-pair reference by preferring pass.
- It still does not reliably choose the intended controlled seven/eight-card exits.

These mixed diagnostics reinforce why game outcomes, not one attractive example,
must determine whether training helped.

Evidence files:

```text
training/foundation/progress/validation-000001000-model.json
training/foundation/progress/validation-000001000-heuristic.json
training/foundation/progress/validation-initial-model.json
training/foundation/progress/validation-initial-heuristic.json
training/foundation/progress/validation-v1-model.json
training/foundation/progress/validation-v1-heuristic.json
training/foundation/reports/review/before-upgrade.pt
training/foundation/reports/review/current-model.json
training/foundation/reports/review/current-teammates.json
training/foundation/reports/review/assessment.json
training/foundation/reports/review/tactical-audit.json
```

## 13. Why more Monte Carlo games did not solve the problem

Observed regression is established. Its precise cause has not been isolated
by controlled ablations.

Plausible weaknesses include:

- Every chosen action in a winning round gets the same positive terminal label,
  including bad moves rescued by teammates.
- Every action in a losing round gets a negative label even when it was a useful contribution.
- Replay mixes decisions from different behavior policies without PPO importance ratios.
- The v2 learner can update while other workers are still playing in-flight games.
- Sparse outcomes make long-horizon, imperfect-information team credit assignment difficult.
- Weak/self-changing partners and limited historical diversity can produce unstable coordination.
- The old monitor checked 1,000 games and then 10,000, leaving a long blind interval.
- The newest checkpoint was not necessarily the strongest checkpoint.
- Recurrent memory supplies representational capacity, not guaranteed strategic understanding.

PPO/GAE is intended to improve training structure and credit estimates.
It does not magically make each action's feedback causally correct.

## 14. Primary-source research: OpenAI Five and AlphaGo Zero

### OpenAI Five

Primary paper:
<https://arxiv.org/abs/1912.06680>

Full HTML:
<https://arxiv.org/html/1912.06680v1>

Relevant lessons:

- Shared recurrent policy replicas controlled the five heroes.
- Each replica had its own state/perspective.
- The network produced both policy outputs and a state-value prediction.
- Training used PPO and GAE.
- Historical opponents helped robustness and reduced forgetting.
- Dense shaped rewards mattered for sample efficiency.
- Team spirit blended individual rewards with the team's mean reward.
- The ultimate cooperative objective was team success.
- A sparse-win/loss ablation did learn, but less efficiently.
- "Surgery" preserved useful learned functions across changes.

Crucial correction: OpenAI Five was **not** simply random play with the final
win/loss copied onto every action. It also was not entirely free of designed
reward signals or scripted components.

Its Dota-specific resource, death, building, and combat rewards should not be
blindly translated into "reward shedding cards" or "penalize Jokers."
Those can encourage selfish or strategically wrong card play.

OpenAI Five's compute and model scale were vastly larger than this project.
A single RTX 5090 does not make its convergence timetable transferable.

### AlphaGo Zero

Primary-author explanation:
<https://deepmind.google/blog/alphago-zero-starting-from-scratch/>

Nature paper:
<https://www.nature.com/articles/nature24270>

The author explanation was read; direct Nature access returned an HTTP error.

Relevant lessons:

- Original AlphaGo used human supervision; AlphaGo Zero started from self-play.
- Zero combined policy/value learning with strong search.
- Search improved decisions and produced better training targets.
- It did not merely regress every move toward a game-ending label.

**This project has not implemented AlphaGo-style search-generated policy targets.**
Go is perfect-information; this card game is not.
Directly searching the simulator's actual hidden hands would create an unfair AI.
Any later search should operate over public-consistent beliefs/information sets.

## 15. New shared-team PPO implementation

New files:

```text
training/team_ppo.py
training/test_team_ppo.py
```

Existing integration changes:

- `foundation.py` recognizes `metadata.algorithm = "team-ppo-v1"`.
- PPO checkpoints use their own loader instead of the Monte Carlo model loader.
- Resuming a PPO checkpoint through the Monte Carlo trainer is rejected.
- Existing foundation scoring/inference/evaluation can use the new actor.
- Monitoring supports warm-start baselines and configurable intervals.
- `package.json` adds `train:team-ppo` and `start:team-ppo`.

### Architecture

```text
Own/public observation -> base encoding: 256
Seat-relative public history -> GRU encoding: 128
Shared context: 384

Actor:
  context + action features: 465
  inherited action head -> one logit per legal action

Critic:
  context: 384
  384 -> 128 -> 1
  independent of the selected action

Total parameters: 515,202
```

Both actor and critic use the acting seat's own observation and observed public
history. They do not pool teammates' private hands.

Shared parameters are not the same as shared private information.

The protocol remains encoding version 2: observation 922, action 81, history 71.
There is no unnecessary browser action-schema rewrite.

### Warm start

Source:

```text
training/foundation/checkpoints/milestone-000001000.pt
training/ppo/reports/pre-ppo/reference.pt
```

Source SHA256:

```text
bf653b238fb2b57146df7aa7bec7ba6b8bb2e3d41230c571d0ef398c6e902ce4
```

The source has 1,000 games and 3,988 Monte Carlo optimizer updates.

The base encoder, GRU, and action head are copied exactly.
Before PPO updates, greedy action rankings are preserved.
The new critic's final layer is initialized to zero.

This is transfer from the project's own self-trained weights, not a fresh
random actor and not human imitation.

However, the stochastic training distribution changes:
v2 epsilon-greedy is replaced by categorical sampling of logits at temperature 0.2.
Preserving greedy rankings does **not** mean preserving the old behavior distribution.
This is an important review/ablation point.

### On-policy collection

- Behavior weights stay fixed throughout a complete rollout batch.
- Default batch: eight completed games, four simulator workers.
- Store observation, legal action features, selected action, old log probability,
  old value, actor seat, action kind, and observed history prefix.
- Public episode arrays are shared instead of duplicated for every decision.
- Updates start only after collection is complete.
- Old Monte Carlo replay is not reused as PPO training data.
- Complete batches are consumed once, with the expected multiple PPO epochs.
- Frozen opponents remain frozen during the batch.
- Rollout storage is bounded at 512 MiB.
- Action limits cause explicit failure, not fabricated rewarded truncation.

### Multi-agent credit assignment

GAE is computed separately for each acting player's decision sequence.
It never bootstraps directly from an opponent's value as if that were the same agent.

Each current learner receives its own team's eventual terminal outcome.
Earlier-finished teammates still receive that outcome on their final decision.
When both teams use the current policy, each team receives its corresponding sign.

Conceptually:

```text
delta[t] = shaped_reward[t] + gamma * V[next_same_player_state] - V[current_state]
advantage[t] = delta[t] + gamma * lambda * advantage[t+1]
```

Terminal bootstrap value is zero.

Default gamma is 1 for undiscounted round success; lambda is 0.95.
An action can have a negative early advantage in a winning episode if the
estimated position deteriorates and eventual success arrives much later.
That is an estimate, not perfect counterfactual proof that the action was bad.

### Team progress shaping

No positive reward is added merely for shedding cards, finishing personally,
receiving a specific feed, or preserving/spending particular Jokers.

Instead, the implementation uses a public TEAM potential:

```text
Phi(state) = 0.3 * (finished_teammates - finished_opponents) / 3

shaped_reward = original_reward + gamma * Phi(next_state) - Phi(current_state)
terminal Phi = 0
```

Relative seats 0/2/4 are teammates; 1/3/5 are opponents.
The public finishing-order observation occupies indices 390:396.
Terminal display ranks must not be treated as genuine extra finishers.

Teammate progress benefits all three teammates.
With the correct terminal-zero convention, shaping telescopes and does not
replace the underlying terminal team objective.
Potential scale zero is available for an ablation.

Important interpretation:
the shaped critic estimates shaped return, not directly a pure win probability.
For gamma 1 and terminal Phi zero, the original expected return is obtained by
adding the current Phi back to the shaped value.
Even an unshaped `(V + 1) / 2` estimates expected win-plus-half-draw score,
not necessarily `P(win)` when draws exist.

### PPO update

Defaults:

| Setting | Value |
| --- | ---: |
| Learning rate | 0.0001 |
| Rollout games | 8 |
| Workers | 4 |
| Epochs | 4 |
| Minibatch size | 64 |
| Gamma | 1.0 |
| GAE lambda | 0.95 |
| PPO clip | 0.2 |
| Entropy coefficient | 0.01 |
| Sampling temperature | 0.2 |
| Potential scale | 0.3 |
| Mixed heuristic teammate probability | 0.25 |
| Maximum decisions per episode | 6,000 |

Updates use detached old probabilities/values, clipped PPO objectives,
advantage normalization, a value objective, entropy, finite checks, and gradient clipping.
Variable legal-action batches are encoded without running the GRU once per candidate.
Recurrent gradient forwards use training mode for CuDNN compatibility.

The current implementation should not be represented as having calibrated
role probabilities, explicit hidden-hand beliefs, exact planning, or optimal causal credit.

### Partner/opponent diversity

Target mixture:

- 60% current full self-play.
- 20% uniform frozen historical opponents.
- 20% rule opponents.
- Some one-team episodes use one heuristic teammate.

The historical league is bounded to six policies, including the initial policy.
There is no implemented sophisticated prioritized matchmaking system.

## 16. Latest verification results

### Tests

- Full Python RL test discovery: **60 tests passed**.
- This includes 14 focused team-PPO CPU tests.
- `npm test` passed for the JavaScript game/protocol suite.
- Existing HTTP/browser model checks passed with the new PPO checkpoint.
- Actual CUDA training and optimizer/RNG resume passed.
- Real warm-start monitoring completed in an isolated integration output directory.

Focused tests cover:

- Exact actor transfer and inference compatibility.
- Actor/critic use of own observation and public history.
- Action-independent critic inputs.
- Team potential symmetry and telescoping.
- Per-player GAE boundaries.
- Negative early advantages despite eventual winning outcomes.
- PPO clipping and behavior-policy immutability.
- Public-history prefix safety.
- Asking/return collection.
- Checkpoint lineage, optimizer/RNG resume, and cadence overrides.

### Real GPU run

First 32 PPO games:

```text
cumulative games: 1,032
PPO games:          32
optimizer steps:   372
learner asks:      492
learner returns:    27
```

Then resumed for eight additional games:

```text
status:                    completed
cumulative games:          1,040
source-model games:        1,000
new PPO games:                 40
PPO optimizer steps:          468
learner asks:                 637
learner returns:               31
transitions collected:      7,310
maximum public history:       295
```

Observed throughput was approximately 0.35 games/second in this short run.
It is not a promised long-run rate.

The save cadence was intentionally changed from 8 to 16 on resume.
That operational override was printed and persisted; the core configuration
and seed were inherited.

Verification artifacts:

```text
training/ppo/verification/checkpoints/latest.pt
training/ppo/verification/checkpoints/initial.pt
training/ppo/verification/progress/latest.json
training/runs/verify-team-ppo/
training/runs/verify-team-ppo-eval/
```

The monitor's tiny evaluations used only two games per setting.
They demonstrate transport/reporting compatibility, **not playing strength**.
They are deliberately outside the main TensorBoard run directory.

## 17. Monitoring changes and proposed bounded trial

Old schedule:
first assessment at 1,000 games, then 10,000-game intervals.
This was too slow to expose the observed regression.

New monitor flags:

```text
--initial-kind random|warm-start
--start-games INTEGER
--validation-every INTEGER
```

Warm-start baselines are labelled as transferred pre-PPO policies, not random.
Actual cumulative checkpoint counts are used.
If an exact milestone checkpoint is unavailable, the monitor reports the
available frozen checkpoint and records skipped milestones; it does not invent assessments.

### Proposed next experiment, not yet launched

- Start from the best measured 1,000-game v2 actor.
- Train 2,000 additional PPO games.
- Cumulative target: 3,000, not 100,000.
- Assess after approximately 1,000 and 2,000 new PPO games.
- Use 200 paired games in each teammate mode for exploratory validation.
- Use 400 games per teammate mode with separate held-out seeds for final trial assessment.
- Compare against the pre-PPO policy and preserved v1 baseline.
- Keep the browser default heuristic unless strength evidence supports a deliberate change.
- Do not automatically continue a failed trial for tens of thousands more games.

Example training command, **not executed as production**:

```bash
cd ~/daguai-web
npm run train:team-ppo -- \
  --warm-start training/ppo/reports/pre-ppo/reference.pt \
  --games 2000 \
  --seed 3000000 \
  --workers 4 \
  --rollout-games 8 \
  --epochs 4 \
  --batch-size 64 \
  --device cuda \
  --checkpoint-dir training/ppo/checkpoints \
  --tensorboard-dir training/runs/main/team-ppo-train \
  --save-every 100 \
  --log-every 20
```

Training seed 3,000,000 avoids the earlier low-numbered MC training deals and
the validation/final ranges.

Example monitor command, **not executed as production**:

```bash
cd ~/daguai-web
npm run monitor:foundation -- \
  --checkpoint training/ppo/checkpoints/latest.pt \
  --initial training/ppo/checkpoints/initial.pt \
  --initial-kind warm-start \
  --start-games 1000 \
  --validation-every 1000 \
  --baseline training/progress/baseline-000000080.pt \
  --target-games 3000 \
  --games 200 \
  --final-games 400 \
  --seed 1000000000 \
  --final-seed 2000000000 \
  --output-dir training/ppo/progress \
  --tensorboard-dir training/runs/main/team-ppo-eval \
  --trainer-pid REPLACE_WITH_VERIFIED_PYTHON_PID
```

The PID placeholder must be replaced. Do not copy it literally.
Use separate terminals or an intentionally managed background runner.
Do not accidentally start duplicate trainers or a second server on port 3000.

## 18. Important checkpoints and live-state cautions

| Path | Meaning |
| --- | --- |
| `training/progress/baseline-000000080.pt` | Immutable original v1 assessment baseline |
| `training/progress/v1-stopped.pt` | Later gracefully stopped v1 run |
| `training/foundation/checkpoints/initial.pt` | Original random v2 policy |
| `training/foundation/checkpoints/milestone-000001000.pt` | Best measured v2 snapshot |
| `training/foundation/checkpoints/latest.pt` | Saved long-run v2 checkpoint, not necessarily best |
| `training/foundation/reports/review/before-upgrade.pt` | Immutable reviewed 8,000-game model |
| `training/ppo/reports/pre-ppo/reference.pt` | Immutable PPO warm-start source |
| `training/ppo/verification/checkpoints/latest.pt` | Completed 40-game PPO verification branch |

Other reference SHA256 values:

```text
random initial v2:
7098148b226d1f0d9a7db3692ae6b3d81d9af352e52248434f78c76476468b9f

original v1 baseline:
9176fa9ed44d51fa786b2ab61d63ca65e111ef92450940957c00eace31120729
```

At handoff, the old v2 Python trainer is PID 49671 and was confirmed paused
with process status `T`. Its in-memory progress can exceed the last disk checkpoint.
It was not gracefully stopped after the pause, so that additional state may
still need saving before replacement.

The old CPU monitor remains associated with the v2 checkpoint directory.
It does not assess the new PPO verification directory as its live run.
The old reporting schedule was stopped.

Process IDs are time-sensitive. Verify identity before sending signals;
do not reuse a stale PID or kill unrelated processes on this shared environment.

The live browser inference worker previously loaded the early 32-game v2
checkpoint and retains frozen weights until explicit reload/restart.
Its status can therefore differ from the trainer's current disk checkpoint.
PPO verification used a separate test server; it did not promote the live game.

## 19. What is still missing

1. Evidence that PPO beats the transferred source or the heuristic.
2. A meaningful completed PPO trial with adequate paired evaluations.
3. Explicit, calibrated hidden-hand belief distributions.
4. Explicit role posteriors such as leader/support/controller.
5. Search over public-consistent hidden states or information sets.
6. Search-improved policy targets comparable to AlphaGo Zero.
7. Strength assessment on tribute/return rounds.
8. Human-expert evaluation or a strong independent opponent.
9. A demonstrated large improvement in the user's concrete cooperation complaints.
10. A final decision about replacing the paused long run and restarting scheduled reports.

The recurrent actor/critic can learn implicit inferences, but it is not honest
to say it already understands every hand, teammate role, or winning route.

The critic is trained to evaluate team return conditioned on own/public
information. Its calibration by hand quality has not been separately measured.

## 20. Questions for the reviewing model

Please give a critical technical review, not a generic "train longer" answer.
Prioritize problems that can explain the measured regression or prevent
learning the dynamic team behavior described in section 3.

### Credit assignment and objectives

- Is separate per-player GAE appropriate for variable gaps between turns and early finishers?
- Are final team rewards correctly attached to every current learner, without opponent sign errors?
- Does potential shaping preserve the intended objective at all phase/terminal boundaries?
- Is public finish-count potential useful, or mostly absorbed by the critic without improving learning?
- Does the shaped-value interpretation require additional reporting/calibration safeguards?
- Should objective magnitude reflect more than round win/loss/draw under this particular ruleset?

### Policy transfer and PPO stability

- Is converting old Q scores to logits at temperature 0.2 too exploratory?
- Does the categorical distribution lose useful v2 behavior despite preserved greedy rankings?
- Are four epochs, clip 0.2, entropy 0.01, and learning rate 1e-4 suitable?
- Should approximate KL, explained variance, advantage distributions, or value calibration be logged?
- Is a KL-based early-stop safeguard needed?
- Are rollout boundaries and frozen behavior genuinely on-policy in every branch?

### Hand understanding and partial information

- Does the raw own-hand encoding efficiently represent combinations, flexibility, and control?
- Would structural hand features, a set encoder, or another representation improve sample efficiency
  without injecting heuristic action labels?
- Is recurrent public history enough, or should explicit public-consistent card beliefs be maintained?
- How should passes be used as soft evidence without treating them as hard inability constraints?
- Is one's own previously returned/sent private card sufficiently remembered, given that return
  identities are deliberately excluded from public history?
- Would an asymmetric training critic be legitimate and useful while keeping actor inference private?
  The current implementation deliberately avoids hidden hands in both networks.

### Cooperation and evaluation

- Can the partner/opponent mixture learn stable cooperation rather than exploit weak self-play?
- Does the six-policy historical league preserve enough diversity?
- How should mixed heuristic/human-like teammates be sampled?
- What diagnostics demonstrate adaptive lead/support roles rather than mere card-count shortcuts?
- Should the later preserved v1 model also be evaluated, rather than only its 80-game predecessor?
- How should return-round and human-expert evaluation be added?
- What is the smallest falsifiable experiment that can distinguish credit-assignment,
  representation, exploration, and opponent-distribution problems?

### Search, if recommended

- How should hidden hands be sampled consistently with deck counts and public history?
- How can teammate intent and voluntary passes influence beliefs without unjustified eliminations?
- How can a selective short search improve decisions without omniscient simulator access?
- Should search train a policy, assist only inference, or both?
- What cost is realistic on this single GPU and CPU-bound simulator?

Please separate:

```text
confirmed implementation bug
plausible algorithmic weakness
missing experiment
optional future research
```

And state what evidence would confirm or reject each recommendation.

## 21. Useful commands and source references

Existing validation:

```bash
cd ~/daguai-web
npm test
npm run test:browser
.venv/bin/python -m unittest discover -s training -p 'test*.py' -q
```

Evaluate a frozen checkpoint:

```bash
npm run evaluate:foundation -- \
  --checkpoint PATH_TO_IMMUTABLE_CHECKPOINT \
  --games 200 \
  --device cpu \
  --threads 1 \
  --seed 1000000000 \
  --teammates model \
  --report PATH_TO_NEW_REPORT_JSON
```

Run separately with `--teammates heuristic` for the mixed-team assessment.
Do not confuse it with a full-model-team result.

TensorBoard:

```bash
.venv/bin/tensorboard \
  --logdir training/runs/main \
  --host 127.0.0.1 \
  --port 6006
```

Do not start another TensorBoard instance if the existing one is already serving.

Earlier rules/tactics references, whose variants must not be mixed blindly:

- QQ rules: <https://qqgameplatcdn.qq.com/hallhelp/rulechess/rule159.html>
- Alternative rules: <https://play.cards/zh-hans/rules/wildescape>
- Asking/tactics: <https://wenku.baidu.com/view/af03fdd949fe04a1b0717fd5360cba1aa8118cd5.html>
- General overview: <https://baike.baidu.com/item/大怪路子/7191723>

Primary learning references:

- OpenAI Five paper: <https://arxiv.org/abs/1912.06680>
- Full paper/appendices: <https://arxiv.org/html/1912.06680v1>
- AlphaGo Zero author explanation: <https://deepmind.google/blog/alphago-zero-starting-from-scratch/>
- AlphaGo Zero Nature paper: <https://www.nature.com/articles/nature24270>

## Bottom line

The game, private observation pipeline, local inference, GPU training, and
reporting machinery are real. The old learner's later regression is also real.

The new PPO implementation changes the learning mechanism rather than merely
extending the old run. It has passed software/GPU/inference checks, but its
playing-strength claim remains unproven.

The next useful step is a bounded, properly measured experiment or a concrete
review-driven correction, not another unsupported promise of eventual mastery.
