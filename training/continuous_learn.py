"""Continuous, evidence-gated Daguai self-play training.

Runs on the user's machine (not in a ChatGPT/background service).  A new PPO
checkpoint is trained every cycle, frozen, and evaluated on fresh paired deals.
The training policy is allowed to change; the published champion is not.  A
champion is promoted only on statistically supported full-team improvement.
This is a *selection policy*,
not a promise that self-play will monotonically improve or reach human level.

Usage:
    python training/continuous_learn.py --from-scratch --device cuda
    python training/continuous_learn.py --warm-start training/foundation/checkpoints/milestone-000001000.pt --device cuda
    python training/continuous_learn.py --prepared-start training/ppo/reports/pre-ppo/distilled.pt --device cuda
    python training/continuous_learn.py --ablate-from PATH_TO_COMPLETED_V2_PPO --strategic --device cuda
    python training/continuous_learn.py --device cuda   # resumes an existing run

Ctrl+C stops after restoring the last valid transaction.  --cycles N is a
bounded smoke mode; --cycles 0 runs until interrupted.  This script never
opens the final evaluation split or silently replaces the browser model.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path

import torch

import foundation as f
import seed_registry as registry
import team_ppo as p
from tournament import evaluate

ROOT = f.ROOT / 'ppo' / 'continuous'
SCHEMA = 2
TEAM_SEATS = 3


def stage_seed(base, games, cycle):
    if cycle < 1 or games < 2 or games % 2:
        raise ValueError('invalid stage/evaluation game counts')
    # Never reuse a validation deal for a different checkpoint-selection step.
    return base + (cycle - 1) * (games // 2 + 37)


def decision_from_comparison(full, minimum_full_improvement=0.0):
    if not math.isfinite(minimum_full_improvement) or minimum_full_improvement < 0:
        raise ValueError('invalid promotion thresholds')
    lower_full = full['confidence95'][0]
    if not math.isfinite(lower_full):
        return False, 'non-finite paired confidence interval'
    if lower_full <= minimum_full_improvement:
        return False, 'full-model-team improvement not established by paired CI'
    return True, 'passed paired full-team gain'


def _checkpoint_identity(path):
    if path == 'heuristic':
        return 'heuristic'
    return f.sha256_file(path)


def validate_team_report(report, identity, games, seed, rounds, rules):
    from check_release import validate_report
    validate_report(report)
    if (report['checkpointSha256'] != identity
            or report['opponentSha256'] != 'heuristic'
            or report['games'] != games or report['seed'] != seed
            or report['protocol']['modelSeats'] != TEAM_SEATS
            or report['protocol']['rounds'] != rounds
            or report['protocol']['rules'] != rules
            or report['policy']['sampling'] != 'greedy'):
        raise ValueError('tournament mismatches the frozen full-team protocol')
    for pair in report['pairedSeeds']:
        if ([sorted(match['learnerSeats']) for match in pair['matches']] != [[0, 2, 4], [1, 3, 5]]
                or pair['learnerSeats'] != [match['learnerSeats'] for match in pair['matches']]):
            raise ValueError('tournament did not swap complete teams')
    return report


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('continuous learner already running in this directory') from error
        yield


class Controller:
    def __init__(self, args, trainer=None, evaluator=None):
        self.args = args
        self.root = args.root.resolve()
        self.checkpoints = args.checkpoint_dir.resolve()
        self.trainer = trainer or p.train
        self.evaluator = evaluator or evaluate
        self.plan_path = self.root / 'plan.json'
        self.progress_path = self.root / 'progress.json'
        self.plan = None
        self.progress = None

    def _configuration(self):
        a = self.args
        return {
            'schemaVersion': SCHEMA,
            'checkpointDir': str(self.checkpoints),
            'stageGames': a.stage_games,
            'evaluationGames': a.evaluation_games,
            'evaluationSeed': a.evaluation_seed,
            'confirmationGames': a.confirmation_games,
            'confirmationSeed': a.confirmation_seed,
            'trainingSeed': a.train_seed,
            'workers': a.workers,
            'rolloutGames': a.rollout_games,
            'modelSeats': TEAM_SEATS,
            'rules': f.rules_fingerprint(),
            'minimumFullGain': a.min_full_gain,
            'continuedEvery': a.continued_every,
        }

    def _validate_paths(self):
        if (self.root == self.checkpoints or self.checkpoints in self.root.parents
                or self.root == f.ROOT or self.checkpoints == f.ROOT
                or not self.root.is_relative_to(f.ROOT) or not self.checkpoints.is_relative_to(f.ROOT)):
            raise ValueError('root/checkpoints must be distinct paths inside the training project')
        for legacy in (f.ROOT / 'foundation', f.ROOT / 'checkpoints'):
            if any(path.is_relative_to(legacy.resolve()) for path in (self.root, self.checkpoints)):
                raise ValueError('continuous runs cannot write legacy model directories')

    def _publish(self, **patch):
        self.progress.update(patch)
        f.atomic_json_save(self.progress, self.progress_path)

    def _load_or_initialize(self):
        a = self.args
        self._validate_paths()
        configuration = self._configuration()
        if self.plan_path.exists() != self.progress_path.exists():
            raise RuntimeError('partial controller initialization; restore original files, do not reset evidence')
        if self.plan_path.exists():
            self.plan = json.loads(self.plan_path.read_text())
            self.progress = json.loads(self.progress_path.read_text())
            if any(self.plan.get(key) != value for key, value in configuration.items()):
                raise ValueError('existing plan changed: use original run configuration or a NEW root')
            if any((a.from_scratch, a.warm_start, a.prepared_start, a.ablate_from, a.upgrade_to_strategic)):
                raise ValueError('resuming an existing run must omit initialization arguments')
            return
        if self.root.exists() and any(child.name != '.controller.lock' and child != self.checkpoints
                                     for child in self.root.iterdir()):
            # A previous broken or foreign run must not be overwritten.
            raise ValueError('run root is occupied but lacks its original plan')
        latest = self.checkpoints / 'latest.pt'
        if latest.exists():
            if any((a.from_scratch, a.warm_start, a.prepared_start, a.ablate_from, a.upgrade_to_strategic)):
                raise ValueError('occupied checkpoint directory cannot be reinitialized')
            model, data = p.load_checkpoint(latest)
            initial = self.checkpoints / 'initial.pt'
            if ('optimizer' not in data or 'rng' not in data or not initial.is_file()
                    or f.sha256_file(initial) != data['metadata'].get('initialSha256')):
                raise ValueError('resume requires the original PPO optimizer, RNG and matching initial.pt')
            del model
        elif not any((a.from_scratch, a.warm_start, a.prepared_start, a.ablate_from, a.upgrade_to_strategic)):
            raise ValueError('no PPO latest.pt: provide a new-run initialization flag')
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan = {
            **configuration, 'createdAt': f.now(),
            'initialization': ('resume-existing' if latest.exists() else
                               'prepared-start' if a.prepared_start else
                               'warm-start' if a.warm_start else
                               'strategic-upgrade' if a.upgrade_to_strategic else
                               'strategic-ablation' if a.ablate_from else 'from-scratch'),
            'initialSource': (str(a.prepared_start or a.warm_start or a.ablate_from or a.upgrade_to_strategic)
                              if (a.prepared_start or a.warm_start or a.ablate_from or a.upgrade_to_strategic) else None),
            'initialSourceSha256': (f.sha256_file(a.prepared_start or a.warm_start or a.ablate_from or a.upgrade_to_strategic)
                                    if (a.prepared_start or a.warm_start or a.ablate_from or a.upgrade_to_strategic) else None),
            'limits': 'validation-only, paired, immutable, no final/test seed use, no default deployment',
            'championRule': 'positive validation gain and independently confirmed paired full-team CI gain',
        }
        self.progress = {
            'schemaVersion': SCHEMA, 'status': 'initialized', 'phase': 'ready',
            'cycles': [], 'pending': None, 'champion': {'checkpoint': 'heuristic', 'sha256': 'heuristic'},
            'researchBest': None, 'totalNewPpoGames': 0, 'lastError': None,
        }
        f.atomic_json_save(self.plan, self.plan_path)
        self._publish()

    def _evaluate(self, checkpoint, cycle_dir, label, games, seed, rounds='fresh', split=None):
        report_path = cycle_dir / f'{label}-seats-{TEAM_SEATS}-{rounds}.json'
        identity = _checkpoint_identity(checkpoint)
        validate = lambda report: validate_team_report(report, identity, games, seed, rounds, self.plan['rules'])
        if report_path.is_file():
            return validate(json.loads(report_path.read_text()))
        reservation = registry.record_evaluation(
            split or ('continued' if rounds == 'continued' else 'validation'), seed, games, identity,
            f'continuous:{cycle_dir.name}:{label}:{TEAM_SEATS}:{rounds}')

        def update_evaluation(position):
            finished = position['completedGames']
            if finished % 10 == 0 or finished == games:
                self._publish(liveEvaluation={'label': label, 'rounds': rounds, **position})

        try:
            update_evaluation({'completedGames': 0, 'totalGames': games, 'completedPairs': 0})
            torch.set_num_threads(1)
            report = validate(self.evaluator(checkpoint, opponent_path='heuristic', games=games,
                                             seed=seed, model_seats=TEAM_SEATS, mode='greedy', rounds=rounds,
                                             progress_callback=update_evaluation))
            f.atomic_json_save(report, report_path)
        except BaseException:
            registry.finish_evaluation(reservation, 'failed')
            raise
        registry.finish_evaluation(reservation, 'completed', report_path.resolve())
        return report

    def _train_to_target(self, pending):
        latest = self.checkpoints / 'latest.pt'
        options = ['train', '--checkpoint-dir', str(self.checkpoints),
                   '--tensorboard-dir', str(self.root / 'tensorboard'),
                   '--replay-dir', str(self.root / 'replays'),
                   '--target-games', str(pending['targetGames']),
                   '--device', self.args.device,
                   '--save-every', str(self.args.stage_games),
                   '--log-every', str(min(self.args.stage_games, 8))]
        if latest.exists():
            model, checkpoint = p.load_checkpoint(latest)
            del model
            status = checkpoint['metadata'].get('status')
            if status == 'failed':
                raise RuntimeError('PPO last batch failed; inspect its error before explicit recovery')
            if checkpoint['metadata']['games'] > pending['targetGames']:
                raise ValueError('training went beyond the recorded milestone; reject ambiguous lineage')
            options += ['--resume', str(latest)]
        elif self.plan['initialization'] == 'prepared-start':
            options += ['--prepared-start', str(self.plan['initialSource'])]
        elif self.plan['initialization'] == 'warm-start':
            options += ['--warm-start', str(self.plan['initialSource'])]
        elif self.plan['initialization'] == 'strategic-ablation':
            options += ['--ablate-from', str(self.plan['initialSource']), '--strategic']
        elif self.plan['initialization'] == 'strategic-upgrade':
            options += ['--upgrade-to-strategic', str(self.plan['initialSource'])]
        elif self.plan['initialization'] == 'from-scratch':
            options += ['--from-scratch']
        else:
            raise RuntimeError('cannot resume missing checkpoint')
        if not latest.exists():
            source = self.plan.get('initialSource')
            if source is not None and f.sha256_file(source) != self.plan['initialSourceSha256']:
                raise RuntimeError('initial source checkpoint changed after plan was sealed')
            options += ['--seed', str(self.args.train_seed), '--workers', str(self.args.workers),
                        '--rollout-games', str(self.args.rollout_games)]
        if latest.exists() and checkpoint['metadata']['games'] == pending['targetGames']:
            if checkpoint['metadata']['status'] != 'completed':
                raise RuntimeError('milestone reached but training did not commit completed status')
            return checkpoint['metadata']
        result = self.trainer(p.arguments(options))
        if result['status'] != 'completed' or result['games'] != pending['targetGames']:
            raise KeyboardInterrupt('trainer interrupted before completing the current milestone')
        return result

    def _cycle(self):
        a = self.args
        pending = self.progress['pending']
        if pending is None:
            idx = len(self.progress['cycles']) + 1
            latest = self.checkpoints / 'latest.pt'
            if latest.exists():
                start = p.load_checkpoint(latest)[1]['metadata']['games']
            elif self.plan['initialization'] == 'warm-start':
                start = f.load_checkpoint(self.plan['initialSource'], 'cpu')[1]['metadata']['games']
            elif self.plan['initialization'] == 'prepared-start':
                start = p.load_checkpoint(self.plan['initialSource'])[1]['metadata']['games']
            elif self.plan['initialization'] == 'strategic-upgrade':
                start = p.load_checkpoint(self.plan['initialSource'])[1]['metadata']['sourceGames']
            else:
                start = 0
            if self.progress['cycles']:
                previous = self.progress['cycles'][-1]
                if (not latest.exists() or start != previous['toGames']
                        or f.sha256_file(latest) != previous['checkpointSha256']):
                    raise RuntimeError('live training checkpoint changed outside the controller')
                if p.load_checkpoint(latest)[1]['metadata']['status'] != 'completed':
                    raise RuntimeError('previous completed milestone no longer has a completed checkpoint')
            pending = {'cycle': idx, 'startGames': start, 'targetGames': start + a.stage_games,
                       'seed': stage_seed(a.evaluation_seed, a.evaluation_games, idx),
                       'phase': 'training'}
            registry.validate_split('validation', pending['seed'], a.evaluation_games)
            self._publish(pending=pending, status='running', phase='training', liveEvaluation=None)
        idx = pending['cycle']
        if idx != len(self.progress['cycles']) + 1:
            raise RuntimeError('non-sequential pending milestone; repair evidence before continuing')
        cycle_dir = self.root / 'cycles' / f'{idx:06d}'
        cycle_dir.mkdir(parents=True, exist_ok=True)
        frozen = cycle_dir / 'candidate.pt'
        if not frozen.exists():
            result = self._train_to_target(pending)
            if f.rules_fingerprint() != self.plan['rules']:
                raise RuntimeError('game/protocol source changed during training')
            f.freeze_checkpoint(self.checkpoints / 'latest.pt', frozen)
        else:
            result = p.load_checkpoint(frozen)[1]['metadata']
        if result['games'] != pending['targetGames']:
            raise RuntimeError('frozen milestone does not match the planned game count')
        pending['phase'] = 'evaluating'
        self._publish(pending=pending, phase='evaluating')
        incumbent = self.progress['champion']
        champion = incumbent['checkpoint']
        if _checkpoint_identity(champion) != incumbent['sha256']:
            raise RuntimeError('incumbent champion digest changed')
        current = self._evaluate(frozen, cycle_dir, 'candidate', a.evaluation_games, pending['seed'])
        baseline = self._evaluate(champion, cycle_dir, 'incumbent', a.evaluation_games, pending['seed'])
        compared = f.compare_reports(current, baseline)
        reason = 'validation point estimate did not exceed the required full-team gain'
        confirmation = None
        passed = False
        if compared['delta'] > a.min_full_gain:
            confirmation_seed = stage_seed(a.confirmation_seed, a.confirmation_games, idx)
            registry.validate_split('confirmation', confirmation_seed, a.confirmation_games)
            candidate_confirm = self._evaluate(frozen, cycle_dir, 'confirmation-candidate',
                                               a.confirmation_games, confirmation_seed, split='confirmation')
            incumbent_confirm = self._evaluate(champion, cycle_dir, 'confirmation-incumbent',
                                               a.confirmation_games, confirmation_seed, split='confirmation')
            confirmation = f.compare_reports(candidate_confirm, incumbent_confirm)
            passed, confirmation_reason = decision_from_comparison(confirmation, a.min_full_gain)
            reason = ('confirmed: ' if passed else 'confirmation rejected: ') + confirmation_reason
        # Evaluation-gated promotion is local to this learner's champion registry;
        # the browser model is NEVER implicitly changed/reloaded.
        next_champion = incumbent
        if passed:
            champion_path = self.root / 'champions' / f'champion-cycle-{idx:06d}.pt'
            digest = f.freeze_checkpoint(frozen, champion_path)
            next_champion = {
                'checkpoint': str(champion_path), 'sha256': digest, 'cycle': idx,
                'scoreFull': current['score'],
                'encodingVersion': result['encodingVersion'],
                'reference': _checkpoint_identity(champion),
            }
        # Exploratory best *learned* policy, even when below the rule baseline:
        # do not confuse this point-estimate tracker with a deployment certificate.
        best = self.progress['researchBest']
        if best is None or current['score'] > best['scoreFull'] + 0.005:
            best_path = self.root / 'research' / f'best-cycle-{idx:06d}.pt'
            f.freeze_checkpoint(frozen, best_path)
            best = {
                'checkpoint': str(best_path), 'sha256': f.sha256_file(best_path), 'cycle': idx,
                'scoreFull': current['score'],
                'caveat': 'exploratory point-estimate selection, NOT proven improvement',
            }
        continued = None
        if a.continued_every and idx % a.continued_every == 0:
            continued_seed = registry.SPLITS['continued'][0] + (idx - 1) * (a.evaluation_games // 2 + 37)
            registry.validate_split('continued', continued_seed, a.evaluation_games)
            continued = self._evaluate(frozen, cycle_dir, 'candidate',
                                       a.evaluation_games, continued_seed, rounds='continued')
            # Diagnostic only. Promotion requires a separate continued-round gate
            # before real-world deployment, which this runner does not authorize.
        summary = {
            'cycle': idx, 'checkpoint': str(frozen), 'checkpointSha256': f.sha256_file(frozen),
            'fromGames': pending['startGames'], 'toGames': result['games'],
            'ppoGames': result['ppoGames'], 'updates': result['updates'],
            'seed': pending['seed'], 'gameCountPerMatchup': a.evaluation_games,
            'candidateScores': {'3': current['score']},
            'incumbentScores': {'3': baseline['score']},
            'pairedComparisons': {'3': compared},
            'confirmationComparisons': {'3': confirmation} if confirmation is not None else None,
            'promoted': passed, 'decision': reason,
            'currentChampion': next_champion['checkpoint'],
            'continuedScore': continued['score'] if continued is not None else None,
            'finalHoldoutOpened': False, 'browserDeployed': False,
            'trainingStrengthClaim': False,
        }
        f.atomic_json_save(summary, cycle_dir / 'summary.json')
        self.progress['cycles'].append(summary)
        self._publish(pending=None, status='running', phase='cycle-completed', liveEvaluation=None,
                      champion=next_champion, researchBest=best, lastError=None,
                      totalNewPpoGames=self.progress['totalNewPpoGames'] + a.stage_games)
        print(f"cycle={idx} PPO_games={result['ppoGames']} "
              f"3v3={current['score']:.3f} "
              f"promoted={passed} reason={reason}", flush=True)

    def run(self):
        self._validate_paths()
        with exclusive_lock(self.root / '.controller.lock'), \
                exclusive_lock(self.checkpoints / '.continuous-controller.lock'):
            self._load_or_initialize()
            count = 0
            try:
                while not self.args.cycles or count < self.args.cycles:
                    self._cycle()
                    count += 1
            except KeyboardInterrupt:
                self._publish(status='interrupted', phase='resume-with-same-command')
                print('interrupted safely; re-run without initialization flags to resume', flush=True)
                return
            except BaseException as error:
                self._publish(status='failed', lastError=f'{type(error).__name__}: {error}')
                raise
            self._publish(status='idle', phase='bounded-run-complete')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--checkpoint-dir', type=Path, help='defaults to the checkpoints subdirectory of --root')
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--from-scratch', action='store_true')
    sources.add_argument('--warm-start', type=Path)
    sources.add_argument('--prepared-start', type=Path)
    sources.add_argument('--ablate-from', type=Path)
    sources.add_argument('--upgrade-to-strategic', type=Path)
    parser.add_argument('--strategic', action='store_true',
                        help='required with --ablate-from to initialize a v3 residual policy')
    parser.add_argument('--stage-games', type=int, default=128)
    parser.add_argument('--rollout-games', type=int, default=8)
    parser.add_argument('--evaluation-games', type=int, default=200)
    parser.add_argument('--evaluation-seed', type=int, default=1_100_300_000)
    parser.add_argument('--confirmation-games', type=int, default=1000,
                        help='disjoint confirmation deals for promising candidate; no final holdout')
    parser.add_argument('--confirmation-seed', type=int, default=1_200_300_000)
    parser.add_argument('--continued-every', type=int, default=4,
                        help='0 disables continued-round diagnostics; still no auto release')
    parser.add_argument('--min-full-gain', type=float, default=0.0)
    parser.add_argument('--cycles', type=int, default=0, help='0 = run continuously until Ctrl+C')
    parser.add_argument('--train-seed', type=int, default=3_100_000)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.root / 'checkpoints'
    if (not 1 <= args.rollout_games <= 64
            or args.stage_games < 8 or args.stage_games % args.rollout_games
            or args.evaluation_games < 2 or args.evaluation_games % 2
            or args.confirmation_games < 200 or args.confirmation_games % 2
            or not 1 <= args.workers <= 32 or args.cycles < 0
            or args.continued_every < 0):
        parser.error('invalid bounded training/evaluation parameters')
    if args.strategic and not args.ablate_from:
        parser.error('--strategic requires --ablate-from')
    if args.ablate_from and not args.strategic:
        parser.error('--ablate-from must specify --strategic for a v3 learner')
    if not 0 <= args.train_seed <= 0xFFFFFFFF:
        parser.error('training seed must fit uint32')
    if not math.isfinite(args.min_full_gain) or args.min_full_gain < 0:
        parser.error('invalid statistical promotion thresholds')
    try:
        registry.validate_split('validation', args.evaluation_seed, args.evaluation_games)
        registry.validate_split('confirmation', args.confirmation_seed, args.confirmation_games)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    Controller(parse_args(argv)).run()


if __name__ == '__main__':
    main()
