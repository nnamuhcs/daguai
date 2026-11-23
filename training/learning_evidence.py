"""Learning evidence audit for Daguai continuous PPO.

Two separate questions are answered:
  1. Did the optimizer actually change any model parameter tensors?
  2. Did the new frozen model improve on *the same* deals against the same opponent?

Without --compare-old/--compare-new this is read-only and uses existing reports.
With both, it runs a NEW exploratory matched-deal comparison against the
heuristic. It never promotes a champion and never opens the final test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent / 'ppo' / 'continuous'
DESCRIPTIVE_METRICS = (
    'headFinisher', 'teamCompleted', 'finishPosition', 'cardsLeft',
    'jokersUsed', 'allyOverrides', 'receivesLeavingLegalRemainder',
    'tricksHeld', 'returnDecisions',
)


def checkpoint_delta(older: Path, newer: Path):
    """Compare model state tensors, not file hashes (which change with metadata)."""
    def load(path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(data, dict) or not isinstance(data.get('model'), dict) or
                not data['model'] or not all(torch.is_tensor(value) for value in data['model'].values())):
            raise ValueError(f'not a Daguai model checkpoint: {path}')
        return data

    old, new = load(older), load(newer)
    a, b = old['model'], new['model']
    if set(a) != set(b):
        return {'comparable': False, 'reason': 'different parameter keys; compare within the same architecture',
                'initial': str(older), 'latest': str(newer)}
    if any(a[k].shape != b[k].shape or a[k].dtype != b[k].dtype for k in a):
        return {'comparable': False, 'reason': 'different parameter shapes or dtypes',
                'initial': str(older), 'latest': str(newer)}
    changed_tensors, changed_elements, total_elements, abs_delta = 0, 0, 0, 0.0
    for key in a:
        left, right = a[key], b[key]
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f'non-finite model tensor: {key}')
        total_elements += left.numel()
        different = left.ne(right)
        count = int(different.sum())
        changed_elements += count
        changed_tensors += int(count > 0)
        if count:
            abs_delta += float((left.to(torch.float64) - right.to(torch.float64)).abs().sum())
    m0 = old.get('metadata') or {}
    m1 = new.get('metadata') or {}
    return {
        'comparable': True, 'initial': str(older), 'latest': str(newer),
        'parameterTensors': len(a), 'changedTensors': changed_tensors,
        'changedElements': changed_elements, 'totalElements': total_elements,
        'fractionChanged': changed_elements / total_elements if total_elements else None,
        'sumAbsoluteParameterChange': abs_delta,
        'optimizerStepsBefore': m0.get('updates'), 'optimizerStepsAfter': m1.get('updates'),
        'ppoGamesBefore': m0.get('ppoGames'), 'ppoGamesAfter': m1.get('ppoGames'),
    }


def metric_comparison(new, old):
    # All numbers are descriptive. Fewer Jokers spent is not automatically
    # better, nor is a higher count of teammate overrides automatically worse.
    return {name: {
        'old': float(old['metrics'][name]),
        'new': float(new['metrics'][name]),
        'delta': float(new['metrics'][name]) - float(old['metrics'][name]),
    } for name in DESCRIPTIVE_METRICS
        if name in new.get('metrics', {}) and name in old.get('metrics', {})}


def audit_existing(root: Path):
    root = root.resolve()
    plan_file, progress_file = root / 'plan.json', root / 'progress.json'
    if not (plan_file.exists() and progress_file.exists()):
        raise FileNotFoundError(f'continuous learner plan/progress not found in {root}')
    plan, progress = json.loads(plan_file.read_text()), json.loads(progress_file.read_text())
    ckpt_dir = Path(plan['checkpointDir'])
    if not ckpt_dir.is_absolute():
        ckpt_dir = (Path.cwd() / ckpt_dir).resolve()
    initial, latest = ckpt_dir / 'initial.pt', ckpt_dir / 'latest.pt'
    weights = checkpoint_delta(initial, latest) if initial.exists() and latest.exists() else {
        'comparable': False, 'reason': 'initial.pt or latest.pt missing; cannot prove weight updates',
        'initial': str(initial), 'latest': str(latest),
    }
    recent = []
    for cycle in progress.get('cycles', [])[-5:]:
        row = {
            'cycle': cycle['cycle'], 'games': cycle['toGames'],
            'scores': cycle['candidateScores'], 'promoted': cycle['promoted'],
            'fullPairedDifference': cycle['pairedComparisons']['3'],
            'confirmationPairedDifference': (cycle.get('confirmationComparisons') or {}).get('3'),
        }
        cycle_path = root / 'cycles' / f"{cycle['cycle']:06d}"
        for prefix, section in (('', 'validation'), ('confirmation-', 'confirmation')):
            candidate = cycle_path / f'{prefix}candidate-seats-3-fresh.json'
            incumbent = cycle_path / f'{prefix}incumbent-seats-3-fresh.json'
            if candidate.is_file() and incumbent.is_file():
                c = json.loads(candidate.read_text())
                i = json.loads(incumbent.read_text())
                if (c['protocol'], c['opponentSha256']) != (i['protocol'], i['opponentSha256']):
                    raise ValueError(f'Unmatched reports for cycle {cycle["cycle"]}: {section}')
                row[section] = {
                    'score': c['score'], 'incumbentScore': i['score'], 'games': c['games'],
                    'descriptiveMetrics': metric_comparison(c, i),
                }
        recent.append(row)
    metrics_file = ckpt_dir / 'metrics.jsonl'
    last_metrics = None
    if metrics_file.exists():
        with metrics_file.open(encoding='utf-8') as stream:
            for line in stream:
                if line.endswith('\n') and line.strip():
                    last_metrics = json.loads(line)
    return {
        'scope': 'read-only existing evidence; no new games',
        'status': progress.get('status'), 'phase': progress.get('phase'),
        'trainingWeights': weights,
        'trainerLastBatch': {k: last_metrics.get(k) for k in (
            'games', 'ppoGames', 'updates', 'optimizerSteps', 'gradNorm',
            'approxKL', 'normalizedEntropy', 'explainedVariance', 'clipFraction')}
            if last_metrics else None,
        'champion': progress.get('champion'), 'researchBest': progress.get('researchBest'),
        'recentCycles': recent,
        'limits': [
            'A weight change proves an optimizer update, not better gameplay.',
            'A new candidate is compared with the champion, which may still be heuristic.',
            'Scores from different cycles use different seeds and cannot establish neural-to-neural improvement.',
            'Descriptive metrics do not identify causal reasons for a win or loss.',
            'A confirmed champion is stronger under its specified benchmark, not necessarily against humans.',
        ],
    }


def action_choice_changes(old: Path, new: Path, positions: int, seed: int):
    """Compare decisions on the SAME own-hand/public-history fixtures.

    Different decisions show behavioral change, NOT that either decision is good.
    Private opponent hands remain absent from both policy inputs.
    """
    import numpy as np
    from decision_quality import readable_card
    import foundation as f
    from seed_registry import SPLITS
    import strategic_protocol as strategic
    from tournament import Policy

    if positions < 1 or positions > 2000:
        raise ValueError('decision sample count must be 1..2000')
    low, high = SPLITS['debug']
    if not low <= seed <= seed + positions * 64 - 1 <= high:
        raise ValueError('decision samples must fit registered debug seeds')
    phases = ('fresh', 'mid', 'late', 'returning')
    prior = Policy(str(old), device='cpu', cache=False)
    latest = Policy(str(new), device='cpu', cache=False)
    if prior.version not in (2, 3) or latest.version not in (2, 3):
        raise ValueError('Decision samples require v2/v3 models')
    environment = f.NodeEnvironment(script=f.ROOT / 'strategic-env.js', sizes=strategic.SIZES)
    decisions, changes = [], 0
    from collections import Counter
    old_kinds, new_kinds = Counter(), Counter()
    try:
        for i in range(positions):
            phase = phases[i % len(phases)]
            state = environment.state('sample', seed=seed + i * 64, phase=phase)
            events = state['events']
            old_ids, old_scores = prior.scores(state, events)
            new_ids, new_scores = latest.scores(state, events)
            old_action = state['actions'][old_ids[int(np.argmax(old_scores))]]
            new_action = state['actions'][new_ids[int(np.argmax(new_scores))]]
            def identity(action):
                ids = action.get('ids', [])
                return {'kind': action['kind'], 'cards': [readable_card(c) for c in ids],
                        'ids': ids, 'target': action.get('target')}
            left, right = identity(old_action), identity(new_action)
            old_kinds[left['kind']] += 1
            new_kinds[right['kind']] += 1
            # Physical IDs from the three decks can be interchangeable.
            # Compare canonical action features rather than counting a swap
            # between equivalent copies as new strategic behavior.
            is_changed = (old_action['kind'] != new_action['kind'] or
                          old_action['features'] != new_action['features'])
            changes += is_changed
            if is_changed:
                decisions.append({'seed': seed + i * 64, 'phase': phase, 'player': state['player'],
                                  'legalActions': len(state['actions']), 'old': left, 'new': right})
    finally:
        environment.close()
    return {
        'positions': positions, 'changed': changes, 'fractionChanged': changes / positions,
        'oldActionKindCounts': dict(old_kinds), 'newActionKindCounts': dict(new_kinds),
        'changedDecisions': decisions,
        'interpretation': 'Decision changes are observable, but their correctness requires game or expert evaluation.',
    }


def matched_compare(old: Path, new: Path, games: int, seed: int, rounds='fresh', decision_samples=0):
    from decision_quality import frozen_path
    import foundation as f
    import seed_registry as seeds
    from tournament import evaluate

    old, new = frozen_path(old), frozen_path(new)
    if old == new:
        raise ValueError('old and new are the same checkpoint')
    deal_start, deal_end = seeds.validate_split('debug', seed, games)
    decision_seed = 20_500_000
    if not 0 <= decision_samples <= 2000:
        raise ValueError('decision_samples must be 0..2000')
    if decision_samples and deal_start <= decision_seed + decision_samples * 64 - 1 and deal_end >= decision_seed:
        raise ValueError('Tournament deals must not overlap the debug decision sample range')
    output = {'scope': 'exploratory matched-deal evaluation; not final test or automatic promotion',
              'old': str(old), 'new': str(new), 'oldSha256': f.sha256_file(old),
              'newSha256': f.sha256_file(new), 'gamesPerCheckpoint': games,
              'seed': seed, 'rounds': rounds, 'comparisons': {}}
    reference = evaluate(str(old), opponent_path='heuristic', games=games,
                         seed=seed, model_seats=3, mode='greedy', rounds=rounds)
    candidate = evaluate(str(new), opponent_path='heuristic', games=games,
                         seed=seed, model_seats=3, mode='greedy', rounds=rounds)
    output['comparisons']['3'] = {
        'oldScore': reference['score'], 'newScore': candidate['score'],
        'oldWLD': {k: reference[k] for k in ('wins', 'losses', 'draws')},
        'newWLD': {k: candidate[k] for k in ('wins', 'losses', 'draws')},
        'paired': f.compare_reports(candidate, reference), 'descriptiveMetrics': metric_comparison(candidate, reference),
        'pairedSeedScores': [{
            'seed': o['seed'], 'old': o['scores'], 'new': n['scores'],
        } for o, n in zip(reference['pairedSeeds'], candidate['pairedSeeds'])],
    }
    if decision_samples:
        # The separate audit fixture family must not overlap tournament debug deals.
        output['decisionChanges'] = action_choice_changes(old, new, decision_samples, decision_seed)
    return output


def render(report):
    if 'trainingWeights' in report:
        w = report['trainingWeights']
        print('=== 1. DID WE ACTUALLY CHANGE WEIGHTS? ===')
        if w['comparable']:
            print(f"Changed {w['changedElements']:,}/{w['totalElements']:,} parameter elements "
                  f"in {w['changedTensors']}/{w['parameterTensors']} tensors")
            print(f"Optimizer steps: {w['optimizerStepsBefore']} -> {w['optimizerStepsAfter']}; "
                  f"PPO games: {w['ppoGamesBefore']} -> {w['ppoGamesAfter']}")
            print('ACTUAL PARAMETER UPDATE:', 'YES' if w['changedElements'] else 'NOT DETECTED')
        else:
            print('UNVERIFIED:', w['reason'])
        print('\n=== 2. LAST TRAINING BATCH ===')
        print(json.dumps(report['trainerLastBatch'], indent=2))
        print('\n=== 3. PLAYING STRENGTH ===')
        for cycle in report['recentCycles']:
            full = cycle['fullPairedDifference']
            print(f"cycle={cycle['cycle']} games={cycle['games']} "
                  f"first_test_3v3={cycle['scores']['3']:.3f} paired_diff={full['delta']:+.3f} "
                  f"CI={full['confidence95']} assessment={full['assessment']} "
                  f"promoted={cycle['promoted']}")
            confirmation = cycle.get('confirmation')
            if confirmation:
                paired = cycle['confirmationPairedDifference']
                print(f"  confirmation ({confirmation['games']} games): {confirmation['score']:.3f}"
                      f" vs {confirmation['incumbentScore']:.3f}; 95% CI={paired['confidence95']}")
        if report['recentCycles']:
            print('\n=== 4. WHAT GAME BEHAVIOR CHANGED IN LATEST CYCLE? ===')
            cycle = report['recentCycles'][-1]
            comparison = cycle.get('confirmation', cycle.get('validation', {}))
            for label, numbers in comparison.get('descriptiveMetrics', {}).items():
                print(f"  {label:30s} {numbers['old']:.3f} -> {numbers['new']:.3f} "
                      f"(delta {numbers['delta']:+.3f})")
        print('\nNOTE: Different cycle scores alone do not prove neural-to-neural improvement. '
              'Use --compare-old/--compare-new on matching debug deals.')
        return
    print('=== EXPLORATORY MATCHED-DEAL OLD VS NEW ===')
    for seats, value in report['comparisons'].items():
        d = value['paired']
        print(f"{seats} neural seat(s): {value['oldScore']:.4f} -> {value['newScore']:.4f}; "
              f"delta={d['delta']:+.4f}, 95% CI={d['confidence95']}, {d['assessment']}")
        for name, m in value['descriptiveMetrics'].items():
            print(f"  {name:30s} {m['old']:.3f} -> {m['new']:.3f} ({m['delta']:+.3f})")
    changes = report.get('decisionChanges')
    if changes:
        print('\n=== SAME-HAND / SAME-HISTORY ACTION CHANGES ===')
        print(f"{changes['changed']}/{changes['positions']} decisions changed "
              f"({100*changes['fractionChanged']:.1f}%).")
        print('Old action kinds:', changes['oldActionKindCounts'])
        print('New action kinds:', changes['newActionKindCounts'])
        for row in changes['changedDecisions'][:10]:
            print(f"  seed={row['seed']} phase={row['phase']} "
                  f"old={row['old']['kind']} {row['old']['cards']}  "
                  f"new={row['new']['kind']} {row['new']['cards']}")
    print('NOTE: These game-level metrics are descriptive, not counterfactual proof '
          'of why the action policy changed. Debug deals are exploratory only.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--compare-old', type=Path)
    parser.add_argument('--compare-new', type=Path)
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20_050_000)
    parser.add_argument('--rounds', choices=('fresh', 'continued'), default='fresh')
    parser.add_argument('--decision-samples', type=int, default=100,
                        help='independent identical-position decision changes per matched comparison; 0 disables')
    parser.add_argument('--output', type=Path, help='write immutable JSON comparison report')
    parser.add_argument('--json', action='store_true', help='print full machine-readable report')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if bool(args.compare_old) != bool(args.compare_new):
        parser.error('--compare-old and --compare-new must be supplied together')
    if args.compare_old:
        if args.output is None:
            parser.error('--output is required for matched comparisons')
        if args.output.exists():
            parser.error('report already exists; do not overwrite evidence')
        result = matched_compare(args.compare_old, args.compare_new, args.games, args.seed,
                                 args.rounds, args.decision_samples)
        from decision_quality import save_report
        save_report(result, args.output)
    else:
        if args.output:
            parser.error('--output requires --compare-old/--compare-new')
        result = audit_existing(args.root)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        render(result)
        if args.compare_old:
            print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
