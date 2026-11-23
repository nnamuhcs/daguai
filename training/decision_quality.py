"""Counterfactual decision-quality audit (DIAGNOSTIC ONLY, not PPO rewards).

Compare a frozen neural policy against paired, sampled legal alternatives. Hidden
hands are sampled by the Node bridge using ONLY actor-visible information, then
fed to the game engine for simulated continuations. The evaluator does not use
actual hidden hands to select or score options. The simulation opponent is the
existing heuristic, so rankings are *not* a ground-truth optimality oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import foundation as f
from seed_registry import SPLITS
import strategic_protocol as strategic
from tournament import Policy


def candidate_indices(state, scores, top_k, *, previous=None, budget=12):
    """Always compare policy argmax; add likely actions and a legal pass if present."""
    if not 2 <= top_k <= budget <= 12:
        raise ValueError('invalid candidate budget')
    if not len(scores) or len(scores) != len(state['actions']) or not np.isfinite(scores).all():
        raise ValueError('scores must cover a non-empty set of finite legal actions')
    ranked = np.argsort(-np.asarray(scores), kind='stable').tolist()
    passing = next((i for i, a in enumerate(state['actions']) if a['kind'] == 'pass'), None)
    required = list(dict.fromkeys(i for i in (ranked[0], passing, previous) if i is not None))
    choices = ranked[:top_k]
    if passing is not None and passing not in choices:
        choices[-1] = passing
    if previous is not None and previous not in choices:
        if len(choices) == budget:
            choices.pop(next(i for i in reversed(range(len(choices))) if choices[i] not in required))
        choices.append(previous)
    return choices


def interpret_audit(result):
    if result['status'] != 'ok':
        return {'status': result['status'], 'reason': result.get('error', 'Not enough consistent belief worlds')}
    ranked = sorted(result['estimates'], key=lambda row: row['score'], reverse=True)
    chosen = result['estimates'][0]
    challenger = ranked[0]
    interval = challenger['paired95Approx']
    return {
        'status': 'ok',
        'chosenIndex': chosen['index'],
        'chosenEstimatedScore': chosen['score'],
        'bestSampledIndex': challenger['index'],
        'bestSampledScore': challenger['score'],
        'estimatedRegret': max(0., challenger['score'] - chosen['score']),
        'paired95Approx': interval,
        'searchDisagrees': challenger['index'] != chosen['index'],
        'confidentDisagreement': challenger['index'] != chosen['index'] and interval[0] > 0,
        'note': 'Approximate normal CI; correlated paired worlds; exploratory, not a proof of optimality.',
    }


def readable_card(card_id):
    n = card_id % 54
    if n >= 52:
        return ('小怪', '大怪')[n - 52]
    value = 2 + n % 13
    rank = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(value, str(value))
    return ('♠', '♥', '♣', '♦')[n // 13] + rank


def frozen_path(checkpoint):
    path = Path(checkpoint).resolve(strict=True)
    if not path.is_file() or path.name == 'latest.pt':
        raise ValueError(f'Use a frozen checkpoint, not {path}')
    return path


def save_report(report, output):
    output = Path(output)
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(content)


def audit(checkpoint, positions, worlds, top_k, seed, phases, output, compare_old=None):
    checkpoint = frozen_path(checkpoint)
    compare_old = frozen_path(compare_old) if compare_old else None
    if Path(output).exists():
        raise FileExistsError(f'Audit already exists: {output}')
    if not 1 <= positions <= 1000 or not 4 <= worlds <= 32:
        raise ValueError('positions must be 1..1000 and worlds 4..32')
    budget = min(12, 256 // worlds)
    if not 2 <= top_k <= budget:
        raise ValueError(f'top_k must be 2..{budget} with {worlds} worlds')
    low, high = SPLITS['debug']
    if not low <= seed <= seed + positions * 64 - 1 <= high:
        raise ValueError('decision samples must fit registered debug seeds')
    policy = Policy(str(checkpoint), mode='greedy', device='cpu', cache=False)
    previous = Policy(str(compare_old), mode='greedy', device='cpu', cache=False) if compare_old else None
    if policy.version not in (2, 3) or (previous and previous.version not in (2, 3)):
        raise ValueError('decision-quality audit requires a PPO/foundation v2 or strategic v3 model')
    env = f.NodeEnvironment(script=f.ROOT / 'counterfactual-env.js', sizes=strategic.SIZES)
    rows = []
    try:
        for i in range(positions):
            phase = phases[i % len(phases)]
            pos_seed = seed + 64 * i
            state = env.state('sample', seed=pos_seed, phase=phase)
            history = list(state['events'])
            indices, scores = policy.scores(state, history)
            if indices != list(range(len(state['actions']))):
                raise ValueError('scoring did not cover all legal actions')
            prior_choice = None
            if previous:
                prior_indices, prior_scores = previous.scores(state, history)
                if prior_indices != indices or len(prior_scores) != len(indices) or not np.isfinite(prior_scores).all():
                    raise ValueError('old and new policies require aligned finite legal action scores')
                prior_choice = int(np.argmax(prior_scores))
            selected = candidate_indices(state, scores, top_k, previous=prior_choice, budget=budget)
            # Independent, reproducible belief seed not derived from true hidden hands.
            result = env.request('counterfactual_audit', indices=selected, worlds=worlds,
                                 attempts=min(4096, 32 * worlds), seed=pos_seed + 1)
            insight = interpret_audit(result)
            if previous and insight['status'] == 'ok':
                estimates_by_id = {entry['index']: entry for entry in result['estimates']}
                old_est = estimates_by_id[prior_choice]
                chosen_score = insight['chosenEstimatedScore']
                insight['previousChoiceIndex'] = prior_choice
                insight['previousChoiceEstimatedScore'] = old_est['score']
                insight['newVsOldPairedScoreDelta'] = chosen_score - old_est['score']
                # CIs are relative to the new choice; reverse signs to compare new minus old.
                lo, hi = old_est['paired95Approx']
                insight['newVsOldPaired95Approx'] = [-hi, -lo]
                insight['newChoiceBetterWithApprox95CI'] = lo < 0 and hi < 0
            action_map = {index: {'kind': state['actions'][index]['kind'],
                                  'cards': [readable_card(card) for card in state['actions'][index]['ids']],
                                  'target': state['actions'][index].get('target')}
                          for index in selected}
            rows.append({'positionSeed': pos_seed, 'phase': phase, 'player': state['player'],
                         'legalActions': len(state['actions']), 'selectedActions': action_map,
                         'result': result, 'interpretation': insight})
    finally:
        env.close()
    evaluated = [row for row in rows if row['interpretation']['status'] == 'ok']
    analysis = {
        'schemaVersion': 1, 'checkpoint': str(Path(checkpoint).resolve()),
        'checkpointSha256': policy.digest, 'previousCheckpoint': str(Path(compare_old).resolve()) if previous else None,
        'previousSha256': previous.digest if previous else None, 'positionsRequested': positions,
        'positionsEvaluated': len(evaluated), 'sampledWorldsRequested': worlds,
        'topK': top_k, 'seed': seed,
        'meanEstimatedRegret': float(np.mean([r['interpretation']['estimatedRegret'] for r in evaluated])) if evaluated else None,
        'searchDisagreementRate': float(np.mean([r['interpretation']['searchDisagrees'] for r in evaluated])) if evaluated else None,
        'confidentDisagreements': sum(r['interpretation']['confidentDisagreement'] for r in evaluated),
        'meanNewVsOldPairedScoreDelta': float(np.mean([r['interpretation']['newVsOldPairedScoreDelta'] for r in evaluated])) if evaluated and previous else None,
        'meanOldChoiceEstimatedRegret': float(np.mean([max(e['score'] for e in row['result']['estimates']) - row['interpretation']['previousChoiceEstimatedScore'] for row in evaluated])) if evaluated and previous else None,
        'rolloutPolicy': 'heuristic-all-seats-after-first-action',
        'auditOnly': True, 'usedAsTrainingReward': False,
        'limitations': ['Belief sampler is approximate and may reject evidence-constrained states.',
                        'Heuristic continuations may misrank moves compared with expert play.',
                        'Only a bounded candidate set is searched.',
                        'Normal paired intervals are exploratory; cannot establish optimality.',
                        'No change to PPO gradients, training reward or champion promotion.'],
        'positions': rows,
    }
    save_report(analysis, output)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--compare-old', help='Optional frozen prior checkpoint; compare old/new moves on identical sampled worlds')
    parser.add_argument('--positions', type=int, default=12)
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20_850_000)
    parser.add_argument('--phases', default='fresh,mid,late')
    parser.add_argument('--output', default='training/ppo/continuous/decision-quality/latest.json')
    args = parser.parse_args()
    phases = args.phases.split(',')
    if not phases or any(p not in ('fresh', 'mid', 'late', 'returning') for p in phases):
        parser.error('phases must be comma-separated fresh,mid,late,returning')
    torch.set_num_threads(1)
    result = audit(args.checkpoint, args.positions, args.worlds, args.top_k,
                   args.seed, phases, args.output, compare_old=args.compare_old)
    print(json.dumps({k: v for k, v in result.items() if k != 'positions'}, indent=2))
    print(f'Detailed decisions saved: {args.output}')


if __name__ == '__main__':
    main()
