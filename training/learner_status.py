"""Read-only scoreboard for the continuous training controller."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'ppo' / 'continuous')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    progress = args.root / 'progress.json'
    if not progress.exists():
        parser.error(f'no continuous learner state: {progress}')
    data = json.loads(progress.read_text())
    pending = data.get('pending')
    training = None
    if pending and data['phase'] == 'training':
        plan = json.loads((args.root / 'plan.json').read_text())
        path = Path(plan['checkpointDir']) / 'progress.json'
        if path.is_file():
            checkpoint = json.loads(path.read_text())
            training = {key: checkpoint[key] for key in ('games', 'updates')}
            data['liveTraining'] = training
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print(f"status={data['status']} phase={data['phase']}")
    print(f"completed_cycles={len(data['cycles'])} new_ppo_games={data['totalNewPpoGames']}")
    if pending:
        cycle = pending['cycle']
        start, target = pending['startGames'], pending['targetGames']
        if data['phase'] == 'training':
            done = training['games'] - start if training is not None else 0
            print(f"cycle={cycle} training_games={done}/{target - start}")
        else:
            print(f"pending_cycle={cycle} target_cumulative_games={target}")
    evaluation = data.get('liveEvaluation')
    if evaluation and data['status'] == 'running':
        print(f"evaluating={evaluation['label']} ({evaluation['rounds']}, 3v3) "
              f"games={evaluation['completedGames']}/{evaluation['totalGames']}")
    champion = data['champion']
    print(f"verified_champion={champion['checkpoint']}")
    research = data.get('researchBest')
    if research:
        print(f"best_experimental_neural=cycle-{research['cycle']} "
              f"3v3={research['scoreFull']:.4f}")
    for stage in data['cycles'][-10:]:
        print(f"cycle={stage['cycle']:4d} games={stage['toGames']:7d} "
              f"3v3={stage['candidateScores']['3']:.4f} "
              f"incumbent={stage['incumbentScores']['3']:.4f} "
              f"paired_delta={stage['pairedComparisons']['3']['delta']:+.4f} "
              f"CI_full={stage['pairedComparisons']['3']['confidence95']} "
              f"promoted={stage['promoted']}")
    if data.get('lastError'):
        print(f"ERROR: {data['lastError']}")


if __name__ == '__main__':
    main()
