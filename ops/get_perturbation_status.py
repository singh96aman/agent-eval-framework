#!/usr/bin/env python
"""Get perturbation status for an experiment from MongoDB Atlas."""

import os
import sys
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient


def get_perturbation_status(experiment_id: str, verbose: bool = False):
    """Query and display perturbation statistics for an experiment."""
    load_dotenv()

    uri = os.getenv('MONGODB_URI')
    db_name = os.getenv('MONGODB_DATABASE', 'agent_judge_experiment')

    if not uri:
        print("Error: MONGODB_URI not found in environment")
        sys.exit(1)

    client = MongoClient(uri)
    db = client[db_name]

    print(f'=== PERTURBATION SUMMARY: {experiment_id} ===\n')

    # Total count
    total = db.perturbations.count_documents({'experiment_id': experiment_id})
    print(f'Total perturbations: {total}')

    if total == 0:
        print("\nNo perturbations found for this experiment.")
        return

    # By dataset
    print('\nBy dataset:')
    for dataset in ['toolbench', 'gaia', 'swebench']:
        traj_ids = [t['trajectory_id'] for t in db.trajectories.find(
            {'experiment_id': experiment_id, 'benchmark': dataset},
            {'trajectory_id': 1}
        )]
        c = db.perturbations.count_documents({
            'experiment_id': experiment_id,
            'original_trajectory_id': {'$in': traj_ids}
        }) if traj_ids else 0
        print(f'  {dataset}: {c}')

    # By type
    print('\nBy type:')
    for ptype in ['planning', 'tool_selection', 'parameter', 'data_reference']:
        c = db.perturbations.count_documents({
            'experiment_id': experiment_id,
            'perturbation_type': ptype
        })
        print(f'  {ptype}: {c}')

    # By position
    print('\nBy position:')
    for pos in ['early', 'middle', 'late']:
        c = db.perturbations.count_documents({
            'experiment_id': experiment_id,
            'perturbation_position': pos
        })
        print(f'  {pos}: {c}')

    # By quality tier
    print('\nBy quality tier:')
    for tier in ['high', 'medium', 'low', 'invalid']:
        c = db.perturbations.count_documents({
            'experiment_id': experiment_id,
            'quality_tier': tier
        })
        pct = (c / total * 100) if total > 0 else 0
        print(f'  {tier}: {c} ({pct:.1f}%)')

    # Primary selection count
    primary = db.perturbations.count_documents({
        'experiment_id': experiment_id,
        'is_primary_for_experiment': True
    })
    print(f'\nPrimary samples selected: {primary}')

    # Verbose: Type x Position matrix
    if verbose:
        print('\nType x Position matrix:')
        print('                    early   middle   late')
        for ptype in ['planning', 'tool_selection', 'parameter', 'data_reference']:
            row = f'  {ptype:18}'
            for pos in ['early', 'middle', 'late']:
                c = db.perturbations.count_documents({
                    'experiment_id': experiment_id,
                    'perturbation_type': ptype,
                    'perturbation_position': pos
                })
                row += f'{c:8}'
            print(row)


def main():
    parser = argparse.ArgumentParser(description='Get perturbation status for an experiment')
    parser.add_argument(
        'experiment_id',
        nargs='?',
        default='final_exp_perturbation_quality',
        help='Experiment ID (default: final_exp_perturbation_quality)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed type x position matrix'
    )

    args = parser.parse_args()
    get_perturbation_status(args.experiment_id, args.verbose)


if __name__ == '__main__':
    main()
