"""
Command-line interface for generating visualizations.

Usage:
    python -m src.visualization.cli --results-dir results/exp_poc_toolbench_20260402
    python -m src.visualization.cli --experiment-id exp_poc_toolbench_20260402
"""

import argparse
import sys
from pathlib import Path

from .plots import generate_visualizations


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations from CCG analysis results'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory containing CCG results (ccg_results_*.csv files)'
    )

    parser.add_argument(
        '--experiment-id',
        type=str,
        help='Experiment ID (will look in results/<experiment_id>/)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for visualizations (default: <results-dir>/visualizations)'
    )

    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[12, 8],
        help='Figure size as width height (default: 12 8)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output images (default: 300)'
    )

    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=['png', 'pdf'],
        help='Output formats (default: png pdf)'
    )

    args = parser.parse_args()

    # Determine results directory
    if args.experiment_id:
        results_dir = f'results/{args.experiment_id}'
    elif args.results_dir:
        results_dir = args.results_dir
    else:
        print("❌ Error: Must specify either --results-dir or --experiment-id")
        sys.exit(1)

    # Check if results directory exists
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Check for result files
    result_files = list(results_path.glob('ccg_results_*.csv'))
    if not result_files:
        print(f"❌ Error: No ccg_results_*.csv files found in {results_dir}")
        sys.exit(1)

    print(f"📊 Generating visualizations from: {results_dir}")
    print(f"   Found {len(result_files)} result file(s)")

    try:
        # Generate visualizations
        generated_files = generate_visualizations(
            results_dir=results_dir,
            output_dir=args.output_dir,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            save_formats=args.formats
        )

        # Print summary
        total_files = sum(len(files) for files in generated_files.values())
        print(f"\n✅ Successfully generated {total_files} visualization file(s)")

        # Print by category
        for viz_type, file_list in generated_files.items():
            if file_list:
                print(f"   {viz_type}: {len(file_list)} file(s)")

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
