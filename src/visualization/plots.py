"""
Visualization and reporting tools for CCG analysis.

This module generates:
- Heatmaps: CCG by perturbation type × position
- Scatter plots: TCS vs JPS
- Distribution plots: CCG values
- Bar charts: Condition comparisons
- Summary tables: CSV and Excel exports
- Human-LLM alignment analysis
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    output_dir: str
    experiment_id: str
    judge_names: Optional[List[str]] = None
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = 'seaborn-v0_8-darkgrid'
    save_formats: List[str] = None

    def __post_init__(self):
        if self.save_formats is None:
            self.save_formats = ['png', 'pdf']


class ResultsVisualizer:
    """
    Generate comprehensive visualizations from CCG analysis results.

    Usage:
        visualizer = ResultsVisualizer(config)
        visualizer.load_results(results_dir)
        visualizer.generate_all()
    """

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.results_data = {}  # {judge_name: dataframe}
        self.summary_data = {}  # {judge_name: summary_dict}

        # Set up plotting style
        plt.style.use(self.config.style)
        sns.set_palette("husl")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_results(self, results_dir: str) -> None:
        """
        Load CCG results from CSV files and summary JSON files.

        Args:
            results_dir: Directory containing ccg_results_*.csv and ccg_summary_*.json
        """
        results_path = Path(results_dir)

        # Load CSV files
        for csv_file in results_path.glob('ccg_results_*.csv'):
            judge_name = csv_file.stem.replace('ccg_results_', '')

            # Check if file is empty
            if csv_file.stat().st_size <= 1:
                print(f"Warning: Empty file {csv_file.name}, skipping judge {judge_name}")
                continue

            try:
                # Load CSV
                df = pd.read_csv(csv_file)

                # Skip if empty
                if df.empty or len(df) == 0:
                    print(f"Warning: No data in {csv_file.name}, skipping judge {judge_name}")
                    continue

                self.results_data[judge_name] = df
                print(f"Loaded {len(df)} results for judge: {judge_name}")
            except pd.errors.EmptyDataError:
                print(f"Warning: Cannot parse {csv_file.name}, skipping judge {judge_name}")
                continue

        # Load JSON summaries
        for json_file in results_path.glob('ccg_summary_*.json'):
            judge_name = json_file.stem.replace('ccg_summary_', '')

            with open(json_file, 'r') as f:
                summary = json.load(f)

            self.summary_data[judge_name] = summary
            print(f"Loaded summary for judge: {judge_name}")

    def generate_all(self) -> Dict[str, List[str]]:
        """
        Generate all visualizations and tables.

        Returns:
            Dictionary mapping visualization type to list of generated file paths
        """
        generated_files = {
            'heatmaps': [],
            'scatter_plots': [],
            'distributions': [],
            'bar_charts': [],
            'tables': [],
            'alignment': []
        }

        print(f"\nGenerating visualizations for {len(self.results_data)} judge(s)...")

        for judge_name, df in self.results_data.items():
            print(f"\n=== Judge: {judge_name} ===")

            # Generate heatmaps
            heatmap_files = self._generate_heatmaps(judge_name, df)
            generated_files['heatmaps'].extend(heatmap_files)

            # Generate scatter plots
            scatter_files = self._generate_scatter_plots(judge_name, df)
            generated_files['scatter_plots'].extend(scatter_files)

            # Generate distribution plots
            dist_files = self._generate_distributions(judge_name, df)
            generated_files['distributions'].extend(dist_files)

            # Generate bar charts
            bar_files = self._generate_bar_charts(judge_name, df)
            generated_files['bar_charts'].extend(bar_files)

            # Generate tables
            table_files = self._generate_tables(judge_name, df)
            generated_files['tables'].extend(table_files)

        # Generate cross-judge comparisons if multiple judges
        if len(self.results_data) > 1:
            comparison_files = self._generate_judge_comparisons()
            generated_files['alignment'].extend(comparison_files)

        # Generate summary report
        report_file = self._generate_summary_report(generated_files)

        print(f"\n✅ All visualizations generated in: {self.config.output_dir}")
        print(f"📊 Summary report: {report_file}")

        return generated_files

    def _generate_heatmaps(self, judge_name: str, df: pd.DataFrame) -> List[str]:
        """Generate CCG heatmaps by type × position."""
        files = []

        # Filter out rows with missing CCG values
        df_valid = df[df['ccg'].notna()].copy()

        if df_valid.empty:
            print(f"  ⚠️  No valid CCG data for heatmap")
            return files

        # Create pivot table: rows=type, cols=position, values=mean CCG
        pivot = df_valid.pivot_table(
            values='ccg',
            index='perturbation_type',
            columns='perturbation_position',
            aggfunc='mean'
        )

        # Ensure column order: early, middle, late
        position_order = ['early', 'middle', 'late']
        pivot = pivot.reindex(columns=[c for c in position_order if c in pivot.columns])

        # Generate heatmap
        fig, ax = plt.subplots(figsize=self.config.figsize)

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',  # Red = high CCG (bad), Green = low CCG (good)
            center=0,
            cbar_kws={'label': 'Criticality-Calibration Gap (CCG)'},
            linewidths=1,
            ax=ax
        )

        ax.set_title(
            f'CCG Heatmap by Perturbation Type and Position\n'
            f'Judge: {judge_name}',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Trajectory Position', fontsize=12)
        ax.set_ylabel('Perturbation Type', fontsize=12)

        # Add sample counts
        count_pivot = df_valid.pivot_table(
            values='ccg',
            index='perturbation_type',
            columns='perturbation_position',
            aggfunc='count'
        )
        count_pivot = count_pivot.reindex(columns=[c for c in position_order if c in count_pivot.columns])

        # Annotate with counts
        for i, row_label in enumerate(pivot.index):
            for j, col_label in enumerate(pivot.columns):
                count = count_pivot.loc[row_label, col_label] if not pd.isna(count_pivot.loc[row_label, col_label]) else 0
                ax.text(
                    j + 0.5, i + 0.85,
                    f'n={int(count)}',
                    ha='center', va='center',
                    fontsize=8, color='gray'
                )

        plt.tight_layout()

        # Save in multiple formats
        for fmt in self.config.save_formats:
            filepath = os.path.join(
                self.config.output_dir,
                f'heatmap_ccg_{judge_name}.{fmt}'
            )
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            files.append(filepath)

        plt.close(fig)
        print(f"  ✓ Heatmap: {len(files)} format(s)")

        return files

    def _generate_scatter_plots(self, judge_name: str, df: pd.DataFrame) -> List[str]:
        """Generate TCS vs JPS scatter plots."""
        files = []

        # Filter valid data
        df_valid = df[df['tcs'].notna() & df['jps'].notna()].copy()

        if df_valid.empty:
            print(f"  ⚠️  No valid TCS/JPS data for scatter plot")
            return files

        # Create scatter plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Color by perturbation type
        types = df_valid['perturbation_type'].unique()
        colors = sns.color_palette("husl", len(types))
        type_to_color = dict(zip(types, colors))

        for ptype in types:
            subset = df_valid[df_valid['perturbation_type'] == ptype]
            ax.scatter(
                subset['tcs'],
                subset['jps'],
                label=ptype,
                alpha=0.6,
                s=100,
                color=type_to_color[ptype]
            )

        # Add diagonal (perfect calibration: JPS = TCS)
        min_val = min(df_valid['tcs'].min(), df_valid['jps'].min())
        max_val = max(df_valid['tcs'].max(), df_valid['jps'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', alpha=0.3, linewidth=2, label='Perfect Calibration')

        ax.set_xlabel('True Criticality Score (TCS)', fontsize=12)
        ax.set_ylabel('Judge Penalty Score (JPS)', fontsize=12)
        ax.set_title(
            f'Judge Calibration: TCS vs JPS\n'
            f'Judge: {judge_name}',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        for fmt in self.config.save_formats:
            filepath = os.path.join(
                self.config.output_dir,
                f'scatter_tcs_jps_{judge_name}.{fmt}'
            )
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            files.append(filepath)

        plt.close(fig)
        print(f"  ✓ Scatter plot: {len(files)} format(s)")

        return files

    def _generate_distributions(self, judge_name: str, df: pd.DataFrame) -> List[str]:
        """Generate CCG distribution plots."""
        files = []

        df_valid = df[df['ccg'].notna()].copy()

        if df_valid.empty:
            print(f"  ⚠️  No valid CCG data for distribution")
            return files

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Overall CCG distribution
        ax = axes[0, 0]
        ax.hist(df_valid['ccg'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='CCG = 0 (Perfect)')
        ax.set_xlabel('CCG', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Overall CCG Distribution', fontsize=11, fontweight='bold')
        ax.legend()

        # 2. CCG by perturbation type
        ax = axes[0, 1]
        df_valid.boxplot(column='ccg', by='perturbation_type', ax=ax)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Perturbation Type', fontsize=10)
        ax.set_ylabel('CCG', fontsize=10)
        ax.set_title('CCG by Perturbation Type', fontsize=11, fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

        # 3. CCG by position
        ax = axes[1, 0]
        df_valid.boxplot(column='ccg', by='perturbation_position', ax=ax)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('CCG', fontsize=10)
        ax.set_title('CCG by Position', fontsize=11, fontweight='bold')

        # 4. TCS vs JPS distributions
        ax = axes[1, 1]
        ax.hist(df_valid['tcs'], bins=15, alpha=0.5, label='TCS', edgecolor='black')
        ax.hist(df_valid['jps'], bins=15, alpha=0.5, label='JPS', edgecolor='black')
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('TCS vs JPS Distributions', fontsize=11, fontweight='bold')
        ax.legend()

        fig.suptitle(f'CCG Analysis Distributions - Judge: {judge_name}',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        for fmt in self.config.save_formats:
            filepath = os.path.join(
                self.config.output_dir,
                f'distributions_{judge_name}.{fmt}'
            )
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            files.append(filepath)

        plt.close(fig)
        print(f"  ✓ Distribution plots: {len(files)} format(s)")

        return files

    def _generate_bar_charts(self, judge_name: str, df: pd.DataFrame) -> List[str]:
        """Generate bar charts comparing conditions."""
        files = []

        df_valid = df[df['ccg'].notna()].copy()

        if df_valid.empty:
            print(f"  ⚠️  No valid data for bar charts")
            return files

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Mean CCG by type
        ax = axes[0]
        type_means = df_valid.groupby('perturbation_type')['ccg'].agg(['mean', 'std', 'count'])
        type_means = type_means.sort_values('mean')

        bars = ax.bar(range(len(type_means)), type_means['mean'],
                      yerr=type_means['std'], capsize=5, alpha=0.7)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xticks(range(len(type_means)))
        ax.set_xticklabels(type_means.index, rotation=45, ha='right')
        ax.set_ylabel('Mean CCG', fontsize=11)
        ax.set_title('Mean CCG by Perturbation Type', fontsize=12, fontweight='bold')

        # Add sample counts on bars
        for i, (idx, row) in enumerate(type_means.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.1,
                   f'n={int(row["count"])}',
                   ha='center', fontsize=9)

        # 2. Mean CCG by position
        ax = axes[1]
        pos_means = df_valid.groupby('perturbation_position')['ccg'].agg(['mean', 'std', 'count'])
        position_order = ['early', 'middle', 'late']
        pos_means = pos_means.reindex([p for p in position_order if p in pos_means.index])

        bars = ax.bar(range(len(pos_means)), pos_means['mean'],
                      yerr=pos_means['std'], capsize=5, alpha=0.7)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xticks(range(len(pos_means)))
        ax.set_xticklabels(pos_means.index, rotation=0)
        ax.set_ylabel('Mean CCG', fontsize=11)
        ax.set_title('Mean CCG by Position', fontsize=12, fontweight='bold')

        # Add sample counts
        for i, (idx, row) in enumerate(pos_means.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.1,
                   f'n={int(row["count"])}',
                   ha='center', fontsize=9)

        fig.suptitle(f'CCG Comparisons - Judge: {judge_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        for fmt in self.config.save_formats:
            filepath = os.path.join(
                self.config.output_dir,
                f'bar_charts_{judge_name}.{fmt}'
            )
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            files.append(filepath)

        plt.close(fig)
        print(f"  ✓ Bar charts: {len(files)} format(s)")

        return files

    def _generate_tables(self, judge_name: str, df: pd.DataFrame) -> List[str]:
        """Generate summary tables in CSV and Excel formats."""
        files = []

        df_valid = df[df['ccg'].notna()].copy()

        if df_valid.empty:
            print(f"  ⚠️  No valid data for tables")
            return files

        # 1. Detailed results table (already in CSV, just copy with better formatting)
        detailed_file = os.path.join(
            self.config.output_dir,
            f'detailed_results_{judge_name}.csv'
        )
        df.to_csv(detailed_file, index=False)
        files.append(detailed_file)

        # 2. Summary statistics table
        summary_stats = []

        # Overall stats
        summary_stats.append({
            'Condition': 'Overall',
            'Count': len(df_valid),
            'Mean CCG': df_valid['ccg'].mean(),
            'Std CCG': df_valid['ccg'].std(),
            'Median CCG': df_valid['ccg'].median(),
            'Mean TCS': df_valid['tcs'].mean(),
            'Mean JPS': df_valid['jps'].mean()
        })

        # By type
        for ptype in df_valid['perturbation_type'].unique():
            subset = df_valid[df_valid['perturbation_type'] == ptype]
            summary_stats.append({
                'Condition': f'Type: {ptype}',
                'Count': len(subset),
                'Mean CCG': subset['ccg'].mean(),
                'Std CCG': subset['ccg'].std(),
                'Median CCG': subset['ccg'].median(),
                'Mean TCS': subset['tcs'].mean(),
                'Mean JPS': subset['jps'].mean()
            })

        # By position
        for pos in ['early', 'middle', 'late']:
            if pos in df_valid['perturbation_position'].values:
                subset = df_valid[df_valid['perturbation_position'] == pos]
                summary_stats.append({
                    'Condition': f'Position: {pos}',
                    'Count': len(subset),
                    'Mean CCG': subset['ccg'].mean(),
                    'Std CCG': subset['ccg'].std(),
                    'Median CCG': subset['ccg'].median(),
                    'Mean TCS': subset['tcs'].mean(),
                    'Mean JPS': subset['jps'].mean()
                })

        # By type × position
        for ptype in df_valid['perturbation_type'].unique():
            for pos in ['early', 'middle', 'late']:
                subset = df_valid[
                    (df_valid['perturbation_type'] == ptype) &
                    (df_valid['perturbation_position'] == pos)
                ]
                if len(subset) > 0:
                    summary_stats.append({
                        'Condition': f'{ptype} × {pos}',
                        'Count': len(subset),
                        'Mean CCG': subset['ccg'].mean(),
                        'Std CCG': subset['ccg'].std(),
                        'Median CCG': subset['ccg'].median(),
                        'Mean TCS': subset['tcs'].mean(),
                        'Mean JPS': subset['jps'].mean()
                    })

        summary_df = pd.DataFrame(summary_stats)

        # Save as CSV
        summary_csv = os.path.join(
            self.config.output_dir,
            f'summary_statistics_{judge_name}.csv'
        )
        summary_df.to_csv(summary_csv, index=False, float_format='%.4f')
        files.append(summary_csv)

        # Save as Excel with formatting
        try:
            summary_excel = os.path.join(
                self.config.output_dir,
                f'summary_statistics_{judge_name}.xlsx'
            )
            summary_df.to_excel(summary_excel, index=False, sheet_name='Summary')
            files.append(summary_excel)
        except ImportError:
            print(f"  ⚠️  openpyxl not installed, skipping Excel export")

        print(f"  ✓ Tables: {len(files)} file(s)")

        return files

    def _generate_judge_comparisons(self) -> List[str]:
        """Generate cross-judge comparison visualizations."""
        files = []

        if len(self.results_data) < 2:
            return files

        print(f"\n=== Cross-Judge Comparisons ===")

        # Combine all judge data
        combined_data = []
        for judge_name, df in self.results_data.items():
            df_copy = df.copy()
            df_copy['judge'] = judge_name
            combined_data.append(df_copy)

        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df = combined_df[combined_df['ccg'].notna()]

        if combined_df.empty:
            print(f"  ⚠️  No valid data for cross-judge comparison")
            return files

        # 1. Side-by-side CCG comparison
        fig, ax = plt.subplots(figsize=self.config.figsize)

        combined_df.boxplot(column='ccg', by='judge', ax=ax)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Judge', fontsize=12)
        ax.set_ylabel('CCG', fontsize=12)
        ax.set_title('CCG Distribution Across Judges', fontsize=14, fontweight='bold')

        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        for fmt in self.config.save_formats:
            filepath = os.path.join(
                self.config.output_dir,
                f'comparison_judges_ccg.{fmt}'
            )
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            files.append(filepath)

        plt.close(fig)

        # 2. Human-LLM alignment: correlation analysis
        if len(self.results_data) == 2:
            judge_names = list(self.results_data.keys())
            df1 = self.results_data[judge_names[0]]
            df2 = self.results_data[judge_names[1]]

            # Merge on perturbation_id
            merged = df1.merge(
                df2,
                on='perturbation_id',
                suffixes=('_1', '_2')
            )

            if len(merged) > 0:
                # Correlation plot
                fig, ax = plt.subplots(figsize=self.config.figsize)

                ax.scatter(merged['ccg_1'], merged['ccg_2'], alpha=0.6, s=100)

                # Compute correlation
                corr, p_value = stats.pearsonr(
                    merged['ccg_1'].dropna(),
                    merged['ccg_2'].dropna()
                )

                # Add diagonal
                min_val = min(merged['ccg_1'].min(), merged['ccg_2'].min())
                max_val = max(merged['ccg_1'].max(), merged['ccg_2'].max())
                ax.plot([min_val, max_val], [min_val, max_val],
                       'k--', alpha=0.3, linewidth=2)

                ax.set_xlabel(f'{judge_names[0]} CCG', fontsize=12)
                ax.set_ylabel(f'{judge_names[1]} CCG', fontsize=12)
                ax.set_title(
                    f'Judge Agreement: {judge_names[0]} vs {judge_names[1]}\n'
                    f'Pearson r = {corr:.3f}, p = {p_value:.4f}',
                    fontsize=14,
                    fontweight='bold'
                )
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                for fmt in self.config.save_formats:
                    filepath = os.path.join(
                        self.config.output_dir,
                        f'alignment_judge_correlation.{fmt}'
                    )
                    fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    files.append(filepath)

                plt.close(fig)

        print(f"  ✓ Cross-judge comparisons: {len(files)} file(s)")

        return files

    def _generate_summary_report(self, generated_files: Dict[str, List[str]]) -> str:
        """Generate a markdown summary report."""
        report_path = os.path.join(self.config.output_dir, 'VISUALIZATION_REPORT.md')

        with open(report_path, 'w') as f:
            f.write(f"# Visualization Report\n\n")
            f.write(f"**Experiment ID:** {self.config.experiment_id}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"## Judges Analyzed\n\n")
            for judge_name in self.results_data.keys():
                summary = self.summary_data.get(judge_name, {})
                total = summary.get('total_results', 0)
                f.write(f"- **{judge_name}**: {total} results\n")
            f.write("\n")

            f.write(f"## Summary Statistics\n\n")
            for judge_name, summary in self.summary_data.items():
                f.write(f"### {judge_name}\n\n")
                overall = summary.get('summary', {})

                # Format values safely
                def fmt_val(val, fmt='.4f'):
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        return f"{val:{fmt}}"
                    return 'N/A'

                f.write(f"- Mean CCG: {fmt_val(overall.get('ccg_mean'))}\n")
                f.write(f"- Std CCG: {fmt_val(overall.get('ccg_std'))}\n")
                f.write(f"- Median CCG: {fmt_val(overall.get('ccg_median'))}\n")
                f.write(f"- Mean TCS: {fmt_val(overall.get('tcs_mean'), '.2f')}\n")
                f.write(f"- Mean JPS: {fmt_val(overall.get('jps_mean'), '.2f')}\n\n")

                # Statistical tests
                stats_tests = summary.get('statistical_tests', {})
                anova_type = stats_tests.get('anova_type', {})
                anova_pos = stats_tests.get('anova_position', {})

                f.write(f"**Statistical Tests:**\n")
                f.write(f"- ANOVA (by type): F={fmt_val(anova_type.get('f_stat'))}, p={fmt_val(anova_type.get('p_value'))}\n")
                f.write(f"- ANOVA (by position): F={fmt_val(anova_pos.get('f_stat'))}, p={fmt_val(anova_pos.get('p_value'))}\n\n")

            f.write(f"## Generated Files\n\n")
            for viz_type, file_list in generated_files.items():
                if file_list:
                    f.write(f"### {viz_type.replace('_', ' ').title()}\n\n")
                    for filepath in file_list:
                        filename = os.path.basename(filepath)
                        f.write(f"- `{filename}`\n")
                    f.write("\n")

            f.write(f"## File Locations\n\n")
            f.write(f"All files saved to: `{self.config.output_dir}`\n\n")

        return report_path


def generate_visualizations(
    results_dir: str,
    output_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Convenience function to generate all visualizations.

    Args:
        results_dir: Directory containing CCG results
        output_dir: Output directory (defaults to results_dir/visualizations)
        experiment_id: Experiment ID (auto-detected if None)
        **kwargs: Additional arguments for VisualizationConfig

    Returns:
        Dictionary of generated files by type

    Example:
        >>> files = generate_visualizations(
        ...     results_dir='results/exp_poc_toolbench_20260402',
        ...     figsize=(10, 6)
        ... )
    """
    # Auto-detect experiment ID from directory name
    if experiment_id is None:
        experiment_id = os.path.basename(results_dir.rstrip('/'))

    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'visualizations')

    # Create config
    config = VisualizationConfig(
        output_dir=output_dir,
        experiment_id=experiment_id,
        **kwargs
    )

    # Generate visualizations
    visualizer = ResultsVisualizer(config)
    visualizer.load_results(results_dir)
    generated_files = visualizer.generate_all()

    return generated_files
