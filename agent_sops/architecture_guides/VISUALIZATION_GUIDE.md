# Visualization System Guide

Quick reference for generating and customizing CCG analysis visualizations.

---

## Quick Start

### Generate All Visualizations

**Option 1: Via experiment runner**
```bash
python main.py --config poc_experiment_toolbench --runner visualize
```

**Option 2: Via CLI tool**
```bash
python -m src.visualization.cli --experiment-id exp_poc_toolbench_20260402
```

**Output location:** `results/<experiment_id>/visualizations/`

---

## What Gets Generated

### Figures (8 files per judge)

1. **Heatmap:** CCG by type × position (PNG + PDF)
2. **Scatter plot:** TCS vs JPS (PNG + PDF)
3. **Distributions:** 4-panel distribution analysis (PNG + PDF)
4. **Bar charts:** CCG comparisons (PNG + PDF)

### Tables (3 files per judge)

5. **Detailed results:** All CCG scores (CSV)
6. **Summary statistics:** Aggregated by condition (CSV)
7. **Summary statistics:** Same as #6 with formatting (XLSX)

### Reports (1 file)

8. **Visualization report:** Markdown summary (MD)

**Total:** 12 files per judge

---

## Customization

### Figure Size

```bash
python -m src.visualization.cli \
  --experiment-id exp_poc_toolbench_20260402 \
  --figsize 14 10
```

Default: 12x8 inches

### Resolution (DPI)

```bash
python -m src.visualization.cli \
  --experiment-id exp_poc_toolbench_20260402 \
  --dpi 600
```

- **150 DPI:** Quick preview
- **300 DPI:** Standard quality (default)
- **600 DPI:** Publication quality

### Output Formats

```bash
python -m src.visualization.cli \
  --experiment-id exp_poc_toolbench_20260402 \
  --formats png pdf svg
```

Supported: `png`, `pdf`, `svg`, `jpg`

### Custom Output Directory

```bash
python -m src.visualization.cli \
  --experiment-id exp_poc_toolbench_20260402 \
  --output-dir my_custom_dir
```

Default: `<results-dir>/visualizations/`

---

## Configuration File

Add visualization settings to your experiment config JSON:

```json
{
  "experiment": { ... },
  "execution": { ... },
  "visualization": {
    "figsize": [12, 8],
    "dpi": 300,
    "formats": ["png", "pdf"]
  }
}
```

Then run:
```bash
python main.py --config your_config --runner visualize
```

---

## Python API

### Basic Usage

```python
from src.visualization import generate_visualizations

files = generate_visualizations(
    results_dir='results/exp_poc_toolbench_20260402'
)

# Returns: dict of generated files by type
# {'heatmaps': [...], 'scatter_plots': [...], ...}
```

### Advanced Usage

```python
from src.visualization import ResultsVisualizer, VisualizationConfig

# Create config
config = VisualizationConfig(
    output_dir='my_viz',
    experiment_id='exp_poc_toolbench_20260402',
    figsize=(14, 10),
    dpi=600,
    save_formats=['png', 'pdf', 'svg']
)

# Create visualizer
viz = ResultsVisualizer(config)

# Load results
viz.load_results('results/exp_poc_toolbench_20260402')

# Generate all visualizations
files = viz.generate_all()

print(f"Generated {sum(len(f) for f in files.values())} files")
```

---

## Interpreting Visualizations

### Heatmap: CCG by Type × Position

**Color scheme:**
- 🔴 **Red:** High positive CCG → judge under-penalizes
- 🟡 **Yellow:** Near-zero CCG → good calibration
- 🟢 **Green:** Negative CCG → judge over-penalizes

**Reading the heatmap:**
- Each cell = mean CCG for one condition (type × position)
- Cell annotation shows CCG value
- Gray text shows sample count (n=X)

**Key patterns to look for:**
- Red cells in "early" column → early errors under-penalized
- Horizontal patterns → type-specific calibration issues
- Vertical patterns → position-specific calibration issues

### Scatter Plot: TCS vs JPS

**Axes:**
- X-axis: True Criticality Score (ground truth)
- Y-axis: Judge Penalty Score (judge assessment)

**Reference line:**
- Black diagonal = perfect calibration (JPS = TCS)

**Point positions:**
- **Above line:** Judge over-penalizes (JPS > TCS)
- **Below line:** Judge under-penalizes (JPS < TCS)
- **On line:** Perfect calibration

**Color coding:**
- Different colors = different perturbation types
- Clustering by type suggests type-specific bias

### Distribution Plots

**Panel 1 (top-left): Overall CCG**
- Histogram of all CCG values
- Red dashed line = CCG = 0 (perfect)
- Look for: skewness, outliers, central tendency

**Panel 2 (top-right): CCG by Type**
- Box plots comparing perturbation types
- Look for: systematic differences, variance

**Panel 3 (bottom-left): CCG by Position**
- Box plots comparing trajectory positions
- Look for: early vs. middle vs. late patterns

**Panel 4 (bottom-right): TCS vs JPS**
- Overlapping histograms
- Look for: separation, overlap, distribution shape

### Bar Charts

**Left panel: By Type**
- Bars = mean CCG per perturbation type
- Error bars = ±1 standard deviation
- Gray text = sample count

**Right panel: By Position**
- Bars = mean CCG per trajectory position
- Same interpretation as left panel

**Look for:**
- Bars crossing zero line (over vs. under-penalization)
- Large error bars (high variability)
- Significant differences between bars

---

## Summary Statistics Tables

### Detailed Results CSV

Columns:
- `perturbation_id`: Unique ID
- `perturbation_type`: planning/tool_selection/parameter
- `perturbation_position`: early/middle/late
- `tsd`: Task Success Degradation (0 or 1)
- `ser`: Subsequent Error Rate (count)
- `tcs`: True Criticality Score
- `judge_overall_score`: Judge's score (0-100)
- `jps`: Judge Penalty Score (100 - judge_overall_score)
- `ccg`: Criticality-Calibration Gap

**Use for:**
- Deep dives into specific perturbations
- Filtering and custom analysis
- Exporting to other tools

### Summary Statistics CSV/XLSX

Rows:
- Overall
- By type (3 rows)
- By position (3 rows)
- By condition (9 rows: type × position)

Columns:
- Condition name
- Count (sample size)
- Mean CCG
- Std CCG
- Median CCG
- Mean TCS
- Mean JPS

**Use for:**
- Quick condition comparisons
- Table for paper
- Identifying patterns

---

## Multi-Judge Visualizations

When results from multiple judges exist, additional plots are generated:

### Cross-Judge CCG Comparison

Box plots comparing CCG distributions across judges.

### Judge Agreement Correlation

Scatter plot: Judge 1 CCG vs. Judge 2 CCG
- Diagonal line = perfect agreement
- Pearson r and p-value reported

**Use for:**
- Assessing judge reliability
- Human-LLM alignment analysis
- Inter-judge consistency

---

## Troubleshooting

### No visualizations generated

**Check:**
1. Results directory exists: `ls results/<experiment_id>/`
2. CCG results exist: `ls results/<experiment_id>/ccg_results_*.csv`
3. CCG files not empty: `wc -l results/<experiment_id>/ccg_results_*.csv`

**Fix:**
Run CCG computation first:
```bash
python main.py --config your_config --runner ccg
```

### Empty CSV for judge

**Symptom:**
```
Warning: Empty file ccg_results_gpt-oss-120b.csv, skipping judge gpt-oss-120b
```

**Cause:** No annotations for that judge's evaluations

**Fix:**
1. Run annotations: `--runner annotate`
2. Re-run CCG computation: `--runner ccg`
3. Then run: `--runner visualize`

### Import errors

**Error:** `ModuleNotFoundError: No module named 'openpyxl'`

**Fix:**
```bash
pip install openpyxl
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Low-quality images

**Solution:** Increase DPI:
```bash
python -m src.visualization.cli --experiment-id X --dpi 600
```

### Figures too small/large

**Solution:** Adjust figsize:
```bash
python -m src.visualization.cli --experiment-id X --figsize 16 12
```

---

## Best Practices

### For Paper Figures

1. **Use PDF format** (vector graphics)
   ```bash
   --formats pdf
   ```

2. **High DPI** (600+)
   ```bash
   --dpi 600
   ```

3. **Larger figures**
   ```bash
   --figsize 14 10
   ```

### For Presentations

1. **Use PNG format** (easy embedding)
   ```bash
   --formats png
   ```

2. **Standard DPI** (300)
   ```bash
   --dpi 300
   ```

3. **Standard size**
   ```bash
   --figsize 12 8
   ```

### For Quick Preview

1. **PNG only**
   ```bash
   --formats png
   ```

2. **Low DPI** (150)
   ```bash
   --dpi 150
   ```

---

## Examples

### Regenerate with different settings

```bash
# Original generation (default settings)
python main.py --config poc_experiment_toolbench --runner visualize

# Regenerate with high quality for paper
python -m src.visualization.cli \
  --experiment-id exp_poc_toolbench_20260402 \
  --figsize 14 10 \
  --dpi 600 \
  --formats pdf png

# Regenerate for presentation (16:9 aspect ratio)
python -m src.visualization.cli \
  --experiment-id exp_poc_toolbench_20260402 \
  --figsize 16 9 \
  --dpi 300 \
  --formats png
```

### Batch processing multiple experiments

```bash
#!/bin/bash
for exp_id in exp_1 exp_2 exp_3; do
  python -m src.visualization.cli \
    --experiment-id $exp_id \
    --dpi 300
done
```

---

## File Locations

```
results/<experiment_id>/
├── ccg_results_<judge>.csv           # Input: CCG results
├── ccg_summary_<judge>.json          # Input: CCG summary
└── visualizations/                   # Output directory
    ├── VISUALIZATION_REPORT.md       # Summary report
    ├── heatmap_ccg_<judge>.png       # Heatmap (PNG)
    ├── heatmap_ccg_<judge>.pdf       # Heatmap (PDF)
    ├── scatter_tcs_jps_<judge>.png   # Scatter (PNG)
    ├── scatter_tcs_jps_<judge>.pdf   # Scatter (PDF)
    ├── distributions_<judge>.png      # Distributions (PNG)
    ├── distributions_<judge>.pdf      # Distributions (PDF)
    ├── bar_charts_<judge>.png         # Bar charts (PNG)
    ├── bar_charts_<judge>.pdf         # Bar charts (PDF)
    ├── detailed_results_<judge>.csv   # All results
    ├── summary_statistics_<judge>.csv # Summary table
    └── summary_statistics_<judge>.xlsx # Summary (Excel)
```

---

## Additional Resources

- **Implementation:** [src/visualization/plots.py](../src/visualization/plots.py)
- **CLI:** [src/visualization/cli.py](../src/visualization/cli.py)
- **Results summary:** [paper/PHASE_6_RESULTS_SUMMARY.md](../paper/PHASE_6_RESULTS_SUMMARY.md)
- **Experiment runner:** [docs/RUNNER_GUIDE.md](RUNNER_GUIDE.md)
