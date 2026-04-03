#!/bin/bash
# Quick test script for judge evaluation dry run

echo "🧪 Testing judge evaluation dry run..."
echo ""

# Run dry run with limited output
python main.py --config poc_experiment_toolbench --runner judge --dry-run 2>&1 | head -200

echo ""
echo "✅ If you see '[DRY RUN] Would evaluate...' messages above, the system is working!"
echo ""
echo "Next steps:"
echo "1. Review the output above"
echo "2. If it looks good, run WITHOUT --dry-run to execute actual evaluation"
echo "3. Command: python main.py --config poc_experiment_toolbench --runner judge"
