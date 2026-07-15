#!/usr/bin/env bash
# run_all_datasets.sh
# Runs the full 10-fold experiment on all 5 NASA PROMISE datasets.
# Usage: bash run_all_datasets.sh [--device cuda]

DEVICE=${1:-cpu}
DATA_DIR="data/raw"
OUT_DIR="results"

declare -A PATHS
PATHS["CM1"]="$DATA_DIR/cm1.arff"
PATHS["JM1"]="$DATA_DIR/jm1.arff"
PATHS["KC1"]="$DATA_DIR/kc1.arff"
PATHS["PC1"]="$DATA_DIR/pc1.arff"
PATHS["MC2"]="$DATA_DIR/mc2.arff"

echo "============================================================"
echo " AFS-GAN + BiLSTM-Transformer + Meta-Learner"
echo " Running on 5 NASA PROMISE datasets"
echo " Device: $DEVICE"
echo "============================================================"

for DS in CM1 JM1 KC1 PC1 MC2; do
    echo ""
    echo "------------------------------------------------------------"
    echo " Dataset: $DS"
    echo "------------------------------------------------------------"
    python run_experiment.py \
        --dataset "$DS" \
        --data_path "${PATHS[$DS]}" \
        --device "$DEVICE" \
        --out_dir "$OUT_DIR" \
        --seed 42
done

echo ""
echo "============================================================"
echo " All experiments complete. Results saved to $OUT_DIR/"
echo "============================================================"
