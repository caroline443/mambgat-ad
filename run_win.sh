#!/bin/bash
# WSL / Windows A4000 全量训练脚本
# 用法：bash run_win.sh [smap|msl|smd|all]

set -e
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" checkpoints

TARGET=${1:-"all"}

run_model() {
    local dataset=$1
    local config="config/${dataset}_win.yaml"
    local resume_flag=""

    if [ -f "checkpoints/last_${dataset}.pt" ]; then
        echo "  [Resume] 检测到 last_${dataset}.pt，自动续跑"
        resume_flag="--resume"
    fi

    echo ""
    echo "════════════════════════════════════════════"
    echo "  MambGAT-AD | ${dataset^^} | $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════"
    python train.py --config "$config" $resume_flag 2>&1 | tee "${LOG_DIR}/train_${dataset}.log"
    echo "✓ ${dataset^^} 训练完成"
}

if [[ "$TARGET" == "all" || "$TARGET" == "smap" ]]; then run_model smap; fi
if [[ "$TARGET" == "all" || "$TARGET" == "msl"  ]]; then run_model msl;  fi
if [[ "$TARGET" == "all" || "$TARGET" == "smd"  ]]; then run_model smd;  fi
