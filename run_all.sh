#!/bin/bash
# ──────────────────────────────────────────────────────────────
# MambGAT-AD 全量实验脚本
# 依次训练 SMAP / MSL / SMD，并跑 Telemanom baseline
#
# 用法：
#   bash run_all.sh           # 跑全部
#   bash run_all.sh smap      # 只跑 SMAP
#   bash run_all.sh baseline  # 只跑 baseline
# ──────────────────────────────────────────────────────────────

set -e
LOG_DIR="/root/autodl-tmp/logs"
mkdir -p "$LOG_DIR" checkpoints

TARGET=${1:-"all"}

run_model() {
    local dataset=$1
    local config="config/${dataset}.yaml"
    echo ""
    echo "════════════════════════════════════════════"
    echo "  MambGAT-AD | ${dataset^^} | $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════"
    python train.py --config "$config" 2>&1 | tee "${LOG_DIR}/train_${dataset}.log"
    echo "✓ ${dataset^^} 训练完成"
}

run_baseline() {
    local dataset=$1
    local config="config/${dataset}.yaml"
    echo ""
    echo "────────────────────────────────────────────"
    echo "  Telemanom Baseline | ${dataset^^} | $(date '+%H:%M:%S')"
    echo "────────────────────────────────────────────"
    python baselines/telemanom_baseline.py --config "$config" \
        2>&1 | tee "${LOG_DIR}/baseline_${dataset}.log"
    echo "✓ Telemanom ${dataset^^} 完成"
}

# ── 主模型 ───────────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "smap" ]]; then
    run_model smap
fi

if [[ "$TARGET" == "all" || "$TARGET" == "msl" ]]; then
    run_model msl
fi

if [[ "$TARGET" == "all" || "$TARGET" == "smd" ]]; then
    run_model smd
fi

# ── Baseline ─────────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "baseline" ]]; then
    run_baseline smap
    run_baseline msl
    run_baseline smd
fi

# ── 汇总结果 ─────────────────────────────────────────────────
if [[ "$TARGET" == "all" ]]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "  所有实验完成，汇总结果"
    echo "════════════════════════════════════════════"
    python - << 'EOF'
import json, os, glob

print(f"\n{'='*62}")
print(f"  {'数据集':<8} {'方法':<20} {'VUS-ROC':>8} {'F1-PA':>8} {'AUC-ROC':>8}")
print(f"{'='*62}")

for ds in ['smap', 'msl', 'smd']:
    # 主模型
    path = f"checkpoints/results_{ds}.json"
    if os.path.exists(path):
        r = json.load(open(path))
        vus = r.get('vus_roc', r.get('per_channel_macro', {}).get('vus_roc', 0))
        f1  = r.get('f1_pa',   r.get('per_channel_macro', {}).get('f1_pa',  0))
        auc = r.get('auc_roc', r.get('per_channel_macro', {}).get('auc_roc',0))
        print(f"  {ds.upper():<8} {'MambGAT-AD':<20} {vus:>8.4f} {f1:>8.4f} {auc:>8.4f}")

    # Baseline 日志
    log = f"/root/autodl-tmp/logs/baseline_{ds}.log"
    if os.path.exists(log):
        txt = open(log).read()
        import re
        m = re.search(r'VUS-ROC.*?(\d+\.\d+)', txt)
        if m:
            print(f"  {ds.upper():<8} {'Telemanom':<20} {float(m.group(1)):>8.4f}")

print(f"{'='*62}\n")
EOF
fi
