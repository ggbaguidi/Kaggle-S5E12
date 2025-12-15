#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for Kaggle S5E12 (CPU).
# Runs: LGBM base + OOF, CatBoost base + OOF, adversarial weights, LGBM adv-weighted + OOF,
# distillation, logit-space blend, and OOF stacking.
# Optional: JAX MLP run (set RUN_JAX=1).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p sub logs

LOG_DIR="logs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run() {
  local name="$1"; shift
  echo "[run] ${name}"
  # shellcheck disable=SC2068
  ( $@ ) >"$LOG_DIR/${name}.log" 2>&1
  echo "[done] ${name} (log: $LOG_DIR/${name}.log)"
}

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  echo "ERROR: .venv/bin/activate not found" >&2
  exit 1
fi

# 1) LightGBM base (teacher) + OOF
run "01_lgbm_base" python train_lgbm.py \
  --folds 5 \
  --seeds 42,43,44 \
  --num-boost-round 9000 \
  --early-stopping-rounds 400 \
  --learning-rate 0.025 \
  --num-leaves 192 \
  --min-data-in-leaf 25 \
  --min-sum-hessian-in-leaf 1e-3 \
  --feature-fraction 0.8 \
  --bagging-fraction 0.8 \
  --bagging-freq 1 \
  --lambda-l1 0.0 \
  --lambda-l2 0.0 \
  --cat-smooth 20 \
  --cat-l2 20 \
  --max-cat-to-onehot 4 \
  --min-data-per-group 100 \
  --max-categories 2000 \
  --oof-out sub/lgbm_oof.csv \
  --out sub/submission_lgbm_base.csv \
  --zip-output \
  --verbose 1

# 2) CatBoost base + OOF
run "02_cat_base" python train_catboost.py \
  --folds 5 \
  --seeds 42,43 \
  --iterations 12000 \
  --od-wait 500 \
  --learning-rate 0.03 \
  --depth 8 \
  --l2-leaf-reg 6.0 \
  --random-strength 1.0 \
  --bagging-temperature 0.8 \
  --border-count 128 \
  --thread-count -1 \
  --oof-out sub/cat_oof.csv \
  --out sub/submission_cat_base.csv \
  --zip-output \
  --verbose 1

# 3) Adversarial reweighting
run "03_adv_weights" python adversarial_reweight.py \
  --folds 5 \
  --seed 42 \
  --num-boost-round 2000 \
  --early-stopping-rounds 200 \
  --learning-rate 0.05 \
  --num-leaves 64 \
  --min-data-in-leaf 200 \
  --feature-fraction 0.8 \
  --bagging-fraction 0.8 \
  --bagging-freq 1 \
  --lambda-l2 1.0 \
  --max-categories 2000 \
  --clip-min 0.2 \
  --clip-max 5.0 \
  --out sub/train_weights_adv.csv \
  --verbose 1

# 4) LightGBM with adversarial weights + OOF
run "04_lgbm_advw" python train_lgbm.py \
  --train-weights sub/train_weights_adv.csv \
  --folds 5 \
  --seeds 42,43,44 \
  --num-boost-round 9000 \
  --early-stopping-rounds 400 \
  --learning-rate 0.025 \
  --num-leaves 192 \
  --min-data-in-leaf 25 \
  --feature-fraction 0.8 \
  --bagging-fraction 0.8 \
  --bagging-freq 1 \
  --cat-smooth 20 \
  --cat-l2 20 \
  --max-cat-to-onehot 4 \
  --min-data-per-group 100 \
  --max-categories 2000 \
  --oof-out sub/lgbm_advw_oof.csv \
  --out sub/submission_lgbm_advw.csv \
  --zip-output \
  --verbose 1

# 5a) Distillation (teacher-only)
run "05a_student_teacher_only" python distill_student.py \
  --teacher-oof sub/lgbm_advw_oof.csv \
  --soft-alpha 1.0 \
  --label-smoothing 0.02 \
  --folds 5 \
  --seed 42 \
  --num-boost-round 7000 \
  --early-stopping-rounds 400 \
  --learning-rate 0.03 \
  --num-leaves 64 \
  --min-data-in-leaf 40 \
  --feature-fraction 0.9 \
  --bagging-fraction 0.9 \
  --bagging-freq 1 \
  --lambda-l2 1.0 \
  --max-categories 2000 \
  --out sub/submission_student_a.csv \
  --zip-output \
  --verbose 1

# 5b) Distillation (hard+soft blend)
run "05b_student_hard_soft" python distill_student.py \
  --teacher-oof sub/lgbm_advw_oof.csv \
  --soft-alpha 0.7 \
  --label-smoothing 0.02 \
  --folds 5 \
  --seed 42 \
  --num-boost-round 7000 \
  --early-stopping-rounds 400 \
  --learning-rate 0.03 \
  --num-leaves 64 \
  --min-data-in-leaf 40 \
  --feature-fraction 0.9 \
  --bagging-fraction 0.9 \
  --bagging-freq 1 \
  --lambda-l2 1.0 \
  --max-categories 2000 \
  --out sub/submission_student_b.csv \
  --zip-output \
  --verbose 1

# 6) Logit-space blend
run "06_blend_logit" python blend_submissions.py \
  --inputs sub/submission_lgbm_base.csv sub/submission_lgbm_advw.csv sub/submission_cat_base.csv sub/submission_student_b.csv \
  --weights 0.25 0.35 0.25 0.15 \
  --mode logit \
  --out sub/submission_blend_logit.csv \
  --zip-output \
  --verbose 1

# 7) OOF stacking (meta-learner)
run "07_stack_oof" python stack_oof.py \
  --oof sub/lgbm_oof.csv sub/lgbm_advw_oof.csv sub/cat_oof.csv \
  --subs sub/submission_lgbm_base.csv sub/submission_lgbm_advw.csv sub/submission_cat_base.csv \
  --features logit \
  --C 1.0 \
  --max-iter 400 \
  --out sub/submission_stack.csv \
  --zip-output \
  --verbose 1

# Optional: JAX MLP (blend partner)
if [[ "${RUN_JAX:-0}" == "1" ]]; then
  run "08_jax_mlp" python train_jax_mlp.py \
    --seed 42 \
    --val-frac 0.1 \
    --epochs 20 \
    --batch-size 4096 \
    --hidden 256,128,64 \
    --embed-dim 16 \
    --dropout 0.1 \
    --lr 0.002 \
    --weight-decay 1e-4 \
    --early-stop 3 \
    --out sub/submission_jax.csv \
    --zip-output \
    --verbose 1

  run "09_blend_final" python blend_submissions.py \
    --inputs sub/submission_stack.csv sub/submission_jax.csv \
    --weights 0.8 0.2 \
    --mode logit \
    --out sub/submission_final.csv \
    --zip-output \
    --verbose 1
fi

echo "All done. Key outputs:" 
ls -lh sub/submission_stack.csv sub/submission_blend_logit.csv 2>/dev/null || true
if [[ "${RUN_JAX:-0}" == "1" ]]; then
  ls -lh sub/submission_final.csv 2>/dev/null || true
fi
