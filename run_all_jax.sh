#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for Kaggle S5E12 using ONLY JAX/Flax models.
# Stages: JAX teacher + OOF, JAX adversarial weights, JAX teacher (adv-weighted) + OOF,
# JAX distillation (teacher-only + hard/soft), logit-space blend, JAX OOF stacking.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p sub logs

LOG_DIR="${LOG_DIR:-logs/run_all_jax_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

run() {
  local name="$1"; shift
  local log_file="$LOG_DIR/${name}.log"
  local start_ts end_ts dur_s

  echo "[run] ${name}"

  {
    echo "[meta] started_at=$(date -Is)"
    printf "[meta] cmd="
    printf " %q" "$@"
    echo
  } >"$log_file"

  start_ts="$(date +%s)"
  if "$@" >>"$log_file" 2>&1; then
    end_ts="$(date +%s)"
    dur_s=$((end_ts - start_ts))
    {
      echo "[meta] finished_at=$(date -Is)"
      echo "[meta] duration_s=${dur_s}"
      echo "[meta] status=ok"
    } >>"$log_file"
    echo "[done] ${name} (log: ${log_file}, ${dur_s}s)"
  else
    end_ts="$(date +%s)"
    dur_s=$((end_ts - start_ts))
    {
      echo "[meta] finished_at=$(date -Is)"
      echo "[meta] duration_s=${dur_s}"
      echo "[meta] status=error"
    } >>"$log_file"
    echo "ERROR: stage '${name}' failed (log: ${log_file})" >&2
    exit 1
  fi
}

# -------------------------------
# Environment / configuration
# -------------------------------

: "${PYTHON_BIN:=python}"
: "${VENV_ACTIVATE:=.venv/bin/activate}"

export PYTHONUNBUFFERED=1
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"

# Grokking experiment mode
# When GROK_MODE=1, defaults are adjusted to make a grokking-like transition
# more likely/observable on CPU (very long training + stronger weight decay,
# smaller CV, and more verbose logging). Set GROK_FULL_PIPELINE=1 to keep all stages.
: "${GROK_MODE:=0}"
: "${GROK_FULL_PIPELINE:=0}"

# If enabled, stages are skipped when their expected outputs + checkpoints already exist.
: "${SKIP_IF_EXISTS:=1}"

# Stage toggles (set to 0 to skip)
: "${RUN_TEACHER:=1}"
: "${RUN_ADV_WEIGHTS:=1}"
: "${RUN_TEACHER_ADVW:=1}"
: "${RUN_STUDENT_A:=0}"
: "${RUN_STUDENT_B:=1}"
: "${RUN_BLEND:=1}"
: "${RUN_STACK_OOF:=1}"
: "${RUN_SCRATCH_ADV:=1}"  # long-running stage

# Checkpoint saving
: "${SAVE_BEST_MODELS:=1}"
: "${CKPT_DIR:=checkpoints}"
: "${FT_SAVE_DIR:=$CKPT_DIR/fttransformer}"
: "${SCRATCH_SAVE_DIR:=$CKPT_DIR/jax_scratch_adv}"
: "${SCRATCH_SAVE_METRIC:=logloss}"  # auc|logloss

# Common dataset paths
: "${TRAIN_CSV:=data/train.csv}"
: "${TEST_CSV:=data/test.csv}"

# Common CV settings
: "${FOLDS:=5}"

# FtTransformer defaults (teacher/student)
: "${FT_SEEDS_TEACHER:=42}"
: "${FT_SEEDS_STUDENT:=42}"
: "${FT_NORM:=derf}"
: "${FT_D_MODEL:=128}"
: "${FT_N_HEADS:=8}"
: "${FT_N_LAYERS:=3}"
: "${FT_FF_MULT:=4}"
: "${FT_DROPOUT:=0.1}"
: "${FT_BATCH_SIZE:=2048}"
: "${FT_EPOCHS:=30}"
: "${FT_LR:=0.002}"
: "${FT_WEIGHT_DECAY:=1e-4}"
: "${FT_EARLY_STOP:=4}"
: "${FT_MAX_CATEGORIES:=2000}"
: "${FT_MAX_TRAIN_ROWS:=0}"
: "${FT_VERBOSE:=1}"

# Adversarial weights defaults
: "${ADV_SEED:=42}"
: "${ADV_EPOCHS:=5}"
: "${ADV_BATCH_SIZE:=4096}"
: "${ADV_HIDDEN:=256,128}"
: "${ADV_EMBED_DIM:=16}"
: "${ADV_DROPOUT:=0.1}"
: "${ADV_LR:=0.002}"
: "${ADV_WEIGHT_DECAY:=1e-4}"
: "${ADV_EARLY_STOP:=2}"
: "${ADV_MAX_CATEGORIES:=2000}"
: "${ADV_CLIP_MIN:=0.2}"
: "${ADV_CLIP_MAX:=5.0}"
: "${ADV_NORMALIZE:=1}"
: "${ADV_VERBOSE:=1}"

# Distillation defaults
: "${DIST_LABEL_SMOOTHING:=0.02}"
: "${DIST_SOFT_ALPHA_A:=1.0}"
: "${DIST_SOFT_ALPHA_B:=0.7}"
: "${DIST_VERBOSE:=1}"

# Blend defaults
: "${BLEND_MODE:=logit}"
: "${BLEND_WEIGHTS:=0.35 0.45 0.20}"
: "${BLEND_VERBOSE:=1}"

# Stack (meta-learner) defaults
: "${STACK_FEATURES:=logit}"
: "${STACK_C:=1.0}"
: "${STACK_STEPS:=3000}"
: "${STACK_LR:=0.05}"
: "${STACK_SEED:=42}"
: "${STACK_VERBOSE:=1}"

# Outputs
: "${OUT_OOF_TEACHER:=sub/jax_teacher_oof.csv}"
: "${OUT_SUB_TEACHER:=sub/submission_jax_teacher.csv}"
: "${OUT_ADV_WEIGHTS:=sub/train_weights_adv_jax.csv}"
: "${OUT_OOF_TEACHER_ADVW:=sub/jax_teacher_advw_oof.csv}"
: "${OUT_SUB_TEACHER_ADVW:=sub/submission_jax_teacher_advw.csv}"
: "${OUT_SUB_STUDENT_A:=sub/submission_jax_student_a.csv}"
: "${OUT_SUB_STUDENT_B:=sub/submission_jax_student_b.csv}"
: "${OUT_SUB_BLEND:=sub/submission_jax_blend_logit.csv}"
: "${OUT_SUB_STACK:=sub/submission_jax_stack.csv}"

# Scratch-adv defaults (requested command integrated as defaults)
: "${SCRATCH_FOLDS:=5}"
: "${SCRATCH_SEED:=42}"
: "${SCRATCH_EPOCHS:=600}"
: "${SCRATCH_BATCH_SIZE:=4096}"
: "${SCRATCH_USE_ADV_WEIGHTS:=1}"
: "${SCRATCH_ADD_DIST_FEATURES:=1}"
: "${SCRATCH_DIST_PER_FEATURE_LLR:=1}"
: "${SCRATCH_DIST_NB_LOGIT:=1}"
: "${SCRATCH_USE_MONO:=1}"
: "${SCRATCH_ADV_KIND:=sklearn}"
: "${SCRATCH_ADV_EPOCHS:=3}"
: "${SCRATCH_ADV_MAX_ROWS:=200000}"
: "${SCRATCH_ADV_CLIP_MIN:=0.2}"
: "${SCRATCH_ADV_CLIP_MAX:=5.0}"
: "${SCRATCH_NORM_KIND:=derf}"
: "${SCRATCH_PATIENCE:=120}"
: "${SCRATCH_EVAL_EVERY:=100}"
: "${SCRATCH_MONO_LAMBDA:=0.05}"
: "${SCRATCH_MONO_DELTA:=0.7532}"
: "${SCRATCH_MONO_K:=7}"
: "${SCRATCH_MONO_SPEC:=age:+1,bmi:+1,hdl_cholesterol:-1,physical_activity_minutes_per_week:-1}"
: "${OUT_SUB_SCRATCH:=sub/submission_jax_scratch_adv_dist_mono.csv}"

# Apply grokking defaults (override the usual defaults; you can still override via env)
if [[ "${GROK_MODE}" == "1" ]]; then
  # Keep the experiment lightweight while still showing a long-training effect
  FOLDS="${GROK_FOLDS:-2}"

  # FTTransformer: long training, stronger weight decay, no early stop, verbose per-epoch metrics
  FT_EPOCHS="${GROK_FT_EPOCHS:-4000}"
  FT_WEIGHT_DECAY="${GROK_FT_WEIGHT_DECAY:-0.005}"
  FT_DROPOUT="${GROK_FT_DROPOUT:-0.0}"
  FT_EARLY_STOP="${GROK_FT_EARLY_STOP:-999999}"
  FT_VERBOSE="${GROK_FT_VERBOSE:-2}"
  FT_SEEDS_TEACHER="${GROK_FT_SEEDS_TEACHER:-42}"

  # Use a smaller subset so a multi-thousand-epoch run finishes on laptop CPU.
  FT_MAX_TRAIN_ROWS="${GROK_FT_MAX_TRAIN_ROWS:-120000}"

  # Scratch model: long run + frequent eval; by default simplify features to isolate grokking
  SCRATCH_EPOCHS="${GROK_SCRATCH_EPOCHS:-5000}"
  SCRATCH_PATIENCE="${GROK_SCRATCH_PATIENCE:-4000}"
  SCRATCH_EVAL_EVERY="${GROK_SCRATCH_EVAL_EVERY:-50}"
  SCRATCH_ADV_KIND="${GROK_SCRATCH_ADV_KIND:-sklearn}"

  # By default, turn off extra “tricks” so the dynamics are easier to interpret.
  SCRATCH_USE_ADV_WEIGHTS="${GROK_SCRATCH_USE_ADV_WEIGHTS:-0}"
  SCRATCH_ADD_DIST_FEATURES="${GROK_SCRATCH_ADD_DIST_FEATURES:-0}"
  SCRATCH_USE_MONO="${GROK_SCRATCH_USE_MONO:-0}"

  if [[ "${GROK_FULL_PIPELINE}" != "1" ]]; then
    RUN_ADV_WEIGHTS=0
    RUN_TEACHER_ADVW=0
    RUN_STUDENT_A=0
    RUN_STUDENT_B=0
    RUN_BLEND=0
    RUN_STACK_OOF=0
  fi
fi

# Activate venv (optional but recommended)
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_ACTIVATE"
else
  echo "WARNING: venv activate script not found at '$VENV_ACTIVATE' (continuing with current Python: '$PYTHON_BIN')" >&2
fi

# -------------------------------
# Helpers
# -------------------------------

_split_csv() {
  # Usage: _split_csv "a,b,c"  -> prints one per line
  local s="${1:-}"
  s="${s// /}" # strip spaces
  if [[ -z "$s" ]]; then
    return 0
  fi
  # shellcheck disable=SC2001
  echo "$s" | tr ',' '\n'
}

_ft_ckpts_complete() {
  # Args: dir seeds_csv folds
  local dir="$1"; local seeds_csv="$2"; local folds="$3"
  local seed fold
  for seed in $(_split_csv "$seeds_csv"); do
    for ((fold=0; fold<folds; fold++)); do
      if [[ ! -f "${dir}/ftt_seed${seed}_fold${fold}.msgpack" ]]; then
        return 1
      fi
    done
  done
  return 0
}

_scratch_ckpts_complete() {
  # Args: dir seed folds
  local dir="$1"; local seed="$2"; local folds="$3"
  local fold
  for ((fold=0; fold<folds; fold++)); do
    if [[ ! -f "${dir}/scratch_seed${seed}_fold${fold}.msgpack" ]]; then
      return 1
    fi
  done
  return 0
}

# Preflight checks
if [[ ! -f "$TRAIN_CSV" ]]; then
  echo "ERROR: $TRAIN_CSV not found" >&2
  exit 1
fi
if [[ ! -f "$TEST_CSV" ]]; then
  echo "ERROR: $TEST_CSV not found" >&2
  exit 1
fi

# -------------------------------
# Stages
# -------------------------------

# 1) JAX teacher (FtTransformer) + OOF
if [[ "${RUN_TEACHER}" == "1" ]]; then
  goto_run_teacher=0
  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "$OUT_SUB_TEACHER" && -f "$OUT_OOF_TEACHER" ]]; then
    if [[ "${SAVE_BEST_MODELS}" != "1" ]] || _ft_ckpts_complete "$FT_SAVE_DIR/teacher" "$FT_SEEDS_TEACHER" "$FOLDS"; then
      echo "[skip] 01_jax_teacher (outputs/checkpoints already exist)"
    else
      echo "[info] 01_jax_teacher outputs exist but checkpoints incomplete; will run"
      goto_run_teacher=1
    fi
  else
    goto_run_teacher=1
  fi

  if [[ "${goto_run_teacher:-0}" == "1" ]]; then
  cmd_teacher=(
    "$PYTHON_BIN" train_jax_fttransformer.py
    --folds "$FOLDS"
    --seeds "$FT_SEEDS_TEACHER"
    --train "$TRAIN_CSV"
    --test "$TEST_CSV"
    --norm "$FT_NORM"
    --d-model "$FT_D_MODEL" --n-heads "$FT_N_HEADS" --n-layers "$FT_N_LAYERS" --ff-mult "$FT_FF_MULT"
    --dropout "$FT_DROPOUT"
    --batch-size "$FT_BATCH_SIZE"
    --epochs "$FT_EPOCHS"
    --lr "$FT_LR"
    --weight-decay "$FT_WEIGHT_DECAY"
    --early-stop "$FT_EARLY_STOP"
    --max-categories "$FT_MAX_CATEGORIES"
    --oof-out "$OUT_OOF_TEACHER"
    --out "$OUT_SUB_TEACHER"
    --zip-output
    --verbose "$FT_VERBOSE"
  )
  if [[ "${SAVE_BEST_MODELS}" == "1" ]]; then
    cmd_teacher+=(--save-best-dir "$FT_SAVE_DIR/teacher")
  fi
  if [[ "${FT_MAX_TRAIN_ROWS}" != "0" && -n "${FT_MAX_TRAIN_ROWS}" ]]; then
    cmd_teacher+=(--max-train-rows "$FT_MAX_TRAIN_ROWS")
  fi
  run "01_jax_teacher" "${cmd_teacher[@]}"
  fi
fi

# 2) JAX adversarial weights
if [[ "${RUN_ADV_WEIGHTS}" == "1" ]]; then
  cmd_adv=(
    "$PYTHON_BIN" jax_adversarial_reweight.py
    --folds "$FOLDS"
    --seed "$ADV_SEED"
    --epochs "$ADV_EPOCHS"
    --batch-size "$ADV_BATCH_SIZE"
    --hidden "$ADV_HIDDEN"
    --embed-dim "$ADV_EMBED_DIM"
    --dropout "$ADV_DROPOUT"
    --lr "$ADV_LR"
    --weight-decay "$ADV_WEIGHT_DECAY"
    --early-stop "$ADV_EARLY_STOP"
    --max-categories "$ADV_MAX_CATEGORIES"
    --clip-min "$ADV_CLIP_MIN"
    --clip-max "$ADV_CLIP_MAX"
  )
  if [[ "$ADV_NORMALIZE" == "1" ]]; then
    cmd_adv+=(--normalize)
  fi
  cmd_adv+=(--out "$OUT_ADV_WEIGHTS" --verbose "$ADV_VERBOSE")
  run "02_jax_adv_weights" "${cmd_adv[@]}"
fi

# 3) JAX teacher with adversarial weights + OOF
if [[ "${RUN_TEACHER_ADVW}" == "1" ]]; then
  goto_run_teacher_advw=0
  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "$OUT_SUB_TEACHER_ADVW" && -f "$OUT_OOF_TEACHER_ADVW" ]]; then
    if [[ "${SAVE_BEST_MODELS}" != "1" ]] || _ft_ckpts_complete "$FT_SAVE_DIR/teacher_advw" "$FT_SEEDS_TEACHER" "$FOLDS"; then
      echo "[skip] 03_jax_teacher_advw (outputs/checkpoints already exist)"
    else
      echo "[info] 03_jax_teacher_advw outputs exist but checkpoints incomplete; will run"
      goto_run_teacher_advw=1
    fi
  else
    goto_run_teacher_advw=1
  fi

  if [[ "${goto_run_teacher_advw:-0}" == "1" ]]; then
  cmd_teacher_advw=(
    "$PYTHON_BIN" train_jax_fttransformer.py
    --train-weights "$OUT_ADV_WEIGHTS"
    --folds "$FOLDS"
    --seeds "$FT_SEEDS_TEACHER"
    --train "$TRAIN_CSV"
    --test "$TEST_CSV"
    --norm "$FT_NORM"
    --d-model "$FT_D_MODEL" --n-heads "$FT_N_HEADS" --n-layers "$FT_N_LAYERS" --ff-mult "$FT_FF_MULT"
    --dropout "$FT_DROPOUT"
    --batch-size "$FT_BATCH_SIZE"
    --epochs "$FT_EPOCHS"
    --lr "$FT_LR"
    --weight-decay "$FT_WEIGHT_DECAY"
    --early-stop "$FT_EARLY_STOP"
    --max-categories "$FT_MAX_CATEGORIES"
    --oof-out "$OUT_OOF_TEACHER_ADVW"
    --out "$OUT_SUB_TEACHER_ADVW"
    --zip-output
    --verbose "$FT_VERBOSE"
  )
  if [[ "${SAVE_BEST_MODELS}" == "1" ]]; then
    cmd_teacher_advw+=(--save-best-dir "$FT_SAVE_DIR/teacher_advw")
  fi
  if [[ "${FT_MAX_TRAIN_ROWS}" != "0" && -n "${FT_MAX_TRAIN_ROWS}" ]]; then
    cmd_teacher_advw+=(--max-train-rows "$FT_MAX_TRAIN_ROWS")
  fi
  run "03_jax_teacher_advw" "${cmd_teacher_advw[@]}"
  fi
fi

# 4a) Distillation (teacher-only)
if [[ "${RUN_STUDENT_A}" == "1" ]]; then
  cmd_student_a=(
    "$PYTHON_BIN" jax_distill_student.py
    --teacher-oof "$OUT_OOF_TEACHER_ADVW"
    --soft-alpha "$DIST_SOFT_ALPHA_A"
    --label-smoothing "$DIST_LABEL_SMOOTHING"
    --train "$TRAIN_CSV"
    --test "$TEST_CSV"
    --folds "$FOLDS"
    --seeds "$FT_SEEDS_STUDENT"
    --norm "$FT_NORM"
    --d-model "$FT_D_MODEL" --n-heads "$FT_N_HEADS" --n-layers "$FT_N_LAYERS" --ff-mult "$FT_FF_MULT"
    --dropout "$FT_DROPOUT"
    --batch-size "$FT_BATCH_SIZE"
    --epochs "$FT_EPOCHS"
    --lr "$FT_LR"
    --weight-decay "$FT_WEIGHT_DECAY"
    --early-stop "$FT_EARLY_STOP"
    --max-categories "$FT_MAX_CATEGORIES"
    --out "$OUT_SUB_STUDENT_A"
    --zip-output
    --verbose "$DIST_VERBOSE"
  )
  if [[ "${FT_MAX_TRAIN_ROWS}" != "0" && -n "${FT_MAX_TRAIN_ROWS}" ]]; then
    cmd_student_a+=(--max-train-rows "$FT_MAX_TRAIN_ROWS")
  fi
  run "04a_jax_student_teacher_only" "${cmd_student_a[@]}"
fi

# 4b) Distillation (hard+soft)
if [[ "${RUN_STUDENT_B}" == "1" ]]; then
  goto_run_student_b=0
  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "$OUT_SUB_STUDENT_B" ]]; then
    if [[ "${SAVE_BEST_MODELS}" != "1" ]] || _ft_ckpts_complete "$FT_SAVE_DIR/student_b" "$FT_SEEDS_STUDENT" "$FOLDS"; then
      echo "[skip] 04b_jax_student_hard_soft (output/checkpoints already exist)"
    else
      echo "[info] 04b_jax_student_hard_soft output exists but checkpoints incomplete; will run"
      goto_run_student_b=1
    fi
  else
    goto_run_student_b=1
  fi

  if [[ "${goto_run_student_b:-0}" == "1" ]]; then
  cmd_student_b=(
    "$PYTHON_BIN" jax_distill_student.py
    --teacher-oof "$OUT_OOF_TEACHER_ADVW"
    --soft-alpha "$DIST_SOFT_ALPHA_B"
    --label-smoothing "$DIST_LABEL_SMOOTHING"
    --train "$TRAIN_CSV"
    --test "$TEST_CSV"
    --folds "$FOLDS"
    --seeds "$FT_SEEDS_STUDENT"
    --norm "$FT_NORM"
    --d-model "$FT_D_MODEL" --n-heads "$FT_N_HEADS" --n-layers "$FT_N_LAYERS" --ff-mult "$FT_FF_MULT"
    --dropout "$FT_DROPOUT"
    --batch-size "$FT_BATCH_SIZE"
    --epochs "$FT_EPOCHS"
    --lr "$FT_LR"
    --weight-decay "$FT_WEIGHT_DECAY"
    --early-stop "$FT_EARLY_STOP"
    --max-categories "$FT_MAX_CATEGORIES"
    --out "$OUT_SUB_STUDENT_B"
    --zip-output
    --verbose "$DIST_VERBOSE"
  )
  if [[ "${SAVE_BEST_MODELS}" == "1" ]]; then
    cmd_student_b+=(--save-best-dir "$FT_SAVE_DIR/student_b")
  fi
  if [[ "${FT_MAX_TRAIN_ROWS}" != "0" && -n "${FT_MAX_TRAIN_ROWS}" ]]; then
    cmd_student_b+=(--max-train-rows "$FT_MAX_TRAIN_ROWS")
  fi
  run "04b_jax_student_hard_soft" "${cmd_student_b[@]}"
  fi
fi

# 5) Logit-space blend
if [[ "${RUN_BLEND}" == "1" ]]; then
  # BLEND_WEIGHTS is a whitespace-separated string; split into an array
  read -r -a blend_weights_arr <<<"$BLEND_WEIGHTS"
  cmd_blend=(
    "$PYTHON_BIN" jax_blend_submissions.py
    --inputs "$OUT_SUB_TEACHER" "$OUT_SUB_TEACHER_ADVW" "$OUT_SUB_STUDENT_B"
    --weights "${blend_weights_arr[@]}"
    --mode "$BLEND_MODE"
    --out "$OUT_SUB_BLEND"
    --zip-output
    --verbose "$BLEND_VERBOSE"
  )
  run "05_jax_blend_logit" "${cmd_blend[@]}"
fi

# 6) OOF stacking (meta-learner in JAX)
if [[ "${RUN_STACK_OOF}" == "1" ]]; then
  cmd_stack=(
    "$PYTHON_BIN" jax_stack_oof.py
    --oof "$OUT_OOF_TEACHER" "$OUT_OOF_TEACHER_ADVW"
    --subs "$OUT_SUB_TEACHER" "$OUT_SUB_TEACHER_ADVW"
    --features "$STACK_FEATURES"
    --C "$STACK_C"
    --steps "$STACK_STEPS"
    --lr "$STACK_LR"
    --seed "$STACK_SEED"
    --out "$OUT_SUB_STACK"
    --zip-output
    --verbose "$STACK_VERBOSE"
  )
  run "06_jax_stack_oof" "${cmd_stack[@]}"
fi

# 7) From-scratch JAX/Flax model with adv-weights + dist features + monotonic penalty
if [[ "${RUN_SCRATCH_ADV}" == "1" ]]; then
  goto_run_scratch=0
  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "$OUT_SUB_SCRATCH" ]]; then
    if [[ "${SAVE_BEST_MODELS}" != "1" ]] || _scratch_ckpts_complete "$SCRATCH_SAVE_DIR" "$SCRATCH_SEED" "$SCRATCH_FOLDS"; then
      echo "[skip] 07_jax_scratch_adv_dist_mono (output/checkpoints already exist)"
    else
      echo "[info] 07_jax_scratch_adv_dist_mono output exists but checkpoints incomplete; will run"
      goto_run_scratch=1
    fi
  else
    goto_run_scratch=1
  fi

  if [[ "${goto_run_scratch:-0}" == "1" ]]; then
  cmd_scratch=(
    "$PYTHON_BIN" -m jax_scratch_adv.run
    --train "$TRAIN_CSV"
    --test "$TEST_CSV"
    --out "$OUT_SUB_SCRATCH"
    --folds "$SCRATCH_FOLDS"
    --seed "$SCRATCH_SEED"
    --epochs "$SCRATCH_EPOCHS"
    --batch-size "$SCRATCH_BATCH_SIZE"
    --norm-kind "$SCRATCH_NORM_KIND"
    --patience "$SCRATCH_PATIENCE"
    --eval-every "$SCRATCH_EVAL_EVERY"
  )

  if [[ "${SAVE_BEST_MODELS}" == "1" ]]; then
    cmd_scratch+=(--save-best-dir "$SCRATCH_SAVE_DIR" --save-metric "$SCRATCH_SAVE_METRIC")
  fi

  if [[ "${SCRATCH_USE_ADV_WEIGHTS}" == "1" ]]; then
    cmd_scratch+=(
      --use-adv-weights
      --adv-kind "$SCRATCH_ADV_KIND"
      --adv-epochs "$SCRATCH_ADV_EPOCHS"
      --adv-max-rows "$SCRATCH_ADV_MAX_ROWS"
      --adv-clip-min "$SCRATCH_ADV_CLIP_MIN"
      --adv-clip-max "$SCRATCH_ADV_CLIP_MAX"
    )
  fi

  if [[ "${SCRATCH_ADD_DIST_FEATURES}" == "1" ]]; then
    cmd_scratch+=(--add-dist-features)
    if [[ "${SCRATCH_DIST_PER_FEATURE_LLR}" == "1" ]]; then
      cmd_scratch+=(--dist-per-feature-llr)
    fi
    if [[ "${SCRATCH_DIST_NB_LOGIT}" == "1" ]]; then
      cmd_scratch+=(--dist-nb-logit)
    fi
  fi

  if [[ "${SCRATCH_USE_MONO}" == "1" ]]; then
    cmd_scratch+=(
      --use-mono
      --mono "$SCRATCH_MONO_SPEC"
      --mono-lambda "$SCRATCH_MONO_LAMBDA"
      --mono-delta "$SCRATCH_MONO_DELTA"
      --mono-k "$SCRATCH_MONO_K"
    )
  fi

  run "07_jax_scratch_adv_dist_mono" "${cmd_scratch[@]}"
  fi
fi

echo "All done. Key outputs:"
ls -lh "$OUT_SUB_STACK" "$OUT_SUB_BLEND" 2>/dev/null || true
if [[ "${RUN_SCRATCH_ADV}" == "1" ]]; then
  ls -lh "$OUT_SUB_SCRATCH" 2>/dev/null || true
fi
