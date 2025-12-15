# S5E12 – Diabetes probability (CPU-only)

This workspace includes a RAM-friendly trainer that streams CSVs in chunks (works well on a 16GB laptop, no GPU).

## Run

```zsh
cd /home/ggbaguidi/Documents/Contests/Kaggle/S5E12
source .venv/bin/activate
python train_streaming.py --train data/train.csv --test data/test.csv --out submission.csv
```

## Tune (recommended before submitting)

Runs a small hyperparameter grid and selects the best config by validation logloss using a deterministic split based on `id`.

```zsh
python train_streaming.py --tune --val-frac 0.1 --tune-passes 1 --passes 3 --out submission.csv
```

## Stronger models (CPU)

### LightGBM (recommended)

Multi-seed ensembling usually boosts leaderboard scores.

```zsh
python train_lgbm.py \
	--folds 5 \
	--seeds 42,43,44 \
	--num-boost-round 8000 \
	--early-stopping-rounds 300 \
	--learning-rate 0.03 \
	--num-leaves 128 \
	--min-data-in-leaf 30 \
	--feature-fraction 0.8 \
	--bagging-fraction 0.8 \
	--bagging-freq 1 \
	--cat-smooth 10 \
	--cat-l2 10 \
	--max-cat-to-onehot 4 \
	--min-data-per-group 100 \
	--out sub/submission_lgbm.csv \
	--zip-output
```

### Using the original Diabetes dataset (optional)

If you download the original dataset and format it to have the SAME feature columns as `data/train.csv` (and a compatible binary target), you can add it as extra training data.

- Put it somewhere like `data/original.csv`
- Use a small downweight to avoid distribution-shift overfitting (start with `--extra-weight 0.1` to `0.5`).
- Extra rows are only added to training folds; validation stays competition-only.

```zsh
python train_lgbm.py \
	--extra-train data/original.csv \
	--extra-target-col diagnosed_diabetes \
	--extra-weight 0.3 \
	--folds 5 --seeds 42,43,44 \
	--out sub/submission_lgbm_extra.csv \
	--zip-output
```

### CatBoost (for blending)

```zsh
python train_catboost.py \
	--folds 5 \
	--seeds 42,43 \
	--iterations 8000 \
	--od-wait 300 \
	--learning-rate 0.03 \
	--depth 8 \
	--out sub/submission_cat.csv \
	--zip-output
```

With extra data:

```zsh
python train_catboost.py \
	--extra-train data/original.csv \
	--extra-target-col diagnosed_diabetes \
	--extra-weight 0.3 \
	--folds 5 --seeds 42,43 \
	--out sub/submission_cat_extra.csv \
	--zip-output
```

### Blend submissions

Start with a simple weighted average (tune weights on CV).

```zsh
python blend_submissions.py \
	--inputs sub/submission_lgbm.csv sub/submission_cat.csv \
	--weights 0.6 0.4 \
	--mode prob \
	--out sub/submission_blend.csv \
	--zip-output
```

Logit-space blending (often better than prob-averaging when models are miscalibrated):

```zsh
python blend_submissions.py \
	--inputs sub/submission_lgbm.csv sub/submission_cat.csv \
	--weights 0.6 0.4 \
	--mode logit \
	--out sub/submission_blend_logit.csv \
	--zip-output
```

## Synthetic-data tricks

### Adversarial reweighting (train vs test)

Learns weights so training focuses on rows that look like the test distribution.

```zsh
python adversarial_reweight.py \
	--folds 5 \
	--num-boost-round 1500 \
	--early-stopping-rounds 100 \
	--out sub/train_weights_adv.csv

python train_lgbm.py \
	--train-weights sub/train_weights_adv.csv \
	--folds 5 --seeds 42,43,44 \
	--out sub/submission_lgbm_advw.csv \
	--zip-output
```

### Teacher → student distillation

Train a strong teacher, export OOF probabilities, then train a student to mimic them.

```zsh
python train_lgbm.py \
	--folds 5 --seeds 42,43,44 \
	--oof-out sub/teacher_oof.csv \
	--out sub/submission_teacher.csv

python distill_student.py \
	--teacher-oof sub/teacher_oof.csv \
	--soft-alpha 1.0 \
	--folds 5 \
	--out sub/submission_student.csv \
	--zip-output
```

### OOF stacking (meta-learner)

Fit a logistic-regression meta-model on OOF predictions, then apply it to the corresponding test submissions.

```zsh
python train_lgbm.py \
	--folds 5 --seeds 42,43,44 \
	--oof-out sub/lgbm_oof.csv \
	--out sub/submission_lgbm_for_stack.csv

python train_catboost.py \
	--folds 5 --seeds 42,43 \
	--oof-out sub/cat_oof.csv \
	--out sub/submission_cat_for_stack.csv

python stack_oof.py \
	--oof sub/lgbm_oof.csv sub/cat_oof.csv \
	--subs sub/submission_lgbm_for_stack.csv sub/submission_cat_for_stack.csv \
	--features logit \
	--C 1.0 \
	--out sub/submission_stack.csv \
	--zip-output
```

## Optional: JAX MLP (CPU)

Sometimes helps on synthetic-tabular data. Requires installing `jax[cpu]`, `flax`, and `optax`.

```zsh
python train_jax_mlp.py \
	--epochs 20 \
	--batch-size 4096 \
	--hidden 256,128,64 \
	--embed-dim 16 \
	--out sub/submission_jax.csv \
	--zip-output
```

## Useful knobs

- `--chunk-size 50000` (lower if you hit RAM pressure)
- `--passes 1` (faster) or `--passes 3` (often a bit better)
- `--hash-dim 262144` (default) or larger for more categorical capacity
- `--tune-max-train-rows 200000` (speed up tuning)
- `--zip-output` (zip the submission CSV for upload)


## Competitions
https://www.kaggle.com/competitions/playground-series-s5e12