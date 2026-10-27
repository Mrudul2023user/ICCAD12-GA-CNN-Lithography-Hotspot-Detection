# GA-Optimized CNN for Lithography Hotspot Detection (ICCAD-12)

**Group 8 · VIT B.Tech · AI/ML for IC (BEVD402L)**

This repository documents a CNN-based lithography-hotspot classification project using the five ICCAD-12 benchmark splits (`iccad1`–`iccad5`). It compares a fixed baseline CNN with hyperparameter search using a Genetic Algorithm (GA), Random Search, and Bayesian Optimization (Optuna TPE).

> **Reproducibility note:** The clean revised workflow uses a stratified 80/20 train/validation split from the training folders only, validation balanced accuracy for search fitness, and validation-based early stopping. Historical legacy notebook cells previously used test data, so these reported test results should not be described as coming from a never-before-seen blind test set. Use only the cleaned final notebook for reproduction.

## Project highlights

- Binary classification of hotspot (HS) versus non-hotspot (NHS) image patches.
- Five benchmark sets: `iccad1` through `iccad5`.
- Input images resized to `64 × 64 × 3`.
- Seven-gene CNN chromosome: `num_blocks`, `base_filters`, `kernel_size`, `dropout`, `learning_rate`, `dense_units`, and `batch_size`.
- Gradient clipping (`clipnorm=1.0`) to reduce training instability.
- Equal search budgets: 100 evaluations per seed × 3 seeds = 300 candidate evaluations per optimizer.
- Metrics include balanced accuracy, precision, recall, specificity, F1, and TP/FP/FN/TN.

## Novelty / contribution

The contribution is the experimental workflow, not a claim that CNNs or genetic algorithms are individually new: seven-gene CNN hyperparameter search for imbalanced lithography-hotspot detection, comparison of GA with equal-budget Random Search and Bayesian Optimization, exploratory per-gene sensitivity, and evaluation across all five benchmarks. Specificity and full confusion counts are reported alongside balanced accuracy to expose false-positive/false-negative trade-offs.

## Repository layout

```text
.
├── README.md
├── src/
│   └── analyze_results.py
├── notebooks/                 # Add cleaned final notebook here
├── results/                   # Add saved CSV results/plots here
├── presentation/              # Add final PPT and speaking script here
└── requirements.txt           # Add/verify dependencies here
```

## Dataset layout

The dataset is expected locally and is **not included** in this repository:

```text
iccad-official/
├── iccad1/
│   ├── train/train_hs/
│   ├── train/train_nhs/
│   ├── test/test_hs/
│   └── test/test_nhs/
├── iccad2/  (same structure)
├── iccad3/  (same structure)
├── iccad4/  (same structure)
└── iccad5/  (same structure)
```

Set `base_path` in the notebook to your local dataset path. Do not upload benchmark images unless redistribution is explicitly permitted by the dataset terms.

## Setup

```bash
python -m pip install -r requirements.txt
jupyter lab
```

TensorFlow compatibility varies by Python version and hardware. Record the working environment versions after confirming them, for example with `python -m pip freeze > environment-lock.txt`.

## Recommended execution order

Use the cleaned final notebook and run cells in order; do not select **Run All** in an uncleaned notebook containing legacy experiments.

1. Imports, dataset path, benchmark list.
2. Image loading and stratified train/validation splitting from training folders only.
3. Class weights, baseline CNN, metrics, and validation-based early stopping.
4. Tuned CNN model builder and validation-only fitness function.
5. Use `iccad4` for optimizer comparison; cap the majority class only in candidate-training data while preserving the full validation set.
6. GA: 100 evaluations per seed × 3 seeds.
7. Random Search: 100 evaluations per seed × 3 seeds.
8. Bayesian Optimization (Optuna TPE): 100 trials per seed × 3 seeds.
9. Combine logs, calculate per-gene sensitivity and optimizer comparisons.
10. Select configuration from validation fitness only.
11. Train tuned CNN per benchmark and select epochs by validation.
12. Fix decision threshold using validation results before test evaluation.
13. Evaluate test folders and save confusion-matrix results. Do not tune on test performance.

## Optimizer results

Best validation balanced accuracy in each run:

| Method | Seed 1 | Seed 2 | Seed 3 | Mean best fitness | Std. dev. |
|---|---:|---:|---:|---:|---:|
| GA | 0.97699 | 0.97475 | 0.97138 | 0.97437 | 0.00282 |
| Random Search | 0.97755 | 0.97587 | 0.97306 | 0.97550 | 0.00227 |
| Bayesian Optimization | 0.97587 | 0.98316 | 0.98485 | **0.98129** | 0.00477 |

Each optimizer used 300 evaluations. Bayesian Optimization had the highest mean of each run's best validation fitness; three seeds do not establish statistical significance.

## Selected configuration

Best observed candidate: Bayesian Optimization, seed 3; validation balanced accuracy `0.98485`.

| Hyperparameter | Value |
|---|---:|
| `num_blocks` | 3 |
| `base_filters` | 8 |
| `kernel_size` | 5 |
| `dropout` | 0.5 |
| `learning_rate` | 0.0005 |
| `dense_units` | 128 |
| `batch_size` | 16 |
| `clipnorm` | 1.0 |

## Final test results (fixed threshold = 0.20)

These metrics are recorded from the completed notebook. Because legacy cells exposed test data during earlier experiments, do not describe them as a completely blind final evaluation.

| Benchmark | Baseline balanced accuracy | Tuned balanced accuracy | Baseline precision | Tuned precision | Baseline recall | Tuned recall | Baseline specificity | Tuned specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iccad1 | 0.8150 | 0.8119 | 0.1155 | 0.1138 | 1.0000 | 1.0000 | 0.6300 | 0.6239 |
| iccad2 | 0.9851 | 0.9648 | 0.6750 | 0.3850 | 0.9759 | 0.9478 | 0.9943 | 0.9817 |
| iccad3 | 0.9675 | 0.9618 | 0.4713 | 0.4012 | 0.9779 | 0.9806 | 0.9572 | 0.9429 |
| iccad4 | 0.9089 | 0.8912 | 0.0329 | 0.0522 | 0.9774 | 0.8701 | 0.8403 | 0.9123 |
| iccad5 | 0.7502 | 0.9407 | 0.0043 | 0.0415 | 0.9756 | 0.9268 | 0.5248 | 0.9546 |
| **Macro average** | **0.8854** | **0.9141** | **0.2598** | **0.1987** | **0.9814** | **0.9451** | **0.7893** | **0.8831** |

Macro-average F1 was `0.3427` for baseline and `0.2998` for the tuned CNN. Tuned CNN improves macro balanced accuracy by about 2.87 percentage points and specificity by about 9.37 percentage points, while macro precision, recall, and F1 decrease. False positives remain a major limitation, particularly on iccad4 and iccad5.

## Feedback points addressed

1. **Validation fitness instead of test fitness:** revised search fitness and early stopping use validation data from training directories; exclude historical legacy cells from the final code release.
2. **Larger GA budget and repeated runs:** 100 candidate evaluations × 3 seeds, 300 total.
3. **Parameter importance:** per-gene sensitivity added; surrogate permutation ranking had holdout `R² = -0.0186` and should not be treated as reliable evidence.
4. **Equal-budget comparison:** GA, Random Search, and Optuna TPE each used 300 evaluations under the same iccad4 search setup.
5. **All five benchmarks and clean metrics:** balanced accuracy, precision, recall, specificity, F1, and TP/FP/FN/TN are recorded for all five.

## Files to add next

- `notebooks/ICCAD12_revision_working.ipynb` after cleaning/annotating legacy cells and removing stale outputs.
- Raw optimizer logs: `ga_search_log.csv`, `random_search_log.csv`, `bayesian_search_log.csv`, and `combined_search_log.csv`.
- Sensitivity outputs: `per_gene_sensitivity_by_method.csv` and `per_gene_sensitivity.png`.
- Test/validation CSVs and the final PowerPoint/script under `results/` and `presentation/`.

Do not commit datasets, private credentials, local environment folders, or model weights unless intentionally publishing them. Add the official ICCAD-12 dataset/reference citation from your course or dataset provider in the report; do not invent one.
