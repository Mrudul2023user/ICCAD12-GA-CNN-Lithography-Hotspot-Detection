# Results

Put exported result CSVs, plots, and compact configuration files here.

Recommended outputs include:
- `ga_search_log.csv`
- `random_search_log.csv`
- `bayesian_search_log.csv`
- `combined_search_log.csv`
- `per_gene_sensitivity_by_method.csv`
- `per_gene_sensitivity.png`
- `revised_baseline_validation_results.csv`
- `final_tuned_validation_results.csv`
- `final_baseline_test_results.csv`
- `final_tuned_test_results.csv`
- `final_test_baseline_vs_tuned.csv`
- `final_test_comparison_summary.csv`
- `validation_comparison_by_threshold.csv`

Ensure CSV column names agree with `src/analyze_results.py`. Label results as validation or test metrics, and do not use test results for model selection. The currently recorded test data was exposed in historical legacy experiments; document that limitation.
