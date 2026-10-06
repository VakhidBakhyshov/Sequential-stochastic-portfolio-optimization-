# Where each table and figure comes from

Scripts run from the package root unless stated; `results/<arm>` folders come from `run_arms.py`, and
`reproduce.py` runs the analysis scripts in order. Reference copies of every output are under `reference/`.

| Paper item | Script | Output |
|---|---|---|
| Tables 1–4, J.1, K.2 (notation, structural results, specification, benchmarks, audit record, controls) | text only | – |
| Section 6.1 eligible set, Figure 2 | `build_universe.py` (liquidity screen), `prepare_data.py`, `fig_universe_and_decorr.py` | `datasets/excel/last_filtered_weights.xlsx`, `new_etf_returns.csv`, `new_etf_ewma.csv`, `business_dates.xlsx`, `universe_eligible_count.png` |
| Section 7.1 signal validity and complementarity, Figure 3, Figure L.1 | `signal_validity.py`, `paper_measurements.py`, `complementarity_diagnostics.py` | `signal_validity.csv`, `fig_exposure_paths.png`, `complementarity_*.csv`, `fig_conditional_risk.png` |
| Section 7.3 single-channel and combined arms | `fast_voltarget.py`, `cvar_target_overlay.py`, `overlay_arms.py main_dyn_strong` | `results/main_voltarget`, `results/main_cvartarget`, `results/main_combined` (`pnl.csv`, `exposure.csv`), `arms_main_dyn_strong.csv` |
| Table 5, Section 7.2 paired bootstraps, Table K.1 (two-signal row) | `verify_all.py` (N_TRIALS=64), `benchmarks.py` | `verified_results.csv`, `benchmarks_vs_strategy.csv`, console |
| Section 6.5 tail comparisons, Sharpe-difference power, daily-path check | `tail_bootstrap.py` | `tail_bootstrap.csv`, `daily_path.csv` |
| Figures 4(a), 4(b), 5, L.2, L.5 | `build_results.py`, `build_drawdown.py`, `plot_dynamic_params.py`, `make_figures.py` | `res_performance.png`, `res_drawdown.png`, `res_turnover.png`, `res_dynamic_params.png`, `res_composition_dynamic.png` |
| Section 7.4 and Table 6 (matched exposure, random timing) | `controls.py`, `paper_measurements.py` | `controls_results.csv`, console |
| Section 6.3 option-implied benchmark | `vix_experiment.py` | `vix_experiment_results.csv` |
| Section 7.5 and Table L.3 (construction diagnostics, generators) | `paper_measurements.py`, `compare_fhs_arm.py`, `fhs_experiment.py`, `block_bootstrap_experiment.py`, `overlay_arms.py main_dyn_strong --cvar-col cvar_model_raw` | `fhs_arm_comparison.csv`, `fhs_experiment_results.csv`, `block_bootstrap_results.csv`, `arms_main_dyn_strong_uncentred.csv`, console |
| Section 7.6 universes (equal weight, HDRC across universes) | `ew_universe_ablation.py`, `universe_compare.py` | `ew_universe_ablation.csv`, `universe_comparison.csv` |
| Section 7.6 and Table L.1, Figures L.3–L.4 (filtration comparison) | `extensions/universe_comparison/` (companion material, shipped as is) | its result folders |
| Section 7.7 and Table L.2 (mechanism experiments) | `run_arms.py` arms `mech_execstate_main_dyn_strong`, `main_dyn_strong_eta1`, `main_dyn_strong_fixeduniv`; `mech_summary.py` | `mech_experiments.csv` |
| Section 7.8 costs and break-even | `paper_measurements.py`, `build_results.py` | console |
| Section 7.9, Table 7 (matched CVaR–CDaR comparison) | `companion`: arms `bayessian_cvar_paired.yaml`, `bayessian_cdar.yaml`; `scripts/runs/compare_cvar_cdar.py` (via `reproduce_companion.py`) | `companion/results/cvar_cdar_comparison/` |
| Figures 6–8 (CDaR path-risk state, drift contribution, wealth of the CDaR arms), Figures L.6–L.8 | `companion/make_companion_figures.py` on arms `bayessian_cvar_paired`, `bayessian_cdar`, `bayessian_cdar_two_signal` | `companion/results/companion_figures/` |
| Sections 7.10–7.11, Figures 9–12 (signal dependence), 15–16 (calendar returns, monthly heatmap), 14 (execution fraction and cost) | `companion`: arm `advanced_smart_bayesian.yaml`, `scripts/validation/signal_research.py`, the engine's metric artifacts, `make_companion_figures.py` | `companion/results/advanced_smart_bayesian/` (`signal_research/`, `annual_returns_vs_spy.png`, `monthly_returns_heatmap.png`), `companion_figures/` |
| Figure 13 (companion framework against the main study) | `extension_comparison.py` | `fig_extension.png`, `extension_comparison.csv` |
| Section 7.12, Figures 17–19 (calibration frontier, parameter effects, metric distributions) | `companion/scripts/optimization/optuna_calibration.py` (the 60-trial search, about 7.5 hours), `companion/calibration_figures.py` on its trial table | `extensions/calibration/results/`, `companion/results/calibration_reanalysis/` |
| Appendix I, Table I.1, Figures I.1–I.2 (theorem-to-computation checks) | `companion`: `scripts/runs/theorem_empirical_bridge.py` | `companion/results/theorem_empirical_bridge/` |
| Appendices E–F known-law checks | `companion`: `scripts/runs/sequential_diffusion_analysis.py`, `analytical_benchmark.py`, `cdar_research_validation.py` | `companion/results/`, `reference/known_law/` |
| Appendix K trial ledger (64 trials: paper arms, scenario-count x history-rule x seed grid, signal constructions) | `run_arms.py --grid`, `grid_table.py`, `seed_table.py`, `grid_composition.py` | `grid_results.csv`, `seed_sensitivity.csv`, `grid_composition.csv` |
| Table K.1 (multi-signal row), Figure K.1 (false-strategy benchmark) | `companion` arm `advanced_smart_bayesian` (research-validation outputs) | `companion/results/advanced_smart_bayesian/` (`research_validation_summary.*`, `false_strategy_heatmap_standardized.png`) |
| Figure 1 (pipeline) | `fig_pipeline.py` | `fig_pipeline.png` |

Additional outputs kept for completeness: `extra_ratios.csv` (beta, Treynor, downside deviation, pain ratio),
`subperiod_table.csv`, `experiment5_metrics.csv` and `overlay_3way_metrics.csv` (combination forms), `extension_comparison.csv`.
Output names carry no date suffixes in this package.
