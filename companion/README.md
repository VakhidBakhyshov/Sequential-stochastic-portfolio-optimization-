# Companion engine

The companion engine is the second implementation of the sequential policy used in the paper. It differs from the
main engine (package root, `scripts/`) in its scenario law (21-day scenario paths, daily log returns summed to the
horizon), its predictive mean (deterministic empirical-Bayes shrinkage) and its nested walk-forward selection, and it
adds the drawdown-constrained (CDaR) allocation, the multi-signal exposure layer, the known-law benchmarks and the
theorem-to-code register. It implements the same execution and cost convention as the main engine: the optimizer is
anchored on its previous target, trading starts from the executed holdings, costs are 10 bp per unit of wealth bought
or sold, and every held position is booked.

The paper uses it for Section 7.9 (matched CVaR–CDaR comparison, Table 7, Figures 6–8, L.6–L.8), Sections 7.10–7.11
(multi-signal extension, Figures 9–16), Section 7.12 (calibration as a multi-objective problem, Figures 17–19),
Appendices E, F and I (known-law identification, Table I.1, Figures I.1–I.2) and the filtration comparison of
Table L.1 and Figures L.3–L.4 (`../extensions/universe_comparison`).

## Layout

```
scripts/            the engine: models/, optimizers/ (CVaR and CDaR programs), executions/, backtesting/ (nested
                    walk-forward selection), risk_controls/ (signal layer), validation/ (signal research, false-strategy
                    and theorem bridge), theory/ (known-law benchmarks), optimization/ (calibration analysis), runs/
scripts/configs/    bayessian_cvar_paired.yaml, bayessian_cdar.yaml, bayessian_cdar_two_signal.yaml,
                    bayessian_cvar_two_signal.yaml, advanced_smart_bayesian.yaml, analytical_benchmark.yaml,
                    optuna_calibration*.yaml (the calibration search of Section 7.12)
results/            the run folders the paper cites (trimmed to the files the tables and figures read) and the
                    analysis outputs: cvar_cdar_comparison/, companion_figures/, calibration_reanalysis/,
                    theorem_empirical_bridge/, sequential_diffusion_analysis/, analytical_benchmark/, cdar_validation/
make_companion_figures.py   Figures 6-8, 14, L.6-L.8 from the run folders
calibration_figures.py      Section 7.12 tables and Figures 17-19 from the trial table (no optuna needed)
finalize_calibration.py     full analysis of a finished search from its study database, with parameter importances,
                            and a printout of the numbers Section 7.12 quotes (needs optuna)
notebooks/          the analysis notebooks the figures were first drawn in (analysis-only mode by default)
tests/              unit tests of the engine: pytest -q tests --ignore=tests/test_optuna_calibration.py (that module
                    needs the optuna package; 61 tests pass without it)
requirements.txt    dependencies of this engine (statsmodels and plotly in addition to the main package)
```

## Running

From the package root, `python reproduce_companion.py` runs the analysis chain on the shipped run folders;
`python reproduce_companion.py --run-arms` first reruns the four arms (the paired CVaR arm takes a few minutes, the
multi-signal arm about 40 minutes, the two CDaR arms about 2.5 hours each; they run in parallel). The engine reads
data relative to its own root, so the driver copies `datasets/` into `companion/datasets` on first use.

Single steps, from this folder with `PYTHONPATH` set to it:

```
python -m scripts.runs.run -c bayessian_cvar_paired.yaml
python -m scripts.validation.signal_research --results-dir results/advanced_smart_bayesian --output-dir results/advanced_smart_bayesian/signal_research
python -m scripts.runs.compare_cvar_cdar --cvar-dir results/bayessian_cvar_paired_walk_forward --cdar-dir results/bayessian_cdar_walk_forward --output-dir results/cvar_cdar_comparison
python make_companion_figures.py --results results --out results/companion_figures
python calibration_figures.py
python -m scripts.runs.theorem_empirical_bridge --output-dir results/theorem_empirical_bridge
python -m scripts.runs.sequential_diffusion_analysis --output results/sequential_diffusion_analysis
python -m scripts.runs.analytical_benchmark --output results/analytical_benchmark --execution-eta 0.5
python -m scripts.runs.cdar_research_validation --output-dir results/cdar_validation
```

## The calibration search (Section 7.12)

```
python -m scripts.optimization.optuna_calibration --config scripts/configs/optuna_calibration.yaml
python finalize_calibration.py
```

The search runs 60 NSGA-II configurations (population 20, seed 42, 13 parameters, four objectives), each a full
nested walk-forward with research validation, as one process: a trial takes about 7.5 minutes and writes about
420 MB under `results/calibration_runs/`, so the whole search takes about 7.5 hours and 25 GB. Run it as a single
process: the sampler is seeded inside each process, so parallel workers sharing the study database would propose
identical configurations. A trial is deterministic (identical configurations give identical objectives). Optuna
5.0.0 was used; parameter importances use the functional-ANOVA evaluator explicitly. The shipped outputs under
`../extensions/calibration/results/advanced_smart_bayesian_risk_return` are the trial table, Pareto set, policy
comparison, parameter ranges and importances, the three figures, the selected configurations (`best_configs/`), the
60 generated trial configurations (`trial_configs/`) and the study database; `calibration_figures.py` rebuilds the
tables and figures from the trial table without optuna.
