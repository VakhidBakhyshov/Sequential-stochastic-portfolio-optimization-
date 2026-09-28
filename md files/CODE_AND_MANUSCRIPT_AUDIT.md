# Code + manuscript audit and final-file decision

## Scope

I reviewed the uploaded framework as a codebase rather than treating `*-upd.py` as automatically newer/better. The archive contains **143 Python files, 7 notebooks, 122 YAML/YML files**, and seven direct `file.py` / `file-upd.py` pairs. I compared each pair, traced the main monthly pipeline, execution state, transaction costs, nested selection, predictive scenarios, CVaR/CDaR, result statistics and exposure overlays against the manuscript and `EXECUTION_CONVENTION_SUMMARY.md`.

The most important finding is that the final code should **not** be either `run.py` or `run-upd.py` unchanged. The strongest final version is a **merge based on the current non-`-upd` code**, with the Policy-C execution/accounting corrections from the markdown.

---

## 1. Which duplicate file should be final?

| Pair | Final choice | Reason |
|---|---|---|
| `scripts/runs/run.py` vs `run-upd.py` | **MERGE, starting from `run.py`** | Current `run.py` contains the richer final research pipeline: scenario-path handling, centered CVaR/CDaR diagnostics, trial ledger, risk-matrix audit, post-processing and current validation hooks. `run-upd.py` contains an older state-reference switch, but neither old version implements Policy C correctly because both effectively use one state for two different economic roles. The patched `run.py` separates previous-target optimizer anchor from actual executed holdings. |
| `scripts/models/base.py` vs `base-upd.py` | **Use `base.py`** | Current file treats scenario rows as a predictive distribution and compares them with the realized holding-period outcome. The `-upd` version contains older KDE/WAIC/R² logic and treats objects less consistently with the current scenario architecture. |
| `scripts/models/bayessian.py` vs `bayessian-upd.py` | **Use `bayessian.py`** | Current file supports horizon-aware scenario generation, daily covariance, reproducible posterior mean construction, ordered scenario paths for CDaR, risk-matrix snapshots and scenario-return-type handling. Those are required by the manuscript's CVaR/CDaR and structural diagnostics. |
| `scripts/optimizers/base.py` vs `base-upd.py` | **Use `base.py`** | Current file has the more precise sample-CVaR audit calculation (including fractional tail mass) and stronger LP/binding/shadow-price diagnostics, which are needed for the budget-degeneracy claims. |
| `scripts/backtesting/walk_forward_engine.py` vs `walk_forward_engine-upd.py` | **Use patched `walk_forward_engine.py`** | Current file preserves CDaR scenario paths, candidate-attempt/trial audit and robust selection diagnostics. Patch additionally splits optimizer reference from execution state and prevents post-execution re-normalization. |
| `scripts/backtesting/engine_integration.py` vs `engine_integration-upd.py` | **Use patched `engine_integration.py`** | Current file includes smart-signal integration and richer audit outputs. Patch passes the two Policy-C states separately. |
| `scripts/results/stats.py` vs `stats-upd.py` | **Use `stats.py`** | This is not close: current file is the full research version (PSR/DSR, extended metrics, train/test, rolling Sharpe/Sortino, annual/monthly plots, false-strategy diagnostics, artifact export). `stats-upd.py` is a much smaller older subset. |

### Practical cleanup rule

After integrating this patch and reproducing the final run, move all seven `*-upd.py` files to an archive folder or delete them from the publication branch. Keeping both names in the active source tree makes it too easy to run the wrong implementation and undermines reproducibility.

---

## 2. Main execution/accounting problem

The framework had three objects that must be distinct:

1. **previous target**: useful as the optimizer's regularization/turnover anchor;
2. **previous executed holdings**: the portfolio economically held at the start of the new trade;
3. **new target**: the current optimization output.

Using one `w_previous` variable for all three roles creates the exact inconsistency described in the markdown. The patched pipeline uses:

```text
optimizer anchor at t  = target_{t-1}
trade starting state   = executed_{t-1}
executed_t              = executed_{t-1} + eta_t * (target_t - executed_{t-1})
optimizer anchor at t+1 = target_t
trade starting state t+1= executed_t
```

This is Policy C.

### Why not Policy B (executed holdings everywhere)?

It is internally neat but changes the optimizer's turnover reference and, according to the supplied rerun summary, causes materially more turnover/oscillation. The state that determines *economic trading* does not have to be identical to the reference point used in a regularized optimization objective. The paper must be explicit about this distinction.

---

## 3. Transaction-cost convention

The execution code computes

```python
turnover = sum(abs(w_exec - w_prev_exec))
cost = c_bps * turnover
```

That quantity is **gross/two-way traded notional (L1 turnover)**: purchases and sales are both counted. If a fully invested portfolio sells 20% and buys 20%, L1 turnover is 40%, while conventional one-way turnover is 20%.

Therefore, with `c_bps = 0.001`:

- the code charges **10 bp per unit bought or sold**;
- equivalently, for a self-financing fully invested rebalance, it is **20 bp per unit of conventional one-way turnover**.

The manuscript should not call `sum(abs(delta w))` “one-way turnover.” The patch uses the unambiguous language **gross traded notional / L1 turnover**.

---

## 4. Missing-return bug for assets leaving the feasible set

This was a real accounting issue. Previously, next-month returns were loaded only for the *current eligible set*. With partial execution, an ETF that leaves the eligible set can still have a positive executed holding while it is being liquidated. Its return must therefore still enter realized PnL.

The patched `run.py` now builds:

- `future_returns`: current eligible names, used by the model/evaluation logic;
- `future_returns_execution`: the complete portfolio-state ticker list in the same order as the full weight vector, used for actual PnL.

`BaseExecution` detects the full-state frame and books all economically held positions. Required exits are also automatically included in turnover because the target weight for an ineligible name is zero while its previous executed holding may be positive.

---

## 5. Exposure-overlay trading cost

The original `overlay_arms.py` changes risky exposure but does not charge any incremental cost for changing the exposure multiplier. The manuscript currently says exposure changes generate trading, so code and text were not aligned.

The patched script adds an explicit option:

```bash
python overlay_arms.py <run_folder> --charge-exposure-costs --exposure-cost-bps 0.001
```

The separate exposure-channel turnover is

```text
|k_t - k_{t-1}|
```

and cost is `rate * |delta k|`.

This is intentionally reported as a **separate scalar exposure-channel cost**, not as an exact additive decomposition of full-vector L1 turnover when both composition and exposure change simultaneously.

### Recommended publication handling

Run and report **both**:

- Policy C with composition/execution costs only;
- Policy C + exposure-overlay cost at 10 bp.

Choose the first as baseline only if the paper explicitly states that overlay implementation costs are excluded from the baseline and the second is a robustness check. If the paper says exposure changes are charged in the baseline, then use the second as the baseline.

---

## 6. Additional harmful/important issue found: `strict_nested_engine` was ignored

Two publication YAML files declare:

```yaml
strict_nested_engine: true
```

but the original `run.py` caught every nested-engine exception and silently fell back to the legacy direct optimization path. That means a supposedly “strict” production run could contain a mixture of two selection procedures without stopping.

The patched `run.py` now raises immediately when `strict_nested_engine: true`. This is important for a publication run because otherwise the trial ledger, selection chronology and reported algorithm can differ across months.

---

## 7. Nested-engine partial-execution bug fixed

Inside `walk_forward_engine.py`, candidate weights were partially executed and then normalized back to one. That can silently undo the economic meaning of partial execution when total risky mass is below one (for example after required exits or when a cash sleeve exists).

The patch removes this post-execution renormalization. Target composition is normalized when it is constructed; the executed state is left as the affine state transition.

---

## 8. Manuscript/code inconsistencies that still require a new empirical rerun

The code patch makes the implementation internally coherent, but **old reported numerical results do not become valid automatically**. The manuscript must be regenerated from a fresh final-policy run.

The supplied markdown already shows that Policy C changes the production numbers. Therefore do not keep the old Table 5 / Section 7 production values and simply change the prose.

At minimum, regenerate:

- Table 5 main performance metrics;
- Table 6 exposure attribution;
- fast/slow/HDRC paths and drawdown plots;
- PSR/DSR from the exact final trial ledger;
- transaction-cost/turnover tables and break-even cost;
- fast/slow signal correlation diagnostics;
- VIX comparison numbers;
- execution mechanism section;
- all downstream slides/tables that use the production run.

The matched CVaR–CDaR experiment, known-law theory and purely structural mathematics only stay unchanged if they are genuinely produced by an independent matched branch that does not use the corrected production-run accounting outputs.

---

## 9. Manuscript wording that should change for Policy C

The current text says the executed portfolio is also the turnover reference passed into the next optimizer. Under the recommended split convention, that is no longer the definition.

Use this distinction throughout notation, Model §3.4/3.6, Theorem 2 discussion, Algorithm Appendix J and execution results:

```text
q_{t-1}: previous target composition used as the optimizer's turnover/reference anchor.
x_{t-1}^{exec}: actual risky holdings from which current trading, realized returns and transaction costs are computed.
x_t^{tar}: current total-wealth/risky target.
x_t^{exec}: actual holdings after partial execution.
```

Do **not** say “the optimizer starts from executed holdings” if using Policy C. It does not. The economic trade starts from executed holdings; the optimizer penalty is anchored on the previous target.

Also change transaction-cost wording from “10 bp per unit of one-way turnover” to something like:

> Transaction cost equals 10 bp times gross traded notional, where gross traded notional is the L1 change in implemented risky holdings and therefore counts buys and sells separately.

If using the optional overlay-cost implementation, add a separate sentence defining `|k_t-k_{t-1}|` and the applied rate.

---

## 10. Files in this patch

### Changed / new final-policy files

- `scripts/runs/run.py`
- `scripts/backtesting/walk_forward_engine.py`
- `scripts/backtesting/engine_integration.py`
- `scripts/executions/base.py`
- `scripts/executions/accounting.py` **(new)**
- `overlay_arms.py`
- `tests/test_execution_convention.py` **(new)**
- `notebooks/execution_transaction_cost_audit.ipynb` **(new)**

### Canonical files selected from duplicate pairs (kept as-is from the stronger current implementation)

- `scripts/models/base.py`
- `scripts/models/bayessian.py`
- `scripts/optimizers/base.py`
- `scripts/results/stats.py`

Do not replace these four with their `*-upd.py` versions.

---

## 11. Validation performed here

- All changed Python files compile successfully when merged into the uploaded framework.
- The whole `scripts/` tree compiles after the patch.
- New execution-convention regression tests: **5 passed**.
- The tests check exact partial-execution contraction/turnover identities, Policy-C state separation, the 10-bp/L1 cost convention, required liquidation, and the separate exposure-turnover channel.

A full 83-month numerical rerun was **not** treated as completed in this review. That rerun is required before updating empirical values in the manuscript, because the corrected state/cost convention is intentionally different from the archived production run.

---

## Final code-branch recommendation

Use the non-`-upd` implementation as the base branch, apply this patch, archive the `*-upd.py` files, set `strict_nested_engine: true` for the publication run, and regenerate all production results once under a single frozen convention. Do not mix archived Policy-A results, Policy-B state semantics, and Policy-C code in the final article.
