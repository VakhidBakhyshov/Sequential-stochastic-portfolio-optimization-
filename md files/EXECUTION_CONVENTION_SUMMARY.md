# Execution and cost convention: summary

## Why we looked

Two points had to be checked before submission:

1. The final policy must carry executed holdings into the next decision, and any archived evidence in Section 7 must be labelled as archived.
2. The 10 bp cost convention in the manuscript must match the code exactly.

## What the code check found

- **Wrong state carried.** The reported run carried last month's target into the next decision, while the manuscript says executed holdings are carried. Both the optimizer's turnover penalty and the partial-execution step started from the previous target.
- **Costs under-charged.** As a result, costs were charged on about half of the actual trading: 0.50 turns per year (0.44% cumulative) were charged, against 1.02 turns (0.79%) actually traded.
- **Returns and exposure changes missing.** In six months, returns of held funds that had left the eligible set were not booked. Changes in the overlay's exposure are not charged.
- **Cost label wrong.** The code charges 10 bp on every unit bought and every unit sold. The manuscript's formula gives the same numbers, but its "one-way" label does not match. The code's rate equals 20 bp per unit of one-way turnover.

## What we changed and how

In a copy of the package (`hdrc_policyC`), we added one option to the run engine, called policy C:

- The optimizer's turnover penalty stays anchored on the previous target.
- Each month's trade starts from the executed holdings.
- Costs are 10 bp on every unit actually traded.
- The returns of all held funds are booked.

We reran all 31 paper configurations and the full analysis chain. The checks pass:

- Holdings and costs match the stated formulas to 10⁻¹⁶ in every month.
- The old setting still reproduces the reported results exactly.

## Options

| Option | What it means | HDRC Sharpe / max DD | Unmanaged base | Verdict |
|---|---|---|---|---|
| A, as reported | Previous target carried; half the trades charged | 0.859 / −6.55% | 0.660 | Contradicts the manuscript and under-charges costs. A reviewer can detect this from the weights. |
| A, corrected costs | Same decisions, with costs and returns on actual holdings | 0.849 / −6.92% | 0.645 | Accounting is right, but the paper would have to describe a strategy that tracks a target it never holds. |
| B, executed book everywhere | What the manuscript currently says | 0.627 / −10.79% | 0.625 | Worst option. The optimizer oscillates: turnover nearly doubles and 37 names are held. |
| **C, split (recommended)** | Optimizer anchored on the previous target; holdings and costs on the executed book | **0.811 / −7.07%** | 0.617 | Exact accounting, and executed holdings are carried. The manuscript only has to describe the anchor. |
| Full execution (η = 1) | Trade fully to the target every month | 0.874 / −6.80% | 0.664 | No gap between target and holdings, and the same trading as C. But it removes partial execution from the model and would be chosen after seeing results. |

A second, smaller choice concerns exposure changes. We can leave them uncharged and say so (C: 0.811), or charge them at 10 bp (C: 0.798 / −7.18%).

## Effects of switching to C

**Numbers.** Every result that comes from the production run changes:

- HDRC goes from 0.859 / −6.55% to 0.811 / −7.07%.
- The deflated Sharpe ratio goes from 96.0% to 94.6%.

These stay the same: data, the 1/N and Markowitz benchmarks, the theory, the known-law work and the CVaR–CDaR comparison.

**Conclusions.** The qualitative results all hold:

- The drawdown advantage over the fast channel, 1/N and the base is significant (P = 0.98–1.00).
- Timing accounts for about half of the drawdown cut.
- HDRC beats random timing.
- It matches the VIX rule.
- The product is the best combination.
- The two signals carry distinct information.

**Four claims need rewording:**

1. The unmanaged allocator is now below 1/N on Sharpe (0.617 vs 0.638).
2. The deflated Sharpe ratio (94.6%) is below the 95% threshold.
3. The slow signal has a small negative link to next-month return (−0.23, p = 0.04). The text can no longer say it has no return content.
4. Partial execution does not reduce trading, because full execution trades as much and earns more. The execution section can claim exact accounting, not a cost saving.

**Manuscript text beyond numbers:**

- **Turnover reference.** Define it as the previous target: notation, equations 23 and 33, Proposition 5, Appendix B.4.
- **Execution.** Write it on the risky composition, with exposure applied in full each month. This covers equations 56–61 and 81–86; restate Theorem 2.
- **Costs.** Say 10 bp per unit bought or sold, with turnover reported two-way. State whether exposure changes are charged.
- **Section 7.7.** Rewrite it so the B run becomes the mechanism test that justifies the anchor.
- **Remove the history.** Delete the "archived" and "development" wording.
- **Earlier items.** Fix S = 5,000, the 0.17 correlation sentence, the missing VIX results and the leftover image alt texts.

## Does this resolve the two points?

- **Point 1, executed holdings: yes.** Executed holdings are carried and booked. Every Section 7 result comes from the same final code, so nothing has to be labelled as archived. The manuscript must state that the optimizer's anchor is the previous target.
- **Point 2, cost convention: yes, in the code.** Costs equal 10 bp times the actual trades, checked in every month. The manuscript still needs the label fix and the exposure-cost statement.

## Next steps if we choose C

1. Switch the companion package to C and regenerate its reference tables and figures.
2. Update the manuscript numbers and text as listed above.
3. Update the slides.

Full results: `COMPARISON.md` and the `comparison/` folder.
