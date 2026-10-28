# When does early elimination waste a tuning budget?

**Research in progress. No model result or winning strategy is claimed yet.**

An analytics team can use a limited search allowance to screen more models cheaply, or fit fewer settings to completion. This study will compare those decisions on four exact public numerical benchmark versions, retaining every trial and independently checking MSE, promotion decisions, compute accounting and later test results.

The [committed protocol](PROTOCOL.md) fixes the objective, 36-setting model space, shared 18-setting candidate pools, six seeds, grouped fit/validation/test boundaries and an equal 540-requested-iteration allowance per strategy. A separate 1,620-iteration full-pool audit measures what early elimination discarded; that additional research computation cannot be advertised as a free search saving. Wall time is measured separately because boosting iterations are not equal FLOPs or cost.

[Sources and rights](SOURCES.md) identify the exact OpenML versions and their original creators. The 19-feature-group Zurich subset is a declared limitation, not thousands of independent test cases. Panels preserve whole identical-predictor groups. The method comparison is conditional on these benchmark subsets and overlapping seeds; it does not validate deployment forecasts or universal strategy superiority.

Manuscript drafts, raw source files and credentials remain outside public Git history. Executed results, figures and reproducibility evidence will replace this progress note only after their checks pass.
