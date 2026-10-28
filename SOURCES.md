# Exact sources and reuse boundary

This study uses four active dataset versions from [OpenML numerical-regression suite 336](https://www.openml.org/search?type=study&id=336), associated with [Grinsztajn, Oyallon and Varoquaux's tabular benchmark](https://github.com/LeoGrin/tabular-benchmark). The suite itself is marked in preparation; individual source records are active. These are transformed benchmark tables, not untouched original releases.

| OpenML ID/version | Name | Role fixed before model fitting | Declared dataset terms |
|---|---|---|---|
| [44140/v6](https://www.openml.org/d/44140) | diamonds | Development | public domain |
| [44144/v8](https://www.openml.org/d/44144) | house_sales | Development | CC0 Public Domain |
| [44148/v4](https://www.openml.org/d/44148) | superconduct | Held dataset | CC BY 4.0 |
| [45034/v3](https://www.openml.org/d/45034) | delays_zurich_transport | Held dataset | CCZero |

Metadata bytes, exact ARFF URLs and hashes are retained in [data/metadata](data/metadata) and [the manifest](data/manifest.json). Panel files are explicitly modified, bounded derivatives: numeric parsing, exact full-row value deduplication, grouping identical predictor vectors, and a 6,000-row cap that preserves whole groups. Targets retain the distributed benchmark values. An original-source physical-unit correspondence has not been established; report errors in benchmark target units and normalized MSE. No endorsement by any source author is implied.

Superconductivity attribution: Kam Hamidieh (2018), *Superconductivty Data*, UCI Machine Learning Repository, [DOI 10.24432/C53P47](https://doi.org/10.24432/C53P47), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [UCI's source description](https://archive.ics.uci.edu/dataset/464/superconductivty+data) includes chemical-formula information in a companion file; this numerical OpenML table does not include those family identities. Grouping identical predictors cannot establish independence across related materials.

The original diamond description credits the classic price/attribute table; the property source describes King County sales in May 2014 to May 2015. The Zurich source credits [OpenDataDayZurich2016/ODDPredictDelays](https://github.com/OpenDataDayZurich2016/ODDPredictDelays), for transport records collected 30 October to 27 November 2016. Dates and entity membership are not reconstructed from benchmark numeric tables. No deployment forecasting claim is made.

Source profiling reveals an important scope limitation: the bounded Zurich panel contains 5,911 distinct full rows but only 19 predictor groups; individual test seeds contain only2--5 feature groups. Millions of identical full-row values in the full source are deduplicated as specified, without claiming they were erroneous duplicate events. These decisions materially change the empirical frequency distribution. That dataset must be interpreted as a small group-transfer stress case, not thousands of independent deployment observations or a representative transport sample.

Method context: [Li et al., Hyperband, JMLR18(185),2018](https://jmlr.org/papers/v18/16-558.html); [scikit-learn1.7 search documentation](https://scikit-learn.org/1.7/modules/grid_search.html). Random search, halving and finite-budget selection are established. This original experiment uses a declared cold-fit iteration ledger and a separate full-pool validation audit. It is not a full Hyperband implementation or a new optimizer.

Raw ARFF files remain local and ignored to keep the repository bounded. Eligible derived panels, metadata, original code and derived numerical evidence are public. Source papers remain excluded. The independent research manuscript is shared in the paper directory and has not been peer reviewed.
