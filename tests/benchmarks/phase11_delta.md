### Phase 11 N=20 benchmark delta — v0.7.17 → 0.8.0

- Before: `fa63011c4b` (v0.7.17)
- After: `b0990e5395` (v0.7.17)
- N=20 trials per cell. All times in milliseconds.

### Construction sweep (wall-clock build)

| test | scale | mode | v0.7.17 p50 | 0.8.0 p50 | Δ p50 | 0.8.0 p95 | 0.8.0 σ | gate |
|---|---:|---|---:|---:|---:|---:|---:|---|
| construction | 100 | disk | – | 0.812 | – | 0.867 | 0.062 |  |
| construction | 100 | mapped | 0.579 | 0.477 | -17.5% | 0.508 | 0.020 | ok |
| construction | 100 | memory | 0.496 | 0.511 | +3.0% | 1.053 | 0.287 | flag |
| construction | 1000 | disk | – | 2.299 | – | 2.331 | 0.059 |  |
| construction | 1000 | mapped | 2.172 | 1.828 | -15.9% | 2.102 | 0.162 | ok |
| construction | 1000 | memory | 1.741 | 1.699 | -2.4% | 1.775 | 0.052 | ok |
| construction | 10000 | disk | – | 18.072 | – | 19.147 | 0.564 |  |
| construction | 10000 | mapped | 18.603 | 15.497 | -16.7% | 16.187 | 0.396 | ok |
| construction | 10000 | memory | 14.019 | 12.447 | -11.2% | 12.756 | 0.173 | ok |
| construction | 50000 | disk | – | 107.840 | – | 126.812 | 10.653 |  |
| construction | 50000 | mapped | 145.076 | 113.365 | -21.9% | 114.134 | 0.730 | ok |
| construction | 50000 | memory | 80.716 | 70.935 | -12.1% | 71.718 | 0.458 | ok |

### Query primitives at 10k nodes

| test | mode | v0.7.17 p50 | 0.8.0 p50 | Δ p50 | 0.8.0 p95 | 0.8.0 σ | gate |
|---|---|---:|---:|---:|---:|---:|---|
| aggregation | disk | – | 1.543 | – | 1.647 | 0.045 |  |
| aggregation | mapped | 1.441 | 1.363 | -5.4% | 1.456 | 0.045 | ok |
| aggregation | memory | 1.442 | 1.366 | -5.3% | 1.465 | 0.069 | ok |
| describe | disk | – | 5.284 | – | 5.512 | 0.155 |  |
| describe | mapped | 3.236 | 2.457 | -24.1% | 2.572 | 0.077 | ok |
| describe | memory | 3.037 | 2.395 | -21.1% | 2.453 | 0.066 | ok |
| find_20x | disk | – | 10.660 | – | 10.861 | 0.162 |  |
| find_20x | mapped | 10.377 | 9.053 | -12.8% | 9.275 | 0.202 | ok |
| find_20x | memory | 4.436 | 4.642 | +4.7% | 4.808 | 0.142 | flag |
| multi_predicate | disk | – | 0.708 | – | 0.854 | 0.065 |  |
| multi_predicate | mapped | 0.609 | 0.636 | +4.3% | 0.740 | 0.049 | ok |
| multi_predicate | memory | 0.660 | 0.664 | +0.7% | 0.780 | 0.073 | ok |
| order_by_limit | disk | – | 0.098 | – | 0.104 | 0.003 |  |
| order_by_limit | mapped | 0.089 | 0.086 | -2.5% | 0.091 | 0.003 | ok |
| order_by_limit | memory | 0.087 | 0.088 | +1.4% | 0.101 | 0.005 | ok |
| pagerank | disk | – | 5.105 | – | 5.573 | 0.248 |  |
| pagerank | mapped | 5.375 | 4.426 | -17.7% | 4.721 | 0.233 | ok |
| pagerank | memory | 5.185 | 4.276 | -17.5% | 4.531 | 0.203 | ok |
| pattern_match | disk | – | 15.441 | – | 15.857 | 0.294 |  |
| pattern_match | mapped | 13.664 | 5.435 | -60.2% | 5.557 | 0.101 | ok |
| pattern_match | memory | 13.756 | 5.417 | -60.6% | 6.054 | 0.612 | ok |
| schema | disk | – | 0.001 | – | 0.001 | 0.000 |  |
| schema | mapped | 0.001 | 0.001 | -12.1% | 0.002 | 0.000 | ok |
| schema | memory | 0.001 | 0.001 | -7.5% | 0.002 | 0.002 | ok |
| simple_filter | disk | – | 0.687 | – | 0.706 | 0.012 |  |
| simple_filter | mapped | 0.578 | 0.596 | +3.1% | 0.624 | 0.014 | ok |
| simple_filter | memory | 0.534 | 0.546 | +2.3% | 0.564 | 0.012 | flag |
| two_hop_10x | disk | – | 0.057 | – | 0.062 | 0.002 |  |
| two_hop_10x | mapped | 0.073 | 0.055 | -24.9% | 0.056 | 0.002 | ok |
| two_hop_10x | memory | 0.071 | 0.054 | -23.8% | 0.055 | 0.000 | ok |

### Summary

- **Wins** (14): queries at least 2 % faster.
  - `pattern_match_10000_memory`: -60.6%
  - `pattern_match_10000_mapped`: -60.2%
  - `two_hop_10x_10000_mapped`: -24.9%
  - `describe_10000_mapped`: -24.1%
  - `two_hop_10x_10000_memory`: -23.8%
  - `describe_10000_memory`: -21.1%
  - `pagerank_10000_mapped`: -17.7%
  - `pagerank_10000_memory`: -17.5%
  - `find_20x_10000_mapped`: -12.8%
  - `schema_10000_mapped`: -12.1%
- **Flags** (4): queries at least 2 % slower.
  - `find_20x_10000_memory`: +4.7%
  - `multi_predicate_10000_mapped`: +4.3%
  - `simple_filter_10000_mapped`: +3.1%
  - `simple_filter_10000_memory`: +2.3%
