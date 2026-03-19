<p align="center">
  <h1 align="center">🧪 ExperimentLab</h1>
  <p align="center">
    Open-source experimentation platform for data scientists.<br/>
    From power analysis to causal inference — the full experiment lifecycle.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/tests-147%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit" alt="Streamlit">
</p>

---

## What Is This?

ExperimentLab is a **complete experimentation toolkit** that covers every stage of running experiments — whether you're doing A/B tests, geo experiments, or causal studies.

Most teams cobble together scripts for power analysis, copy-paste z-test code, and have no systematic way to validate assumptions. ExperimentLab gives you a single, tested, interactive platform that handles it all.

## Who Is This For?

- **Data Scientists** running A/B tests and geo experiments at scale
- **Product Managers** who want to understand experiment design tradeoffs
- **Researchers** who need rigorous causal inference methods
- **Students** learning experiment design and statistical testing

## Quick Start

```bash
# Clone and install
git clone https://github.com/NiShITa-code/experimentlab.git
cd experimentlab
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

# Run tests
pytest tests/ -v
```

## Features

### Pre-Experiment

| Feature | Description |
|---------|-------------|
| **Power Calculator** | Sample size for binary & continuous metrics, MDE curves, duration estimates |
| **Geo Allocator** | Stratified market assignment with covariate balance checking |
| **Metric Validation** | Validate metric definitions — nulls, types, sensitivity (MDE at given N) |

### During Experiment

| Feature | Description |
|---------|-------------|
| **Sequential Testing** | O'Brien-Fleming & Pocock spending functions for valid early stopping |
| **SRM Detection** | Chi-squared test for sample ratio mismatch (randomization bugs) |
| **Live Boundaries** | Visual boundary tracking across interim analyses |

### Post-Experiment

| Feature | Description |
|---------|-------------|
| **Frequentist A/B** | Two-proportion z-test, Welch's t-test, CUPED variance reduction |
| **Bayesian A/B** | Beta-Binomial & Normal conjugate models, ROPE, expected loss |
| **Diff-in-Differences** | 2×2 DiD with parallel trends test and event study |
| **Synthetic Control** | Abadie (2010) method with placebo tests and RMSPE validation |
| **Sentiment Analysis** | VADER-based pre/post sentiment shift with word frequency analysis |
| **Privacy Audit** | k-anonymity checks, Laplace/Gaussian DP noise, utility-privacy tradeoff |

## Architecture

```
experimentlab/
├── app.py                    # Streamlit dashboard (10 tabs)
├── config/
│   └── config.yaml           # All parameters in one place
├── src/
│   ├── config.py             # Frozen dataclass config loader
│   ├── data/
│   │   ├── schema.py         # Typed data contracts (enums, dataclasses, validators)
│   │   └── simulator.py      # A/B test, geo experiment, feedback data generators
│   ├── design/
│   │   ├── power.py          # Sample size, MDE, power analysis
│   │   └── geo_allocator.py  # Market allocation with balance checking
│   ├── analysis/
│   │   ├── frequentist.py    # z-test, t-test, CUPED, SRM
│   │   ├── bayesian.py       # Beta-Binomial, Normal-Normal, ROPE, expected loss
│   │   ├── sequential.py     # Group sequential testing (OBF, Pocock)
│   │   ├── diff_in_diff.py   # DiD with parallel trends + event study
│   │   └── synthetic_control.py  # Abadie (2010) + placebo tests
│   ├── nlp/
│   │   └── feedback.py       # VADER sentiment + word shift analysis
│   ├── privacy/
│   │   └── aggregation.py    # k-anonymity, differential privacy
│   ├── metrics/
│   │   └── builder.py        # Metric validation framework
│   └── utils/
│       └── logging.py        # Structured logging
├── tests/                    # 147 tests across 11 test files
├── requirements.txt
└── setup.py
```

## Dashboard

The Streamlit dashboard has **10 interactive tabs**:

1. **Overview** — Quick demo, project stats
2. **Power Calculator** — Interactive power curves and sensitivity tables
3. **Geo Allocator** — Market assignment with balance visualization
4. **Frequentist A/B** — Full test with CI plots and segment breakdown
5. **Bayesian A/B** — Posterior distributions, ROPE, decision matrix
6. **Sequential Testing** — Boundary plots with interim analysis results
7. **Causal Methods** — DiD event study and synthetic control gap plots
8. **Sentiment Analysis** — Pre/post sentiment comparison with word shifts
9. **Privacy Audit** — k-anonymity + privacy-utility tradeoff curves
10. **Metric Validation** — Suite-level metric health checks

## Methods & References

| Method | Reference |
|--------|-----------|
| Diff-in-Differences | Angrist & Pischke (2009), Chapter 5 |
| Synthetic Control | Abadie, Diamond & Hainmueller (2010), JASA |
| Bayesian A/B Testing | Kruschke (2013), "Bayesian estimation supersedes the t-test" |
| CUPED | Deng et al. (2013), "Improving the Sensitivity of Online Controlled Experiments" |
| Sequential Testing | Jennison & Turnbull (1999), Group Sequential Methods |
| Differential Privacy | Dwork & Roth (2014), "The Algorithmic Foundations of Differential Privacy" |
| VADER Sentiment | Hutto & Gilbert (2014), ICWSM |

## Testing

```bash
# Run all 147 tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and contribution areas.

## License

MIT License — see [LICENSE](LICENSE) for details.
