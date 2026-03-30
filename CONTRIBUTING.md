# Contributing to ExperimentLab

Thank you for considering contributing to ExperimentLab! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/NiShITa-code/experimentlab.git
cd experimentlab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v
```

## Code Style

- Use type hints for all function signatures
- Write docstrings (NumPy style) for all public functions
- Keep functions focused — one responsibility per function
- Use dataclasses for structured outputs

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new analysis method
fix: correct p-value calculation in sequential testing
test: add tests for Bayesian A/B
docs: update README with geo allocator example
refactor: simplify power analysis interface
```

## Testing

- Write tests for every new feature or bug fix
- Aim for > 90% coverage on new code
- Tests should be deterministic (use seeds for random operations)
- Run the full suite before submitting a PR: `pytest tests/ -v`

## Areas for Contribution

- **New analysis methods**: Multi-armed bandits, switchback experiments, regression discontinuity
- **Dashboard improvements**: Better visualizations, CSV upload, export reports
- **Documentation**: Tutorials, method explanations, example notebooks
- **Performance**: Optimize simulation speed, parallel processing
- **Testing**: Edge cases, property-based tests, integration tests

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for your changes
3. Ensure all tests pass
4. Submit a PR with a clear description of the changes

## Questions?

Open an issue on GitHub — we're happy to help!
