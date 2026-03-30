"""ExperimentLab — Open-source experimentation platform."""

from setuptools import setup, find_packages

setup(
    name="experimentlab",
    version="1.0.0",
    description="End-to-end experimentation platform for data scientists",
    author="NiShITa-code",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.11",
        "pyyaml>=6.0",
        "vaderSentiment>=3.3",
        "streamlit>=1.32",
        "plotly>=5.18",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-cov>=4.1"],
    },
)
