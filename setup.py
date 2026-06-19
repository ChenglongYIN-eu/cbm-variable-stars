from setuptools import setup, find_packages

setup(
    name="cbm_variable_stars",
    version="0.1.0",
    description="Concept Bottleneck Models for Variable Star Classification",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "torch>=2.1.0",
        "astroquery>=0.4.7",
        "astropy>=6.0.0",
        "pyvo>=1.5",
        "xgboost>=2.0.0",
        "pyarrow>=14.0.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "loguru>=0.7.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "viz": ["umap-learn>=0.5.4"],
        "explain": ["shap>=0.43.0"],
        "dev": ["pytest>=7.4.0"],
    },
)
