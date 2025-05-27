from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nvdya_kit",
    version="0.1.0",
    author="Dneth Nethsara and DPC Media Unit",
    author_email="support@nvdyakit.org",
    description="A comprehensive GPU-accelerated machine learning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dneth/nvdya-kit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "joblib>=0.16.0",
        "tqdm>=4.47.0",
        "numba>=0.50.0",
        "cupy-cuda11x>=9.0.0;platform_system!='Darwin'",  # GPU support for CUDA
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.0.0",
            "flake8>=3.8.0",
            "sphinx>=3.1.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "deep": [
            "torch>=1.7.0",
            "tensorflow>=2.4.0",
        ],
        "distributed": [
            "dask>=2.30.0",
            "distributed>=2.30.0",
        ],
    },
)