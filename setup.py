from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="orbital-collision-predictor",
    version="1.0.0",
    author="Vinay Raja",
    author_email="vinayraja005@gmail.com",
    description="ML-powered orbital debris tracking and collision prediction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vraja05/orbital-collision-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "sgp4>=2.22",
        "tensorflow>=2.13.0",
        "plotly>=5.15.0",
        "streamlit>=1.25.0",
    ],
    entry_points={
        "console_scripts": [
            "orbital-tracker=src.examples.live_tracking_demo:main",
        ],
    },
)