from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="comprehensive-photo-analyzer",
    version="0.1.0",
    author="Photo Analysis Team",
    description="Comprehensive photo and image analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0",
        "scikit-image>=0.20.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "advanced": [
            "colour-science>=0.4.0",
            "colorspacious>=1.1.0",
            "pywavelets>=1.4.0",
            "mahotas>=1.4.0",
            "scikit-learn>=1.3.0",
            "plotly>=5.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "photo-analyzer=main:main",
        ],
    },
)