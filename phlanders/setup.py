#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


setup(
    name="phlanders",
    version="0.1.0a",
    license="NIST",
    description="Phenotype landscape deep regression",
    author="Dr. Peter Tonner",
    author_email="peter.tonner@nist.gov",
    url="https://www.github.com/ptonner/laci-landscape-dnn",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={},
    keywords=["computational biology", "machine learning"],
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "torch",
        "scipy",
        "attrs",
        "cattrs",
        "tqdm",
        "sklearn",
        "Biopython",
        "logomaker",
        "networkx",
    ],
    tests_requires=["pytest", "pytest-cov"],
    extras_require={},
    entry_points={"console_scripts": ["phlanders=phlanders.cli:main"]},
)
