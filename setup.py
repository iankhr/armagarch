# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:00:13 2020

@author: Ian Khrashchevskyi
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="armagarch",
    version="1.0.2",
    author="Ian Khrashchevskyi",
    author_email="iankhr@yahoo.com",
    description="Library for flexible mean and volatility modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iankhr/armagarch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)