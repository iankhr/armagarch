# armagarch package
The package provides a flexible framework for modelling time-series data. The main focus of the package is implementation of the ARMA-GARCH type models.

**Full documentation is coming soon.**

## Installation

The latest stable version can be installed by using pip
```
pip install armagarch
```

The master branch can be installed with

```
git clone https://github.com/iankhr/armagarch
cd armagarch
python setup.py install
```
  

## Example: Modelling conditional volatility of the US excess market returns

The code requires: NumPy, Pandas, SciPy, Shutil, Matplotlib, Pandas_datareader and Statsmodels


```
import armagarch as ag
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np

# load data from KennethFrench library
ff = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench')
ff = ff[0]

# define mean, vol and distribution
meanMdl = ag.ARMA(order = {'AR':1,'MA':0})
volMdl = ag.garch(order = {'p':1,'q':1})
distMdl = ag.normalDist()

# create a model
model = ag.empModel(ff['Mkt-RF'].to_frame(), meanMdl, volMdl, distMdl)
# fit model
model.fit()

# get the conditional mean
Ey = model.Ey

# get conditional variance
ht = model.ht
cvol = np.sqrt(ht)

# get standardized residuals
stres = model.stres

# make a prediction of mean and variance over next 3 days.
pred = model.predict(nsteps = 3)

# pred is a list of two-arrays with first array being prediction of mean
# and second array being prediction of variance

````


## Authors

* **Ian Khrashchevskyi** - [iankhr](https://github.com/iankhr)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Special thanks to Kevin Sheppard for his [Python for Econometrics](https://www.kevinsheppard.com/Python_for_Econometrics), which was an inspiration to write current code

