# ARMA-GARCH module
The code performs joint estimation of ARMA(P,Q)-GJR-GARCH(p,o,q) model. The code creates garch-object:

INPUT:
-----
*  data - pandas DataFrame or numpy array. (If your data are stock returns, it is advised to multiply them by 100 for stability reasons)
*  PQ - tuple with AR and MA lags in ARMA model (default: (0,0))
*  poq - tuple which specifies the amount of lags in GJR-GARCH model, if o is set to 0 then GARCH model is estimated instead. (default: (1,0,1))
*  startingVals - 2+P+Q+p+o+q array with starting vals in ARMA-GJR-GARCH model, if not sepcified the grid search is performed to find starting values. (default: None)
*  constant - if False, forces to estimate GARCH model and assume the data is demeaned (default: True)
*  debug - if True, shows additional output necessary for debuging (default: False)
*  printRes - if False, does not shows the resulted estimations (default:True)
*  fast - if True, skips estimation of standard errors and only estimates the coefficients (default: False)

FUNCTIONS:
-----------
*  fit() - estimate the ARMA-GJR-GARCH model
*  summary() - print the table with estimated parameters and statistics
*  applyModel(newData, reconstruct = False, y0=0, h0=1) - if reconstruct is set to False, applies the estimated model on newData and returns list of innovations and consitional variances. If reconstruct = True, then treats newData as simulated standardized innovations and returns list with returns and conditional variances
*  predict(step=1) - makes n-step forecast. (EXPERIMENTAL: works stable only for 1 step forecast for now)
  
PROPERTIES:
-----------
*  ht - array of conditional variances
*  params - estimates of the model
*  vcv - variance-covariance matrix (robustly estimated)
*  data - raw data
*  AIC - Aikake Information Criteria
*  BIC - Bayesian Information Criteria
*  HQIC - Harley-Quinn Information Criteria
*  ll - log-likelihood
*  et - innovations
*  stres - standardized residuals
*  uncvar - unconditional variance of the data  
  

## Getting Started

The code requires: NumPy, Pandas, SciPy, Shutil and Statsmodels
Copy garch.py and basicFun.py in a directory of your project. Then basic usage is as follows

```
import pandas as pd
from garch import garch

# load your data here
data = pd.read_csv('PATH_TO_MY_DATA.csv')

# create garch object
garchModel = garch(data, PQ = (1,1), poq = (1,0,1))
# fit model in the data
garchModel.fit()

````


## Authors

* **Ian Khrashchevskyi** - [iankhr](https://github.com/iankhr)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Special thanks to Kevin Sheppard for his [Python for Econometrics](https://www.kevinsheppard.com/Python_for_Econometrics), which was an inspiration to write current code

