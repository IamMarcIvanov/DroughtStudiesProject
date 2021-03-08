# DroughtStudiesProject

**File**: linear_regression2.py<br/>
**What it does**: <br/>
&nbsp;&nbsp;&nbsp;&nbsp;does linear regression on monthly data for hyderabad<br/>
**Output**: finds MSE and R-squared loss<br/>
**Conclusion**:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;As can be seen the testing and training losses fluctuate a lot over iterations and relative to each other.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;Thus the monthly data is insufficient.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;Additionally, linear regression is unable to capture the subtleties in the data.<br/>
**File**: 8Mar2021PCA1.py<br/>
**What it does**: <br/>
&nbsp;&nbsp;&nbsp;&nbsp;does PCA on the following variables: HURS HUSS PSL TAS CLT ZG500 VAS UAS<br/>
&nbsp;&nbsp;&nbsp;&nbsp;prints the explained variance as a percentage<br/>
&nbsp;&nbsp;&nbsp;&nbsp;it does pca for 10 different random orders of variables<br/>
&nbsp;&nbsp;&nbsp;&nbsp;data is daily data.
**Output**: 8Mar2021PCA1_output.txt<br/>
**Conclusion**: PCA is giving wildly different results solely based on the order, thus it cannot be relied on to give correct results.<br/>
