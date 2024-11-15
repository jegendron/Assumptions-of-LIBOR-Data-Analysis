# Assumptions of LIBOR Data Analysis

This final project from my PhD course work shows the importance of testing the probabilistic assumptions (distribution, dependence, heterogeneity) of your data set before conducting any analysis on said data.

The file below was ran to test if the data was Normal, Independent and Identically distributed:
- <em>0) Prepare Data & Analyze resids of Normal Regression.py</em>

The files below were ran to test if the data was Student's T, Independent, and Identically distributed:
- <em>1) Prepare Data, Analyze resids of Student's T Regression.py</em>
- <em>2) Run Student's T Regression.R</em>
- <em>1) Prepare Data, Analyze resids of Student's T Regression.py</em>

The Student's T Python file is listed twice because after running the first half, then the second half needs the Student T residuals generated from the R file.



## <em>Here is what the first Python file does:</em>
1. Imports the data and prepares it for analysis by lagging it, and also calculates the difference between the lagged data
2. To test for any first-order dependence in the data, lags of the prepared data are calculated (in order to test in the auxiliary regressions)
3. To test for any second-order dependence in the data, squares of the first-order lags are calculated
4. To test for any time-invariance in the data, trend polynomials are calculated
5. In the block comment is code to view t-Plots to see if there are any indicators of violations of probabilistic assumptions in the data
6. A histogram of the residual from the standard regression model is compared to the Normal distribution (to ensure it's truly Normally distributed, or else further model specification is needed)
7. To be precise in testing the probabalistic assumptions now auxiliary regressions are ran for both the first and second moments (like before, if any probabilistic assumptions are violated, then further 
model specification is needed)

## <em>Here is what the first half of the second Python file does:</em>
1. Imports the data and prepares it for analysis by lagging it, and also calculates the difference between the lagged data
2. To test for any first-order dependence in the data, lags of the prepared data are calculated (in order to test in the auxiliary regressions)
3. To test for any second-order dependence in the data, squares of the first-order lags are calculated
4. To test for any time-invariance in the data, trend polynomials are calculated
5. In the block comment is code to view t-Plots to see if there are any indicators of violations of probabilistic assumptions in the data
6. The data and new variables are exported to analyze in the R file

## <em>The R file does the following:</em>
1. Generates a trend $(1,2,...,n)$ to add to the imported data
2. Adds all the data to the StVAR model ('beta' and 'coef' allows us to see the results of the StVAR model)
3. Exports the following to further analyze in the second half of the Python file: residuals, estimated dependent variable $(\hat{y})$ and estimated variance $(\hat{\sigma}^2)$

## <em>Lastly, here is what the second half of the second Python file does:</em>
1. The following dat is imported from the R file: residuals, estimated dependent variable $(\hat{y})$ and estimated variance $(\hat{\sigma}^2)$
2. The program now prompts you to which variable you want to test
3. A histogram of the residual from the StVAR model is compared to the Normal and the Student's t distribution (to ensure it's truly Student's t distributed, or else further model specification is needed)
4. To be precise in testing the probabalistic assumptions now auxiliary regressions are ran for both the first and second moments (like before, if any probabilistic assumptions are violated, then further 
model specification is needed)
