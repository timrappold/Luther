# Project Luther:  How do Home Prices Vary with Location and Features?

Author: Tim Rappold

GitHub: timrappold

Email: tim.rappold@gmail.com



## Project Link:

https://github.com/timrappold/Luther



## Project Summary

This is a webscraping and linear regression project. The goal is to describe and quantify how home prices change based on zip code and features. What's the difference between a 2-bed, 1-bath and a 3-bed, 2-bath home in 94605? What about in 90056? 

This project develops the code  to scrape home sales data from Redfin.com, focusing on sales within the last three years in Oakland and Los Angeles. The Python module `redfin.py` contains all code related to scraping, using both the `bs4` and `selenium` webscraping libraries.

In the module `homepricemodels`, the project develops two linear regression models using `statsmodels.OLS` and `scikit-learn.Lasso` to take two quite different approaches to multivariate regression and model selection.

The main analysis is located in `homepricemodels_client.ipynb. `There, the project concludes with an analysis of the above scenario (home prices in Eastmont Hills, zip code 94605 and in Ladeira Hts, zip code 90056) using the model developed with `Lasso` regression and a home-brewed Backward Stepwise model selection technique.

