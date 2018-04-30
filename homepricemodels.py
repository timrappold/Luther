import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import matplotlib.pyplot as plt

import scipy.stats as stats
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


def load_all_home_stats(pickle_file='pickles/combined_home_stats.pkl'):
    """
    Loads pickle file that contains the stats of all individual home listings on scraped from Redfin.

    :param pickle_file: type(str)
    :return: list_of_relative_urls, type(list)
    """
    with open(pickle_file, 'rb') as picklefile:
        all_home_stats = pickle.load(picklefile)

    return all_home_stats


def clean_lot_size(string):
    """
    Converts the string in column Lot Size to a float in units sq. ft. If original string references units 'Acres', the
    value is converted to square feet.
    :param string: str.
    :return: float.
    """
    string = string.replace(',', '')

    if string.endswith('Acres'):
        string = string.strip('Acres')
        mult = 43560.
        return float(string) * mult

    elif string.endswith('Sq. Ft.'):
        string = string.strip('Sq. Ft.')
        mult = 1.
        return float(string) * mult

    else:
        return np.nan


def clean_home_stats_df(all_home_stats):
    """
    Accepts the list of dictionaries loaded via load_all_home_stats and returns a cleaned DataFrame.

    :param all_home_stats: accepts a raw home_stats_df straight from the pickle via load_all_home_stats
    :return: pd.DataFrame home_stats_df
    """
    """ Removes columns with spotty data. 

    ['Sales Price']: Removes rows without a Sales Price, removes all non-numeric characters from 'Sales Price'
    """

    home_stats_df = pd.DataFrame(all_home_stats)

    drop_list = ['Accessible',
                 'APN',
                 'Basement',
                 'Features',
                 'Finished Sq. Ft.',
                 'Garage',
                 'HOA Dues',
                 'Parking Spaces',
                 'Stories',
                 'Unfinished Sq. Ft.',
                 ]

    for col in home_stats_df.columns:
        if col in drop_list:
            home_stats_df.drop(col, axis=1, inplace=True)

    home_stats_df.fillna(value='-', inplace=True)  # Do this so string methods can be universally applied below.

    to_numeric_list = ['Baths',
                       'Beds',
                       'Sales Price',
                       'Total Sq. Ft.',
                       'Year Built',
                       'Year Renovated',
                       ]

    for key in to_numeric_list:
        home_stats_df[key] = home_stats_df[key].map(lambda string: string.replace('$', ''))
        home_stats_df[key] = home_stats_df[key].map(lambda string: string.replace(',', ''))

        home_stats_df[key] = pd.to_numeric(home_stats_df[key], errors='coerce')

    home_stats_df = home_stats_df[pd.notnull(home_stats_df['Sales Price'])]

    # Convert "Last Sold" date to datetime and remove errant rows that reference sale dates prior to filter window.
    home_stats_df = home_stats_df[pd.notnull(home_stats_df['Last Sold'])]
    home_stats_df['Last Sold'] = pd.to_datetime(home_stats_df['Last Sold'], format='%b %d, %Y')
    home_stats_df = home_stats_df[home_stats_df['Last Sold'] > datetime.strptime('2015-05-01', '%Y-%m-%d')]

    # Creates a new Lot Size column in units sq. ft. and drop the old column
    home_stats_df['Lot Size Sq. Ft.'] = home_stats_df['Lot Size'].map(clean_lot_size)
    home_stats_df.drop('Lot Size', axis=1, inplace=True)

    # print('Length of Home_stats_DF after Lot Size: ', len(home_stats_df))

    # Remove +4 zip code extension and eliminate errant zip codes
    home_stats_df['Zip Code'] = home_stats_df['Zip Code'].map(lambda string: string.split('-')[0])

    if len(home_stats_df) > 1000:
        zip_group = home_stats_df.groupby('Zip Code')
        home_stats_df = zip_group.filter(lambda x: len(x) > 100)

    # Down-select Zip Codes to Oakland and Los Angeles only:
    home_stats_df = (home_stats_df[home_stats_df['Zip Code'].str.startswith('946') | home_stats_df['Zip Code']
                     .str.startswith('90')])

    # Fill NaNs in Year Renovated with values from Year Built. I.e. the home was last "new" when it was built.
    home_stats_df['Year Renovated'].fillna(home_stats_df['Year Built'], inplace=True)

    # Remove Styles that don't fit squarely into the single family home category:
    drop_styles_list = ['Vacant Land', 'Other', 'Unknown', 'Mobile/Manufactured Home',
                        'Multi-Family (2-4 Unit)', 'Multi-Family (5+ Unit)']
    home_stats_df = home_stats_df[~home_stats_df['Style'].isin(drop_styles_list)]

    # Drop all remaining rows that have ANY NaNs:
    home_stats_df.dropna(axis=0, how='any', inplace=True)

    return home_stats_df


def get_engineered_features(home_stats_df, cross_terms=False):
    """

    :param home_stats_df: clean DataFrame from clean_home_stats_df
    :param cross_terms: bool. cross_terms=True will add cross-term columns to dataframe.
    :return:
    """

    home_stats_df['Weeks ago'] = ((home_stats_df['Last Sold'])
                                  .map(lambda td: (td - home_stats_df['Last Sold'].max()).days // 7)
                                  )

    def map_city(row):
        if row['Zip Code'].startswith('946'):
            city = 'Oakland'
        elif row['Zip Code'].startswith('90'):
            city = 'Los Angeles'
        else:
            city = 'unknown'
        return city

    home_stats_df['City'] = home_stats_df.apply(map_city, axis=1)

    home_stats_df['Month Sold'] = home_stats_df['Last Sold'].map(lambda s: s.month)
    home_stats_df['Year Sold'] = home_stats_df['Last Sold'].map(lambda s: s.year)

    home_stats_df['Sqrt Sales Price'] = np.sqrt(home_stats_df['Sales Price'])

    y_list = ['Sales Price', 'Sqrt Sales Price']

    for key in ['Style', 'City', 'Zip Code']:
        home_stats_df = pd.concat((home_stats_df, pd.get_dummies(home_stats_df[key])), axis=1)

    y = home_stats_df[y_list]
    X = home_stats_df.drop(y_list, axis=1)

    if cross_terms is True:
        X = get_cross_terms(X, incl_dtypes=['float64', 'uint8'])

    return X, y


def get_cross_terms(df, incl_dtypes='float64'):
    """

    :param df:
    :param incl_dtypes: a string or a list of strings. E.g. 'float64', 'int64'
    :return:
    """
    X = df.select_dtypes(include=incl_dtypes)
    column_list = list(X.columns)  # make copy to prevent list mutation
    cross_terms = []
    while len(column_list) > 0:
        popped = column_list.pop()
        # print popped
        for col in column_list:
            cross_term = popped + '*' + col
            X[cross_term] = X[popped] * X[col]
            cross_terms.append(cross_term)
    return pd.concat([df, X[cross_terms]], axis=1)


def sm_ols_wrapper(y, X=None):
    """
    Accepts target vector y and design matrix X and returns a
    statsmodels OLS.results object. This object can be used like so:
        results.predict
    :param y:
    :param X:
    :return:
    """
    if X is None:
        X = pd.Series(1, index=y.index)
    else:
        X = sm.add_constant(X, prepend=True)

    # print('X = ', X)
    model = sm.OLS(y.astype(float), X.astype(float))
    results = model.fit()
    y_pred = results.predict(X)
    print(results.summary())
    diagnostic_plot(y_pred, y)
    return results


def get_gridsearch_lasso(X_train, y_train, score='r2', kfold=5):
    """
    Wrapper for gridsearchCv. Returns optimized hyper-parameters and provides a number of easy-to-follow print-outs.
    :param X_train: design matrix training set
    :param y_train: target training set.
    :param score: scoring method, defaults to R^2.
    :param kfold: Number of folds in the cross-validation.
    :return: dict of best_params_.
    """
    # Set the parameters by cross-validation
    tuned_parameters = [{'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                         'normalize': [True, False]}]

    print("# Tuning hyper-parameters for %s" % 'r2')
    print()

    reg = GridSearchCV(Lasso(), tuned_parameters, cv=kfold,
                       scoring=score)
    reg.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(reg.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = reg.cv_results_['mean_test_score']
    stds = reg.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, reg.cv_results_['params']):
        print("%0.4f (+/-%0.04f) for %r"
              % (mean, std * 2, params))
    print()
    return reg.best_params_


def get_lasso(X_train, y_train, alpha, normalize):
    """
    Wrapper for Lasso. Returns the sklearn regression object as well as a list of features whose coefficients aren't zero.
    :param X_train:
    :param y_train:
    :param alpha:
    :param normalize:
    :return: tuple of regression object and "not_null" list of features: (reg, not_null_list)
    """
    reg = Lasso(alpha=alpha, normalize=normalize)
    reg.fit(X_train, y_train)

    not_null_list = []

    for tup in list(zip(X_train.columns, reg.coef_)):
        print(tup)
        if abs(tup[1]) > 0.001:
            not_null_list.append(tup)
    return reg, not_null_list


def get_lasso_with_gridsearch(X_train, y_train):
    """
    Combines get_gridsearch_lasso and get_lasso into one pipeline.
    :param X_train:
    :param y_train:
    :return:
    """

    best_params_ = get_gridsearch_lasso(X_train, y_train, score='r2', kfold=5)
    print(best_params_)
    return get_lasso(X_train, y_train, best_params_['alpha'], best_params_['normalize'])


def lasso_loop(X_train, y_train):
    """
    Step-backward Model selection algorithm using Lasso and Gridsearch. Returns sklearn Estimator `reg`, not_null_list (which
    is a list of tuples of the form [(feature, values), ...] and the list `features` of the form ['feature1, 'feature2', ...]

    :param X_train: design matrix. Pandas DataFrame.
    :param y_train: target vector. Pandas Series.
    :return: (reg, not_null_list, features)
    """
    diff_not_null_list = 1

    while diff_not_null_list > 0:

        print('The design matrix has {} features.'.format(len(X_train.columns)))

        reg, not_null_list = get_lasso_with_gridsearch(X_train, y_train)

        features = []
        for feature in not_null_list:
            features.append(feature[0])

        diff_not_null_list = len(X_train.columns) - len(features)

        X_train = X_train[features]

    return reg, not_null_list, features


def get_vif(X):
    """
    Get a DataFrame of Variance Inflation Factors.
    :param X: pandas DataFrame. Design matrix/feature matrix.
    :return: A DataFrame with a VIF score for each feature.
    """

    X = X.select_dtypes(include=['float64', 'uint8'])

    vif = pd.DataFrame()

    for i in X.columns:
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['Features'] = X.columns

    return vif


def diagnostic_plot(y_pred, y):
    """
    Makes three diagnostic plots:
        1) Predicted vs Actual
        2) Residual
        3) Q-Q plot to inspect Normality

    :param y_pred: the target predicted by the model
    :param y: the actual/measured target
    :return: None
    """

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, y, alpha=0.2)
    plt.plot(y_pred, y_pred, color='red', linewidth=1)
    plt.title("Predicted vs Actual")
    plt.xlabel("Y Predicted")
    plt.ylabel("Y Actual")
    # plt.ylim([0, 1.5e7])  # remove in the future
    plt.ylim([0, 1.05*y_pred.max()])
    plt.xlim([0, 1.05*y_pred.max()])

    plt.subplot(1, 3, 2)
    res = y - y_pred
    plt.scatter(y_pred, res, alpha=0.2)
    plt.title("Residual plot")
    plt.xlabel("prediction")
    plt.ylabel("residuals")
    #plt.ylim([-1.5e7, 1.5e7])  # remove in the future

    ax = plt.subplot(1, 3, 3)
    # Generates a probability plot of sample data against the quantiles of a
    # specified theoretical distribution
    stats.probplot(res, dist="norm", plot=plt)
    ax.get_lines()[0].set_alpha(0.2)
    # ax.get_lines()[0].set_marker('p')
    # ax.get_lines()[0].set_markerfacecolor('r')
    plt.title("Normal Q-Q plot")
    #plt.ylim([-0.2e7, 0.2e7])  # remove in the future
    #plt.xlim([-2, 2])          # remove in the future


def main():
    pass


if __name__ == '__main__':
    pass
