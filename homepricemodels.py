import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats


# import redfin


def load_all_home_stats(pickle_file='pickles/home_stats_all.pkl'):
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
    # Drop a few columns right away because they're too sparse (too many NaNs) or because they're not interesting.
    home_stats_df.drop(drop_list, axis=1, inplace=True)

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

    # Remove +4 zip code extension and eliminate errant zip codes
    home_stats_df['Zip Code'] = home_stats_df['Zip Code'].map(lambda string: string.split('-')[0])
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


def get_engineered_features(home_stats_df):
    """

    :param home_stats_df:
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

    home_stats_df['Log Sales Price'] = np.log10(home_stats_df['Sales Price'])
    home_stats_df['Sqrt Sales Price'] = np.sqrt(home_stats_df['Sales Price'])


    for key in ['Style', 'City', 'Zip Code']:
        home_stats_df = pd.concat((home_stats_df, pd.get_dummies(home_stats_df[key])), axis=1)

    return home_stats_df


def diagnostic_plot(x, y):
    plt.figure(figsize=(20, 5))

    rgr = LinearRegression()
    rgr.fit(x.reshape(s, 1), y)
    pred = rgr.predict(x.reshape(s, 1))

    plt.subplot(1, 3, 1)
    plt.scatter(x, y)
    plt.plot(x, pred, color='blue', linewidth=1)
    plt.title("Regression fit")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 3, 2)
    res = y - pred
    plt.scatter(pred, res)
    plt.title("Residual plot")
    plt.xlabel("prediction")
    plt.ylabel("residuals")

    plt.subplot(1, 3, 3)
    # Generates a probability plot of sample data against the quantiles of a
    # specified theoretical distribution
    stats.probplot(res, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")

    def main():
        pass