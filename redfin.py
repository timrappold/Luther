import os
import sys
import pickle
import time

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 3)


### GET LISTING URLS ######################################################################


def construct_filter_url(zipcode, page_num=1):
    """
    Return a landing url that filters by zipcode. page_num is used to determine the page of the search
    results. Redfin search results are up to 18 pages long.

    In the current version, the filter is set to 'sold-3yr' and
    could be expanded to include 'sold-all'.

    :param zipcode: str or int.
    :param page_num:
    :return: url, type string.
    """

    assert 1 <= page_num <= 18, 'URL page_num is outside range [1,18].'
    if page_num == 1:
        page_num_string = ''
    else:
        page_num_string = '/page-' + str(page_num)

    zipcode = str(zipcode)

    filter_ = 'sold-3yr'

    url = ('https://www.redfin.com/zipcode/'
           + zipcode
           + '/filter/include='
           + filter_
           + page_num_string)

    return url


def make_soup_via_selenium(url):
    """
    Return a Soup object from a url using Selenium and Chromedriver. Use for landing page urls to ensure
    that filters are applied.

    :param url: str
    :return: BeautifulSoup object
    """

    chromedriver = "~/Downloads/chromedriver"  # path to the chromedriver executable
    chromedriver = os.path.expanduser(chromedriver)
    print('chromedriver path: {}'.format(chromedriver))
    sys.path.append(chromedriver)

    driver = webdriver.Chrome(chromedriver)

    driver.get(url)
    html = driver.page_source

    return BeautifulSoup(html, "lxml")


def find_home_listing_urls(soup):
    """
    Finds all the relative individual house data links on a landing page.

    :param soup: a bs4 Soup object
    :return:  list of strings. Each string is a relative URL representing one house listing.
    """
    listing_url_list = []
    for url in soup.find_all("a", class_="cover-all"):
        listing_url_list.append(url['href'])

    return listing_url_list


def save_links_for_every_zipcode(zipcode_dict=None, page_range=18):
    """
    Saves the landing pages for every zipcode in zipcode_dict in a pickle file in ./pickles/.
    The data structure for the landing page urls is a list of strings.

    :param zipcode_dict: dict with key = 'zipcode', value = ('City','State'). E.g.: {'94605': ('Oakland', 'CA'),...}.
                        Defaults to zipcode_dict defined in function body.
    :param page_range: int. Works with an integer in the range [1,18]. Redfin search results contain 18 pages or fewer.
                        Defaults to page_range = 18.
    :return: None

    This function depends on the following functions:

        - construct_filter_url(zipcode, page_num=c+1)
        - make_soup_via_selenium(url)
        - find_home_listing_urls(landing_soup)

    """

    assert 1 <= page_range <= 18, 'page_range is outside [1,18].'

    if zipcode_dict is None:
        print('Using DEFAULT zipcode_dict...')

        zipcode_dict = {'94605': ('Oakland', 'CA'),
                        '94610': ('Oakland', 'CA'),
                        }

    for zipcode, city in zipcode_dict.items():
        print('Getting landing page for {} in {}'.format(zipcode, city))

        listings = []

        for c in range(page_range):
            url = construct_filter_url(zipcode, page_num=c+1)
            landing_soup = make_soup_via_selenium(url)
            listings = listings + find_home_listing_urls(landing_soup)

        # the following saves a pickle with every zipcode
        with open('pickles/listing_urls_' + zipcode + '_all' + '.pkl', 'wb') as picklefile:
            pickle.dump(listings, picklefile)

        print(listings)

    return None


def combine_zipcode_listings_pickles_into_one(pickle_directory='pickles/'):
    """
    Combines all pickles ending in ...18.pkl into one pickle named 'listing_urls_all.pkl'. The list of strings are
    combined into one list of strings; only unique URLs are retained.
    :param pickle_directory: str.
    :return: None
    """
    listing_urls = []

    for file in os.listdir(pickle_directory):
        with open(pickle_directory + file, 'rb') as picklefile:
            if file.endswith("_all.pkl"):
                listing_urls += pickle.load(picklefile)

    listing_urls_unique = list(set(listing_urls))

    with open(pickle_directory + 'listing_urls_all.pkl', 'wb') as picklefile:
        pickle.dump(listing_urls_unique, picklefile)

    return None

### SCRAPE DATA AND SAVE: ###################################################


def load_all_urls(pickle_file='pickles/listing_urls_all.pkl'):
    """
    Loads pickle file that contains the list of strings of relative URLs of individual home listings on Redfin.

    :param pickle_file: type(str)
    :return: list_of_relative_urls, type(list)
    """
    with open(pickle_file, 'rb') as picklefile:
        list_of_relative_urls = pickle.load(picklefile)

    return list_of_relative_urls


def get_home_soup(home_rel_url):
    """
    Accepts the relative URL for an individual home listing and returns a BeautifulSoup object.

    :param home_rel_url: str. Relative URL string that can be appended to the root 'https://www.redfin.com'.
                             E.g. '/CA/Oakland/9863-Lawlor-St-94605/home/1497266'
    :return: BeautifulSoup object for an individual listing's website.
    """

    url = 'https://www.redfin.com' + home_rel_url
    hdr = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=hdr)
    assert response.status_code == 200, "HTML status code isn't 200 for {}.".format(home_rel_url)

    return BeautifulSoup(response.text, "lxml")


def get_property_history(home_soup):
    """
    Returns a dict of the uppermost "sold event" row of the Property History Table on a Redfin listing.
    :param home_soup: BeautifulSoup object made by get_home_soup(home_rel_url).
    :return: dict. Keys are 'Last Sold' and 'Sales Price'. Values are strings.
    """
    sold_row_soup = home_soup.find("tr", class_="sold-row PropertyHistoryEventRow")

    if sold_row_soup is not None:
        date = sold_row_soup.find('td', class_='date-col nowrap').get_text()
        price = sold_row_soup.find('td', class_='price-col number').get_text()

        property_history = {'Last Sold': date,
                            'Sales Price': price,
                            }

    else:
        property_history = {'Last Sold': None,
                            'Sales Price': None,
                            }

    return property_history


def get_home_facts(home_soup):
    """
    Returns a dict of all rows of the Home Facts Table on a Redfin listing.
    :param home_soup: BeautifulSoup object made by get_home_soup(home_rel_url).
    :return: dict of (key, value) pairs from the Home Facts table.
    """
    facts_table = home_soup.find("div", class_="facts-table")
    table_row = facts_table.find_all(class_="table-row")

    home_facts = {}
    for row in table_row:
        label = row.find(class_='table-label').get_text()
        value = row.find(class_='table-value').get_text()
        home_facts[label] = value

    return home_facts


def get_zipcode(home_soup):
    """
    Returns a dictionary with a listing's zip code.
    :param home_soup: BeautifulSoup object made by get_home_soup(home_rel_url).
    :return: dict. E.g. {'Zip Code': '94605'}
    """
    citystatezip_soup = home_soup.find('span', class_='citystatezip')
    zipcode = citystatezip_soup.find('span', class_='postal-code').get_text()

    return {'Zip Code': zipcode}


def get_home_stats(home_rel_url):
    """
    Aggregates the functions get_zipcode, get_home_facts, and get_property_history and returns
    the single-source-of-truth dictionary of a home listing's statistics.
    :param home_rel_url: relative URL of home listing. E.g.: '/CA/Oakland/888-Warfield-Ave-94610/home/1881044'
    :return: dict. Contains stats for one home. Ready to be aggregated in a list and then converted to pd.DataFrame.
    """
    home_soup = get_home_soup(home_rel_url)

    zipcode = get_zipcode(home_soup)
    property_history = get_property_history(home_soup)
    home_facts = get_home_facts(home_soup)

    home_rel_url_dict = {'rURL': home_rel_url}

    return {**home_rel_url_dict, **zipcode, **property_history, **home_facts}


def scrape_home_stats(list_of_relative_urls=None, pickle_directory='pickles/', save_every=250):
    """
    Loops through list_of_relative_urls, scrapes home listing data, and stores it all in a pickle. As as
    safeguard against crashes or website defenses, this function uses a sleep timer which delays each iteration
    by half a second +/- some random noise. The function also saves intermediate results before loop is complete
    using save_every.
    :param list_of_relative_urls: list of strings. Relative URLs for indivi. home listings; append to https://www.redfin.com
    :param pickle_directory: str. Path where pkl files are saved to.
    :param save_every: int. Saves every

    :return: list of dicts. Contains all scraped home listing data. Convert to DF via pd.DataFrame(all_home_stats).
    """
    if list_of_relative_urls is None:
        print("List of relative URLS is NONE, loading default data set.")
        list_of_relative_urls = load_all_urls()

    all_home_stats = []

    for i, home_rel_url in enumerate(list_of_relative_urls):

        print('Processing link #{}: {}'.format(i, home_rel_url))
        all_home_stats.append(get_home_stats(home_rel_url))

        if i in list(range(0, len(list_of_relative_urls), save_every)):
            with open(pickle_directory + 'home_stats_all_{}.pkl'.format(i), 'wb') as picklefile:
                pickle.dump(all_home_stats, picklefile)

        r = 0.2 * np.random.randn(1) + .5
        time.sleep(r)

    with open(pickle_directory + 'home_stats_all.pkl', 'wb') as picklefile:
        pickle.dump(all_home_stats, picklefile)

    return all_home_stats


def main():
    pass


if __name__ == '__main__':

    zipcode_dict2 = {'94605': ('Oakland', 'CA'),
                     '94610': ('Oakland', 'CA'),
                     '94611': ('Oakland', 'CA'),
                     '94110': ('San Francisco', 'CA'),
                     '95476': ('Sonoma', 'CA'),
                     '94549': ('Lafayette', 'CA'),
                     '90403': ('Santa Monica', 'CA'),
                     '90049': ('Los Angeles', 'CA'),
                     '90292': ('Los Angeles', 'CA'),
                     '90301': ('Los Angeles', 'CA'),
                     '11211': ('Brooklyn', 'NY'),
                     '10024': ('New York', 'NY'),
                     '48503': ('Flint', 'MI'),
                     '77373': ('Houston', 'TX'),
                     '94601': ('Fruitvale', 'Oakland', 'CA'),
                     '94603': ('Elmhurst', 'Oakland', 'CA'),
                     '94607': ('West Oakland', 'Oakland', 'CA'),
                     '94618': ('Rockridge', 'Oakland', 'CA'),
                     '94609': ('Temescal', 'Oakland', 'CA'),
                     '94602': ('Lincoln Highlands', 'Oakland', 'CA'),
                     '94619': ('Redwood Heights', 'Oakland', 'CA'),
                     '94606': ('Highland Terrace', 'Oakland', 'CA'),
                     '90221': ('East Compton', 'Los Angeles', 'CA'),
                     '90056': ('Ladeira Heights', 'Los Angeles', 'CA'),
                     '90232': ('Culver City', 'Los Angeles', 'CA'),
                     '90039': ('Atwater Village', 'Los Angeles', 'CA'),
                     '90026': ('Silver Lake', 'Los Angeles', 'CA'),
                     '90041': ('Eagle Rock', 'Los Angeles', 'CA'),
                     '90024': ('Westwood', 'Los Angeles', 'CA'),
                     '90069': ('Sunset Strip', 'Los Angeles', 'CA'),
                     '90027': ('Los Feliz', 'Los Angeles', 'CA'),
                     }

    save_links_for_every_zipcode(zipcode_dict=zipcode_dict2)
