import os
import sys
import pickle

import pandas as pd

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

pd.set_option('display.max_columns', None)


def construct_filter_url(zipcode, crawler=1):
    """
    Return a landing url that filters by zipcode. Crawler is used to determine the page of the search
    results. Redfin search results are up to 18 pages long.

    In the current version, the filter is set to 'sold-3yr' and
    could be expanded to include 'sold-all'.

    :param zipcode: str or int.
    :param crawler:
    :return: url, type string.
    """

    assert 1 <= crawler <= 18, 'URL page crawler is outside range [1,18].'
    if crawler == 1:
        crawler_string = ''
    else:
        crawler_string = '/page-' + str(crawler)

    zipcode = str(zipcode)

    filter_ = 'sold-3yr'

    url = ('https://www.redfin.com/zipcode/'
           + zipcode
           + '/filter/include='
           + filter_
           + crawler_string)

    return url


def make_soup_via_requests(url):
    """
    Return a Soup object from a url using requests. Use for home_listing urls. Doesn't
    successfully apply filters in landing url.

    :param url: str
    :return: BeautifulSoup object
    """

    hdr = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(url, headers=hdr)

    assert response.status_code == 200, "HTML status code isn't 200 on page {}.".format(url)
    page = response.text
    #print('Page is {} long.'.format(len(page)))

    return BeautifulSoup(page, "lxml")


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
    """
    listing_url_list = []
    for url in soup.find_all("a", class_="cover-all"):
        listing_url_list.append(url['href'])

    return listing_url_list


def parse_home_listing(listing_url):
    """
    Finds all the relative individual house data links on a landing page.
    """
    url = 'https://www.redfin.com/' + listing_url

    home_soup = make_soup_via_requests(url)
    return home_soup



def main():
    """
    Runs
    :return: None
    """
    #zipcode = '94605'  # listing links are pickled!
    #zipcode = '94610'
    #zipcode = '94110'
    #zipcode = '95476'
    #zipcode = '94611'

    list_of_zipcodes = ['94549',
                        '90403',
                        '90049',
                        '90292',
                        '90301',
                        '11211',
                        '10024',
                        '48503',
                        '77373',
                        ]

    for zipcode in list_of_zipcodes:

        listings = []

        for c in range(18):
            url = construct_filter_url(zipcode, crawler=c+1)
            landing_soup = make_soup_via_selenium(url)
            listings = listings + find_home_listing_urls(landing_soup)

            # the following saves a pickle with every iteration
            with open('pickles/listing_urls_' + zipcode + 'page_' + str(c+1) + '.pkl', 'wb') as picklefile:
                pickle.dump(listings, picklefile)

        print(listings)


    return listings


if __name__ == '__main__':
    listings = main()