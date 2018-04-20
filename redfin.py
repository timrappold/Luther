import requests
from bs4 import BeautifulSoup
import pandas as pd

pd.set_option('display.max_columns', None)


def build_landing_url(zipcode, filter_):
    """
    Generate the url for the landing page.
    :param zipcode: type str or int.
    :param filter: type str.
    :return: type str.

    zipcode is passed as '94605'
    filter: 'sold-3yr'
            'sold-all'
    """

    url_filter = {'sold-3yr': '/filter/include=sold-3yr',
                  'sold-all': '/filter/include=sold-all'}


    try:
        filter_piece =  url[filter_]
    except:
        raise KeyError("The filter {} isn't defined in url_filter dictionary".format(filter_))

    if
    zipcode_piece = 'https://www.redfin.com/zipcode/94605/filter/include=sold-3yr'


def main():
    """
    Runs
    :return: None
    """
    zipcode = '94605'
    pass

if __name__ == '__main__':
    main()