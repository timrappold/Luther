# Project Luther:  Seasonal Variability in the Housing Market

Author: Tim Rappold

GitHub: timrappold

Email: tim.rappold@gmail.com



## Project Summary

This is a webscraping and linear regression project. The goal is to describe and quantify seasonal swings in the housing market, be it seasonal turnover or seasonal fluctuations in housing prices that can be extracted from longer-term housing trends. The goal is to identify the best month to buy or sell, if there is such thing, or else assert that seasonal patterns don't matter.

## Scope:

- **Spatial:** Study housing data from four ZIP codes (potentially will reduce the scope or the binning of the units to individual neighborhoods, which typically are smaller than zip codes and could act as more cohesive units within a market):
  - 94605 (Oakland)
  - 94110 (San Francisco)
  - Some neighborhood in Houston
  - Some neighborhood in Tarrytown, NY
- **Temporal:** Choose data from at least three years. The multiyear trend will be necessary to attempt to separate seasonal price and supply variations from an overal trend.
- **Regression Model**: The regression will take into account a number of potential features: 
  - A regression for long-term (non-seasonal) trends
  - A regression for seasonal trends (i.e.: data - longterm trends)
  - After observing the patterns, identify clues and angles to explain or correlate patterns



## Approach:

#### Web Scraping:

The goal of webscraping is to get housing _sales_ data from the previous three years for the zipcodes in question. `requests` and `BeautifulSoup` will tbe the go-to libraries to grab the following information:

* Sale price

* Sale date

* Address

* Neighborhood

* Date of first offering (if available)

* House features (contained in box/table called "Home Facts":

  `home_soup.find("div", class_="facts-table")`:

  - [x] Num Bedrooms `Beds`

  - [x] Num Bath `Baths`

  - [x] Finished square footage `Finished Sq. Ft.`

  - [x] Unfinished square footage `Unfinished Sq. Ft.`

  - [x] Total square feet `Total Sq. Ft.`

  - [x] `Stories`

  - [x] Lot size `Lot Size`

  - [x] Single family home/condo/etc `Style` -> Single Family Residential

  - [x] `Year Built`

  - [x] `Year Renovated`

  - [x] `County`

    ​

TODO: 

- [ ] generate landing site by search term (i.e. ZIP code)
- [ ] Interact with Filter settings to alter landing site to include 3-years of recently sold homes, and `show all`: https://www.redfin.com/zipcode/94605/filter/include=sold-3yr The goal is to get all the search results on one page. Otherwise, figure out how the next page is generated.
- [ ] Build a prototype to cull information from the first house of a page.
- [ ] Build a Series object that holds the information for one house
- [ ] figure out whether to keep series in a dict or a list?
- [ ] Build a sleep timer?
- [ ] Scrape & Pickle



#### Regression:



