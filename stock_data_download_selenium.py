# -*- coding: utf-8 -*-
"""
Created in 2018

@author: JiaRong
"""

# https://www.investing.com/stock-screener/?sp=country::42|sector::a|industry::a|equityType::a|exchange::62%3Ceq_market_cap;1

import time
from time import strftime
import sys
import numpy as np
import pandas as pd
from selenium.webdriver import Chrome
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup


#---- 1a. Scrape listed stocks from investing.com (Using Selenium) ----        
def get_stocks_info_selenium(country_code_investingcom):
    screener_table = []
    country_referer = np.where(country_code_investingcom==42, 'https://www.investing.com/stock-screener/?sp=country::' + str(country_code_investingcom) + '|sector::a|industry::a|equityType::a|exchange::62%3Ceq_market_cap;',
                               'https://www.investing.com/stock-screener/?sp=country::' + str(country_code_investingcom) +'|sector::a|industry::a|equityType::a%3Ceq_market_cap;')
    
    # find the total pages of listed stocks
    screener_first_page = str(country_referer) + '1'
    driver = Chrome()
    time.sleep(5)
    driver.get(screener_first_page)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, 'lxml') 
    page_text = soup.find_all('a', attrs={'class': 'pagination'})
    last_page = page_text[-1].text
    
    # Scrape stocks information through all the pages using selenium
    range_pages = range(1,(int(last_page) + 1))
    investing_screener = country_referer + pd.Series(list(map(str, range_pages)))
    
    print('Scrapping the stocks information:')
    for i in range_pages:
        
        print('Working on page '+ str(i))
        screener_page = str(investing_screener[i-1])
        driver.get(screener_page)
        time.sleep(5)
        try:
            #WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, "midDiv inlineblock"))) # waits till the element with the specific id appears
            soup = BeautifulSoup(driver.page_source, 'lxml') 
            time.sleep(5)
            # Beautiful Soup grabs the HTML table on the page    
            table = soup.find_all('table', attrs={'id': 'resultsTable'})
            time.sleep(5)
            df = pd.read_html(str(table),header=0)
            screener_table.append(df[0])
        except:
            driver.quit()
            sys.exit('Time out. Load time not enough!')
        
    stocks_table = pd.concat(screener_table)
    driver.quit()
    print("All Done. Happy Investing!")
    return stocks_table




#---- 1b. Scrape historical stock prices from investing.com (Using Selenium) ----        
def historical_extract_selenium(profile_link, start, end, user_agent):
    stocks_info = []
    stock_site = profile_link
    
    #define a driver instance, for example Chrome
    driver = Chrome()
    #navigate to the website
    driver.get(stock_site)
    time.sleep(5)
    # click on date picker to modify the dates
    datepicker=driver.find_element_by_xpath("//div[@id='widgetFieldDateRange']")
    datepicker.click()
    time.sleep(3)
    # 
    startdate_pick = driver.find_element_by_xpath("//input[@id='startDate']")
    startdate_pick.clear()
    startdate_pick.send_keys(start.strftime("%m/%d/%Y"))
    time.sleep(3)
    #
    enddate_pick = driver.find_element_by_xpath("//input[@id='endDate']")
    enddate_pick.clear()
    enddate_pick.send_keys(end.strftime("%m/%d/%Y"))
    time.sleep(3)
    #
    # https://stackoverflow.com/questions/38286371/selenium-python-cannot-click-on-an-element?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    apply_button = driver.find_element_by_id('applyBtn')        
    apply_button.click()
    time.sleep(3)
    #Selenium hands of the source of the specific job page to Beautiful Soup
    soup = BeautifulSoup(driver.page_source, 'lxml')
    #Beautiful Soup grabs the HTML table on the page
    table = soup.find_all('table', attrs={'id': 'curr_table'})
    
    #Giving the HTML table to pandas to put in a dataframe object
    stocks_info = pd.read_html(str(table),header=0)
    
    return stocks_info
            














