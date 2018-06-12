# -*- coding: utf-8 -*-
"""
Created in 2018

@author: JiaRong
"""

import numpy as np
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

# Reference table for "country_code_investingcom":
# Asia: (1)Malaysia: 42, (2)Singapore: 36, (3)Australia: 25, (4)China: 37, (5)Japan: 35, (6)Hong Kong: 39
# American: (2)United States: 5
# Euro: (1)United Kingdom: 4, (2)Germany: 17

#---- 1a. Scrape listed stocks from investing.com (Using requests) ----    
def get_stocks_info(country_code_investingcom, user_agent):
    
    search_investing_url = 'https://www.investing.com/stock-screener/Service/SearchStocks'
    country_referer = str(np.where(country_code_investingcom==42, 'https://www.investing.com/stock-screener/?sp=country::' + str(country_code_investingcom) + '|sector::a|industry::a|equityType::a|exchange::62%3Ceq_market_cap;',
                               'https://www.investing.com/stock-screener/?sp=country::' + str(country_code_investingcom) +'|sector::a|industry::a|equityType::a%3Ceq_market_cap;'))
    stocks_info = None

    def dictionary_edit(dictionary, keys_edited):
        keys, values = zip(*dictionary[keys_edited].items())
        dictionary.update(dictionary[keys_edited])
        del dictionary[keys_edited]
        return dictionary
    
    # function used to extract screener list
    def stocks_extract(country_code_investingcom, page, parse=True):
        page_updated = str(page)
        
        # Start a request session
        session = requests.Session()
        session.headers["User-Agent"] = user_agent
        session.headers['Referer'] = country_referer + page_updated
        session.headers['X-Requested-With'] = 'XMLHttpRequest'
        
        params_form = {
           'country[]' : country_code_investingcom,
           'sector' : '7,5,12,3,8,9,1,6,2,4,10,11',
           'industry' : '81,56,59,41,68,67,88,51,72,47,12,8,50,2,71,9,69,45,46,13,94,102,95,58,100,101,87,31,6,38,79,30,77,28,5,60,18,26,44,35,53,48,49,55,78,7,86,10,1,34,3,11,62,16,24,20,54,33,83,29,76,37,90,85,82,22,14,17,19,43,89,96,57,84,93,27,74,97,4,73,36,42,98,65,70,40,99,39,92,75,66,63,21,25,64,61,32,91,52,23,15,80',
           'equityType' : 'ORD,DRC,Preferred,Unit,ClosedEnd,REIT,ELKS,OpenEnd,ParticipationShare,CapitalSecurity,PerpetualCapitalSecurity,GuaranteeCertificate,IGC,Warrant,SeniorNote,Debenture,ETF,ADR,ETC,ETN',
           'exchange[]': '62',
           'pn': page_updated,
           'order[col]' : 'name_trans',
           'order[dir]' : 'd'
        }
    
        base_page = session.post(url = search_investing_url, data = params_form)
        #print(base_page.content.decode("utf-8"))
        #base_page.headers
        #print(len(base_page.content)) 
        
        if parse==True:
            json_result = base_page.json()['hits']
            #print(json.dumps(json_result, indent=4))
            json_result = list(map(lambda x: dictionary_edit(x, keys_edited='viewData'), json_result))
            df = pd.DataFrame.from_dict(json_result)
            return df
        else:
            number_pages = base_page.json()['paginationHTML']
            number_pages = int(re.findall(r'(class=\"(pagination)\")\>(\d+)', number_pages)[-1][-1])
            return number_pages

    # Get the number of pages of screener results
    first_page = 0
    page_string = str(first_page)
    last_page = stocks_extract(country_code_investingcom = country_code_investingcom, page = page_string, parse=False)

    # Extract stocks information from screener list
    for page in range(1,(last_page+1)):
        info_table = stocks_extract(country_code_investingcom = country_code_investingcom, page = page, parse=True)
        if stocks_info is None:
            stocks_info = pd.DataFrame(info_table)
        else:
            stocks_info = stocks_info.append(info_table)
        print("We have successfully scrapped the information of stocks listed on country code " + str(country_code_investingcom) + ' -page' + str(page))
    
    stocks_info = stocks_info.sort_values('name')
    stocks_info = stocks_info.reset_index()
    stocks_info = stocks_info.drop(columns = 'index')
    stocks_info['split_link'] = stocks_info['link'].str.contains('?', regex=False)
    
    stocks_info = stocks_info.assign(
            price_weblink = 'https://www.investing.com' + stocks_info['link'].str.split("?").apply(lambda x: x[0]) + '-historical-data')
    
    indexf = [(index) for (index, x) in enumerate(stocks_info['split_link']) if x==True]
    for i in indexf:
        stocks_info.loc[i,'price_weblink'] = 'https://www.investing.com' + stocks_info.loc[i,'link'].split("?")[0] + '-historical-data' + '?' + stocks_info.loc[i,'link'].split("?")[1]
  
    columns_keep = ['name','stock_symbol','security_type','sector_trans','industry_trans','link','price_weblink','pair_ID','exchange_ID','exchange_trans','flag','last','pair_change_percent',
                    'daily','week','month','ytd','eq_one_year_return','3year','a52_week_high','a52_week_low','a52_week_high_diff','a52_week_low_diff','month_change','tech_sum_300_constant',
                    'tech_sum_900_constant','tech_sum_1800_constant','tech_sum_3600_constant','tech_sum_86400_constant','tech_sum_week_constant','tech_sum_month_constant','eq_market_cap',
                    'turnover_volume','avg_volume','eq_pe_ratio','eq_revenue','eq_eps','eq_beta','eq_dividend','yield_us','peexclxor_us','ttmpr2rev_us','aprfcfps_us','ttmprfcfps_us',
                    'price2bk_us','pr2tanbk_us','epschngyr_us','ttmepschg_us','epstrendgr_us','revchngyr_us','ttmrevchg_us','revtrendgr_us','csptrendgr_us','ttmastturn_us','ttminvturn_us','ttmrevpere_us',
                    'ttmniperem_us','ttmrecturn_us','ttmgrosmgn_us','grosmgn5yr_us','ttmopmgn_us','opmgn5yr_us','ttmptmgn_us','ptmgn5yr_us','ttmnpmgn_us','margin5yr_us','qquickrati_us','qcurratio_us',
                    'qltd2eq_us','qtotd2eq_us','yld5yavg_us','divgrpct_us','ttmpayrat_us','ADX','ATR','BullBear','CCI','HL','ROC','RSI','STOCH','STOCHRSI','UO','WilliamsR','MACD']

    new_column_name = ['name','stock_symbol','security_type','sector','industry','link','price_weblink','pair_ID','exchange_ID','exchange','country','last','pair_change_percent(%)',
                       'daily_change(%)','week_change(%)','month_change(%)','ytd_change(%)','one_year_return(%)','3year_change(%)','52_week_high','52_week_low','52_week_high_diff(%)','52_week_low_diff(%)','month_change','summary_5mins',
                       'summary_15mins','summary_30mins','summary_hourly','summary_daily','summary_weekly','summary_monthly','market_cap','turnover_volume','average_volume','pe_ratio','revenue','eps','beta','dividend',
                       'dividend_yield(%)','pe_ratio_TTM','price_to_sales_TTM','price_to_cash_flow_MRQ','price_to_free_cash_flow_TTM','price_to_book_MRQ','price_to_tangible_book_MRQ','eps_MRQ_change_1yr(%)','eps_TTM_change_1yr(%)',
                       'eps_growth_5yr(%)','sales_MRQ_change_1yr(%)','sales_TTM_change_1yr(%)','sales_growth_5yr(%)','capitalspending_growth_5yr(%)','asset_turnover','inventory_turnover','revenue_per_employee(thousand)',
                       'netincome_per_employee(thousand)','receivable_turnover','gross_margin_TTM(%)','gross_margin_5yr(%)','operating_margin_TTM(%)','operating_margin_5yr(%)','pretax_margin_TTM(%)','pretax_margin_5yr(%)',
                       'netprofit_margin_TTM(%)','netprofit_margin_5yr(%)','quick_ratio','current_ratio','LT_debt_to_equity_MRQ(%)','total_debt_to_equity(%)','dividend_yield_avg_5yr(%)','dividend_growth_rate(%)',
                       'payout_ratio','ADX','ATR','BullBear','CCI','HL','ROC','RSI','STOCH','STOCHRSI','UO','WilliamsR','MACD']
    
    # Keep and rename neccessary columns
    stocks_info = stocks_info[columns_keep]
    stocks_info.columns = new_column_name
    return stocks_info


#---- 1b. Scrape historical prices from investing.com (Using requests) ----        
# requests get and post to extract historical prices
def historical_extract(profile_link, start, end, user_agent):
    #print(profile_link)
    
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    
    res = session.get(profile_link)
    content = res.text
    ro1 = re.search(r"window.siteData.smlID = (\d+);", content)
    sml_id = ro1.group(1)
    ro2 = re.search(r"pairId: (\d+),", content)
    curr_id = ro2.group(1)
    ro3 = re.search(r"\((\w+)\)",re.search(r"<title>\b(.*?)\b<\/title>", content).group(1))
    header = ro3.group(1)
    soup = BeautifulSoup(content, 'lxml')
    code = soup.find_all('span', attrs={'class':'elp'})[-1].text.rjust(4, '0') 
    
    klse_url = "https://klse.i3investor.com/servlets/stk/" + code + ".jsp"
    page = requests.get(url = klse_url, headers = {'User-Agent': user_agent})
    soup2 = BeautifulSoup(page.content, 'html.parser')
    sector_text  = re.sub('.* : ','', soup2.find('span', attrs={'class' : 'boarAndSector'}).text).replace(';', '')
    
    session.headers['Referer'] = profile_link
    session.headers['X-Requested-With'] = 'XMLHttpRequest'
    
    historical_url = 'https://www.investing.com/instruments/HistoricalDataAjax'
    header_updated = header + ' Historical Data'
    historical_param = {
            'curr_id': curr_id,                       #'950488'
            'smlID': sml_id,                          #'1682012',
            'header': header_updated,                 #'SEVE Historical Data',
            'st_date': start.strftime("%m/%d/%Y"),    #'04/02/2018',
            'end_date': end.strftime("%m/%d/%Y"),     #'05/24/2018',
            'interval_sec': 'Daily',
            'sort_col': 'date',
            'sort_ord': 'DESC',
            'action': 'historical_data'
    }

    response = session.post(url = historical_url, data = historical_param)
    df = pd.read_html(response.text)[0]
    return(df, sector_text, code)

