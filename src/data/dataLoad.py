import requests
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
from src.logs.loggingProject import logger
import pandas as pd


def get_rate_by_day(date, currencies):
    formatted_date = date.strftime('%d/%m/%Y')
    url = f"https://www.cbr.ru/scripts/XML_daily.asp?date_req={formatted_date}"
    response = requests.get(url)

    if response.status_code == 200:
        logger.info(f"Successfully load data for {date} response_code = 200")
        root = ET.fromstring(response.content)
        res = {"date": date.strftime('%Y-%m-%d')
               }
        for child in root:
            for currency in currencies:
                if child.find('CharCode').text == currency:
                    res[f"rate_{currency}"] = float(child.find('Value').text.replace(',', '.'))

        if len(res.keys()) > 1:
            return res
        else:
            logger.warn(f"data for {date} does not contain data for currencies {currencies}")

    logger.warn(f"No data for {date} response_code = {response.status_code}")
    return None

def get_rates_between_dates(start_date, end_date, currencies):
    logger.info(f"start data load for {start_date.date()} and {end_date.date()}")
    rates = []
    current_date = start_date
    while current_date <= end_date:
        rate = get_rate_by_day(current_date, currencies)
        if rate:
            rates.append(rate)
        current_date += timedelta(days=1)
    logger.info(f"Successfully  data load")
    return pd.DataFrame(rates)

def saving_data(data, data_name):
    logger.info(f"start data saving {data_name}")
    data.to_csv(f'./data/{data_name}.csv')
    logger.info(f"Successfully saving {data_name}")
    