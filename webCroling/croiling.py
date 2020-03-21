import time
from datetime import datetime, timedelta
import re
import pandas as pd

ts = time.time()
""" 라이브러리 호출 """

import urllib
from bs4 import BeautifulSoup
import requests


class Croiling:
    def __init__(self):
        pass

    @staticmethod
    def naver_croiling():
        """ 회사코드 및 조회기간 설정 """
        symbol = '005930'
        startTime = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
        count = str(1000)
        """ url 설정 """
        url = 'https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&startTime={}&count={}&requestType=2'.format(
            symbol, startTime, count)
        """ 크롤링 & 전처리 """
        r = requests.get(url)
        html = r.content
        soup = BeautifulSoup(html, 'html.parser')
        tr = soup.find_all('item')
        cols = ['일자', '시가', '고가', '저가', '종가', '거래량']
        list = []
        for i in range(0, len(soup.find_all('item'))):
            list.append(re.search(r'"(.*)"', str(tr[i])).group(1).split('|'))
        df = pd.DataFrame(list, columns=cols)
        df['일자'] = pd.to_datetime(df['일자'].str[:4] + '-' + df['일자'].str[4:6] + '-' + df['일자'].str[6:])
        df.set_index(df['일자'], inplace=True)
        df = df.drop(columns='일자')
        print(f'{df.head()}')

        print('작동소요시간 :', round(time.time() - ts, 1), '초')


if __name__ == '__main__':
    Croiling.naver_croiling()
