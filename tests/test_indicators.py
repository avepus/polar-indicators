# -*- coding: utf-8 -*-
"""Tests for indicators

Created on Sat 3/26/23

@author: Avery

"""
import unittest
from datetime import datetime
import polars as pl
import polars_indicators as pi

COLUMNS = ['Date',
           'Open',
           'High',
           'Low',
           'Close',
           'Adj Close',
           'Volume',
           'Symbol']

class TestIndicators(unittest.TestCase):


    def test_simple_moving_average(self):
        """tests handling of single ticker df and multi-ticker
        sma is built in for polars so it is not tested here"""
        #test single normal case
        single = get_single_symbol_test_df()

        sma_days = 5
        ret = pi.simple_moving_average(single, sma_days)
        result = ret.df.columns
        expected = get_columns()[:-1]
        expected.append(ret.column)

        self.assertEqual(result, expected)

        #test mult normal case
        multi = get_multi_symbol_test_df()

        sma_days = 5
        ret = pi.simple_moving_average(multi, sma_days)
        result = ret.df.columns
        expected = get_columns()
        expected.append(ret.column)

        self.assertEqual(result, expected)

        #testing that calling again has doesn't add duplicate column
        ret = pi.simple_moving_average(ret.df, sma_days)
        result = ret.df.columns
        expected = get_columns()
        expected.append(ret.column)

        self.assertEqual(result, expected)




#helper test functions        
def get_columns():
    """get columns copy since testing appends"""
    return COLUMNS.copy()

def get_single_symbol_test_df():
    dates = ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-30', '2023-01-31', '2023-02-01', '2023-02-02', '2023-02-03', '2023-02-06', '2023-02-07', '2023-02-08', '2023-02-09', '2023-02-10', '2023-02-13', '2023-02-14', '2023-02-15', '2023-02-16', '2023-02-17', '2023-02-20', '2023-02-21', '2023-02-22', '2023-02-23', '2023-02-24', '2023-02-27', '2023-02-28']
    df = get_symbol_dataframe('A', dates)
    return df.select(COLUMNS[:-1])

def get_multi_symbol_test_df():
    dates = ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-30', '2023-01-31', '2023-02-01', '2023-02-02', '2023-02-03', '2023-02-06', '2023-02-07', '2023-02-08', '2023-02-09', '2023-02-10', '2023-02-13', '2023-02-14', '2023-02-15', '2023-02-16', '2023-02-17', '2023-02-20', '2023-02-21', '2023-02-22', '2023-02-23', '2023-02-24', '2023-02-27', '2023-02-28']
    return get_multi_symbol_df(['A', 'AA'], dates)

def get_multi_symbol_df(symbols: list[str], dates: list[str]):
    df_list = []
    for symbol in symbols:
        df_list.append(get_symbol_dataframe(symbol, dates))
    return pl.concat(df_list)

def get_symbol_dataframe(symbol: str, dates: list[str]) -> pl.DataFrame:
    """creates dataframe with data from '2023-03-03' for A (Agilent Technologies)"""
    list_len = len(dates)
    dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
    data = {    'Date':      dates,
                'Open':      [float(i+1) for i in range(list_len)],
                'High':      [float(i+1) for i in range(list_len)],
                'Low':       [float(i+1) for i in range(list_len)],
                'Close':     [float(i+1) for i in range(list_len)],
                'Adj Close': [float(i+1) for i in range(list_len)],
                'Volume':    [i+1 for i in range(list_len)],
                'Symbol':    [symbol] * list_len
    }
    df = pl.DataFrame(data)
    df['Symbol'].cast(pl.Categorical)
    return df
    

if __name__ == '__main__':
    unittest.main()