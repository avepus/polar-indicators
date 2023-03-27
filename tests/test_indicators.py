# -*- coding: utf-8 -*-
"""Tests for indicators

Created on Sat 3/26/23

@author: Avery

"""
import unittest
from datetime import datetime
from polars import testing
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

    def validate_indicator(self, function, args):
        """this runs tests that should pass on every indicator
        it tests that running the indicator adds a single column for that indicator
        also verifies that running the indicator twice returns the same df the second time"""
        #test single normal case
        single = get_single_symbol_test_df()

        ret = function(single, **args)
        result = ret.df.columns
        expected = get_columns()[:-1]
        expected.append(ret.column)

        self.assertEqual(result, expected)

        #test multi normal case
        multi = get_multi_symbol_test_df()

        ret = function(multi, **args)
        result = ret.df.columns
        expected = get_columns()
        expected.append(ret.column)

        self.assertEqual(result, expected)

        #testing that calling again has doesn't add duplicate column
        expected = ret.df
        ret = function(expected, **args)
        result = ret.df

        testing.assert_frame_equal(result,expected)


    def test_simple_moving_average(self):
        """tests handling of single ticker df and multi-ticker
        sma is built in for polars so it is not tested here"""
        #test single normal case
        args = {'days': 5}
        self.validate_indicator(pi.simple_moving_average, args)


    def test_crossover(self):
        pass




#helper test functions        
def get_columns():
    """get columns copy since testing appends"""
    return COLUMNS.copy()

def get_single_symbol_test_df():
    dates = ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13']#, '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-30', '2023-01-31']
    df = get_symbol_dataframe('A', dates)
    return df.select(COLUMNS[:-1])

def get_multi_symbol_test_df():
    dates = ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13']#, '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-30', '2023-01-31']
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