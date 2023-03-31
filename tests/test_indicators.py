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
           'Bool',
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

        #test sngle LazyFrame
        single_lf = single.lazy()

        ret = function(single_lf, **args)
        result = ret.df

        self.assertIsInstance(result, pl.LazyFrame)

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


    def test_validate_simple_moving_average(self):
        """tests handling of single ticker df and multi-ticker
        sma is built in for polars so it is not tested here"""
        #test single normal case
        args = {'days': 5}
        self.validate_indicator(pi.simple_moving_average, args)


    def test_validate_crossover_up(self):
        args = {'column1': 'Close',
                'column2': 'Open'}
        self.validate_indicator(pi.crossover_up, args)

    
    def test_crossover_up(self):
        multi = get_multi_symbol_test_df()
        index_name = 'my_index'
        column_name = 'test'
        
        #this creates a dataframe with a cross up on row 5 (index 4)
        df = multi.with_row_count(name=index_name).with_columns( 
                pl.when(
                    pl.col(index_name) == 3) \
                    .then(pl.col(index_name) - 1) \
                .otherwise(
                    pl.col(index_name) + 1) \
                .alias(column_name))
        

        ret = pi.crossover_up(df, column_name, index_name)
        result = ret.df.filter(pl.col(ret.column) == True)[index_name].to_list()
        expected = [4]

        self.assertEqual(result, expected)

        #test multisymbols. A crossover should not be logged from end of one sybmol to start of another
        first_symbol = multi[pi.SYMBOL_COLUMN].to_list()[0]
        df = multi.with_row_count(name=index_name).with_columns( 
                pl.when(
                    pl.col(pi.SYMBOL_COLUMN) == first_symbol) \
                    .then(pl.col(index_name) - 1) \
                .otherwise(
                    pl.col(index_name) + 1) \
                .alias(column_name))
        
        ret = pi.crossover_up(df, column_name, index_name)

        crossovers = ret.df.filter(pl.col(ret.column) == True)

        self.assertTrue(crossovers.is_empty(), "crossover up found between symbols")


    def test_validate_crossover_down(self):
        args = {'column1': 'Close',
                'column2': 'Open'}
        self.validate_indicator(pi.crossover_down, args)
        
    def test_crossover_down(self):
        multi = get_multi_symbol_test_df()
        index_name = 'my_index'
        column_name = 'test'
        
        #this creates a dataframe with a forced cross under on row 4 (index 3)
        df = multi.with_row_count(name=index_name).with_columns( 
                pl.when(
                    pl.col(index_name) == 3) \
                    .then(pl.col(index_name) - 1) \
                .otherwise(
                    pl.col(index_name) + 1) \
                .alias(column_name))
        

        ret = pi.crossover_down(df, column_name, index_name)
        result = ret.df.filter(pl.col(ret.column) == True)[index_name].to_list()
        expected = [3]

        self.assertEqual(result, expected)

        #test multisymbols. A crossover should not be logged from end of one sybmol to start of another
        first_symbol = multi[pi.SYMBOL_COLUMN].to_list()[0]
        df = multi.with_row_count(name=index_name).with_columns( 
                pl.when(
                    pl.col(pi.SYMBOL_COLUMN) == first_symbol) \
                    .then(pl.col(index_name) + 1) \
                .otherwise(
                    pl.col(index_name) - 1) \
                .alias(column_name))
        
        ret = pi.crossover_down(df, column_name, index_name)

        crossovers = ret.df.filter(pl.col(ret.column) == True)

        self.assertTrue(crossovers.is_empty(), "crossover up found between symbols")

    def test_validate_crossover(self):
        args = {'column1': 'Close',
                'column2': 'Open'}
        self.validate_indicator(pi.crossover, args)

    def test_crossover(self):
        multi = get_multi_symbol_test_df()
        index_name = 'my_index'
        column_name = 'test'
        
        #this creates a dataframe with a forced cross under on row 4 (index 3) and a cross up on row 5 (index 4)
        df = multi.with_row_count(name=index_name).with_columns( 
                pl.when(
                    pl.col(index_name) == 3) \
                    .then(pl.col(index_name) - 1) \
                .otherwise(
                    pl.col(index_name) + 1) \
                .alias(column_name))
        
        ret = pi.crossover(df, column_name, index_name)
        result = ret.df.filter(pl.col(ret.column) == True)[index_name].to_list()
        expected = [3, 4]

        self.assertEqual(result, expected)

        #test multisymbols. A crossover should not be logged from end of one sybmol to start of another
        first_symbol = multi[pi.SYMBOL_COLUMN].to_list()[0]
        df = multi.with_row_count(name=index_name).with_columns( 
                pl.when(
                    pl.col(pi.SYMBOL_COLUMN) == first_symbol) \
                    .then(pl.col(index_name) + 1) \
                .otherwise(
                    pl.col(index_name) - 1) \
                .alias(column_name))
        
        ret = pi.crossover(df, column_name, index_name)

        crossovers = ret.df.filter(pl.col(ret.column) == True)

        self.assertTrue(crossovers.is_empty(), "crossover found between symbols")


    def test_trailing_stop_validate(self):
        args = {"bars": 2,
                "column": 'Close'}
        self.validate_indicator(pi.trailing_stop, args)

    def test_trailing_stop(self):
        multi = get_multi_symbol_test_df()
        index_name = 'my_index'
        column_name = "test"
        bars = 2
        not_enough_to_hit_stop = 1
        enough_to_hit_stop = 3

        df = multi.with_row_count(name=index_name).with_columns( 
                pl.when(
                    pl.col(index_name) == 3) \
                    .then(pl.col(index_name) - not_enough_to_hit_stop) \
                .when(
                    (pl.col(index_name) == 6)) \
                    .then(pl.col(index_name) - enough_to_hit_stop) \
                .when(
                    (pl.col(index_name) == 7)) \
                    .then(pl.col(index_name) - enough_to_hit_stop - enough_to_hit_stop) \
                .when(
                    (pl.col(index_name) == 12)) \
                    .then(pl.col(index_name) - enough_to_hit_stop) \
                .otherwise(
                    pl.col(index_name)) \
                .alias(column_name))
        
        ret = pi.trailing_stop(df, bars, column_name)
        result = ret.df.filter(pl.col(ret.column) == True)[index_name].to_list()
        expected = [6, 7, 12]
        self.assertEqual(result, expected)
        

    def test_create_trade_ids_validate(self):
        args = {"column": 'Bool'}
        self.validate_indicator(pi.create_trade_ids, args)

    def test_create_trade_ids(self):
        multi = get_multi_symbol_test_df()
        index_name = 'my_index'
        column_name = "test"

        df = multi.with_row_count(name=index_name).with_columns( 
                pl.when(
                    pl.col(index_name) == 3) \
                    .then(pl.lit(True)) \
                .when(
                    (pl.col(index_name) == 4)) \
                    .then(pl.lit(True)) \
                .when(
                    (pl.col(index_name) == 7)) \
                    .then(pl.lit(True)) \
                .otherwise(
                    pl.lit(False)) \
                .alias(column_name))
                
        df = df.filter(pl.col(index_name) < 10)

        ret = pi.create_trade_ids(df, column_name)

        result = ret.df[ret.column].to_list()
        expected = [None, 0, 0, 0, 1, 2, 2, 2, 3, 3]

        self.assertEqual(result, expected)






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
                'Adj Close': [float(1) if i == 3 else float(i+1) for i in range(list_len)],
                'Volume':    [i+1 for i in range(list_len)],
                'Bool':      [False] * list_len,
                'Symbol':    [symbol] * list_len
    }
    df = pl.DataFrame(data)
    df['Symbol'].cast(pl.Categorical)
    return df
    

if __name__ == '__main__':
    unittest.main()