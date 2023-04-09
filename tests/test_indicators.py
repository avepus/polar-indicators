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
        args = {"bars": 2}
        self.validate_indicator(pi.trailing_stop, args)

    def test_trailing_stop(self):
        multi = get_multi_symbol_test_df()
        index_name = 'my_index'
        bars = 2

        low_values =  [0, 1, 2, 2, 4, 5, 3, 1, 8, 9, 10, 11, 9]
        open_values = [3, 4, 5, 6, 7, 8, 9, 1, 10, 11, 12, 13, 14]

        df = multi.slice(0, len(low_values)).select(pl.exclude(pi.LOW_COLUMN, pi.OPEN_COLUMN))

        df = df.insert_at_idx(-1, pl.Series(pi.LOW_COLUMN, low_values, dtype=pl.Float64))
        df = df.insert_at_idx(-1, pl.Series(pi.OPEN_COLUMN, open_values, dtype=pl.Float64))
        
        ret = pi.trailing_stop(df, bars)

        #validate the correct indicies got values
        result = ret.df.with_row_count(index_name).filter(pl.col(ret.column).is_not_null())[index_name].to_list()
        expected = [6, 7, 12]
        self.assertEqual(result, expected)

        #validate the values are correct
        result = ret.df.filter(pl.col(ret.column).is_not_null())[ret.column].to_list()
        expected = [4, 1, 10]
        self.assertEqual(result, expected)
        

    def test_create_trade_ids_validate(self):
        args = {"enter_column": "Bool",
                "exit_column": "Bool"}
        self.validate_indicator(pi.create_trade_ids, args)

    def test_create_trade_ids(self):
        multi = get_multi_symbol_test_df()
        index_name = "my_index"
        enter_column = "enter"
        enter_values = [None, 1.2, 1.4, None, 9.9, None, None, 3, None, None]
        exit_column = "exit"
        exit_values = [0.3, None, None, 0.8, 1.1, None, 2, None, None, None]
                
        df = multi.with_row_count(index_name).filter(pl.col(index_name) < 10)

        df = df.insert_at_idx(-1, pl.Series(enter_column, enter_values))
        df = df.insert_at_idx(-1, pl.Series(exit_column, exit_values))

        ret = pi.create_trade_ids(df, enter_column, exit_column)

        result = ret.df[ret.column].to_list()
        expected = [None, 1, 1, 1, 2, None, None, 3, 3, 3]

        self.assertEqual(result, expected)

    def test_validate_targeted_value(self):
        args = {"targets": "High"}
        self.validate_indicator(pi.targeted_value, args)


    def test_validate_limit_entries(self):
        args = {"bars": 5,
                "entries": "High"}
        self.validate_indicator(pi.limit_entries, args) 


    def test_limit_entries(self):
        multi = get_multi_symbol_test_df()
        index_name = 'my_index'
        bars = 2

        enter_column = "enter"
        enter_values = [None, 1.2, 1.4, 1.9, 9.9, None, None, 3, None, 2]
        expected_values = [None, 1.2, None, None, 9.9, None, None, 3, None, None]
                
        df = multi.with_row_count(index_name).filter(pl.col(index_name) < 10)
        df = df.insert_at_idx(len(df.columns), pl.Series(enter_column, enter_values))

        expected = df.clone()

        limit = pi.limit_entries(df, bars, enter_column)
        result = limit.df

        expected = expected.insert_at_idx(len(expected.columns), pl.Series(limit.column, expected_values))

        testing.assert_frame_equal(result, expected)


    def test_validate_end_of_data_stop(self):
        args = {}
        self.validate_indicator(pi.end_of_data_stop, args) 


    def test_end_of_data_stop(self):
        multi = get_multi_symbol_test_df()
        expected_values = [None] * 20
        expected_values[9] = 10.0
        expected_values[19] = 10.0

        eods = pi.end_of_data_stop(multi)
        result = eods.df

        expected = multi.clone()
        expected.insert_at_idx(len(expected.columns), pl.Series(eods.column, expected_values))

        testing.assert_frame_equal(result, expected)


        
    



    # #kept in case needed later. tests nested add_exit_ids used in create_trade_ids
    # def test_add_exit_ids(self):
    #     multi = get_multi_symbol_test_df()
    #     index_name = 'my_index'
    #     column_name = "test"

    #     df = multi.with_row_count(name=index_name).with_columns( 
    #             pl.when(
    #                 pl.col(index_name) == 3) \
    #                 .then(pl.lit(True)) \
    #             .when(
    #                 (pl.col(index_name) == 4)) \
    #                 .then(pl.lit(True)) \
    #             .when(
    #                 (pl.col(index_name) == 7)) \
    #                 .then(pl.lit(True)) \
    #             .otherwise(
    #                 pl.lit(False)) \
    #             .alias(column_name))
                
    #     df = df.filter(pl.col(index_name) < 10)

    #     ret = pi.create_trade_ids(df, column_name)

    #     result = ret.df[ret.column].to_list()
    #     expected = [None, 0, 0, 0, 1, 2, 2, 2, 3, 3]

    #     self.assertEqual(result, expected)





#helper test functions       

def add_column_with_true_indicies(df, indicies: list[int], new_column: str):
    """adds a column where the input list of indicies are True and all other rows are false"""
    index = "index"
    new_columns = df.columns.copy()
    new_columns.append(new_column)
    return df.with_row_count(name=index).with_columns( 
            pl.when(
                pl.col(index).is_in(indicies)) \
                .then(pl.lit(True)) \
            .otherwise(
                pl.lit(False)) \
            .alias(new_column)).select(new_columns) 

def get_columns():
    """get columns copy since testing appends"""
    return COLUMNS.copy()

def get_single_symbol_test_df() -> pl.DataFrame:
    dates = ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13']#, '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-30', '2023-01-31']
    df = get_symbol_dataframe('A', dates)
    return df.select(COLUMNS[:-1])

def get_multi_symbol_test_df() -> pl.DataFrame:
    dates = ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13']#, '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-30', '2023-01-31']
    return get_multi_symbol_df(['A', 'AA'], dates)

def get_multi_symbol_df(symbols: list[str], dates: list[str]) -> pl.DataFrame:
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