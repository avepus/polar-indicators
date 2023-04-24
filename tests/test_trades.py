# -*- coding: utf-8 -*-
"""Tests for indicators

Created on Sat 3/26/23

@author: Avery

"""
import unittest
from datetime import datetime
from polars import testing
import polars as pl
from polars_indicators import indicators
from . import test_indicators

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

    def test_init(self):
        """verify df of trades"""
        
        dates = ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13'] 
        symbol_list = ["A"] * len(dates)
        symbol_list.extend(["AA"] * len(dates))
        dates = dates * 2 #we have two symvbols so we need to repeat dates
        list_len = len(dates)
        


        dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
        data = {    'Date':      dates,
                    'Open':      [3.0] * list_len,
                    'High':      [8.0] * list_len,
                    'Low':       [1.0] * list_len,
                    'Close':     [4.0] * list_len,
                    'Adj Close': [4.0] * list_len,
                    'Volume':    [40] * list_len,
                    'Symbol':    symbol_list
        }
        df = pl.DataFrame(data)
        df['Symbol'].cast(pl.Categorical)


        enter_column = "enter"
        enter_values = []
        exit_colume = "exit"
        exit_values = []