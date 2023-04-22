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
        enter_column = "enter"
        enter_values = []
        exit_colume = "exit"
        exit_values = []

        data = {pi}