"""
This calculates indictors given input polars DataFrame
Each indicator can handle a LazyFrame or a DataFrame and will return the input type
This handles DataFrames for a single symbol or Dataframes with multiple symbols and a column called "Symbol" that tracks them
Each indicator will also only be added if the column doesn't already exist in the DataFrame
    An indicator is assumed to exist if a column with the name for that indicator + parameters already exists inthe DataFrame
"""

from dataclasses import dataclass
import polars as pl

SYMBOL_COLUMN = 'Symbol'

@dataclass
class IndicatorResult:
    """Holds the dataframe with the added column and the name of that column
    Every indicator returns this."""
    df: pl.DataFrame | pl.LazyFrame
    column: str

def simple_moving_average(df: pl.DataFrame | pl.LazyFrame, days: int, column: str='Close') -> pl.DataFrame | pl.LazyFrame:
    """returns dataframe with simple moving average added as a column"""
    column_name = 'SMA' + str(days)
    if column_name in df.columns:
        pass
    elif SYMBOL_COLUMN in df.columns:
        df =  df.with_columns(pl.col(column).rolling_mean(days).over(SYMBOL_COLUMN).alias(column_name))
    else:
        df = df.with_columns(pl.col(column).rolling_mean(days).alias(column_name))
    return IndicatorResult(df, column_name)


def crossover(df: pl.DataFrame | pl.LazyFrame, column1: str, column2: str, direction: str='') -> pl.DataFrame | pl.LazyFrame:
    """Adds column indicator crossover of input columns
    Can check if crossover direction of up, down, or both
    direction='' will check either direction
    THIS DOESN'T CURRENTLY HANDLE MULTISYMBOLDATAFRAMES
    ALSO NEEDS TESTS"""
    column_name = f'{column1}_crossing_{direction}_over_{column2}'

    if direction == 'up':
        df = df.with_columns(
            (
                (pl.col(column1) > pl.col(column2))
                &
                (pl.col(column1).shift(1) < pl.col(column2).shift(1))
            ).alias(column_name)
        )

    elif direction == 'down':
        df = df.with_columns(
            (
                (pl.col(column1) < pl.col(column2))
                &
                (pl.col(column1).shift(1) > pl.col(column2).shift(1))
            ).alias(column_name)
        )

    else:
        df = df.with_columns(
            (
                (
                    (pl.col(column1) > pl.col(column2))
                    &
                    (pl.col(column1).shift(1) < pl.col(column2).shift(1))
                ) | (
                    (pl.col(column1) < pl.col(column2))
                    &
                    (pl.col(column1).shift(1) > pl.col(column2).shift(1))
                )
            ).alias(column_name)
        )
    return IndicatorResult(df, column_name)


