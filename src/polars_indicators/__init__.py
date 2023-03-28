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
    This does not handle situations where the values are the same.
    I.e. the example below would not be considered a crossover
        day 1: column1 < column2 
        day 2: column1 == column2
        day 3: column1 > column2"""

    column_name = column1 + '_cross_' + (direction + '_' + column2 if direction else column2)

    if column_name in df.columns:
        return IndicatorResult(df, column_name)

    same_symbol = 'same_symbol'
    cross_up = 'cross_up'
    cross_down = 'cross_down'

    new_columns = df.columns.copy()
    new_columns.append(column_name)

    lf = df.lazy()
    
    if SYMBOL_COLUMN in lf.columns:
        lf = lf.with_columns((pl.col(SYMBOL_COLUMN).shift(1) == pl.col(SYMBOL_COLUMN)).alias(same_symbol))
    else:
        lf = lf.with_columns(pl.lit(True).alias(same_symbol))


    lf = lf.with_columns(
            (
                (pl.col(column1) > pl.col(column2))
                &
                (pl.col(column1).shift(1) < pl.col(column2).shift(1))
            ).alias(cross_up)
        )

    lf = lf.with_columns(
        (
            (pl.col(column1) < pl.col(column2))
            &
            (pl.col(column1).shift(1) > pl.col(column2).shift(1))
        ).alias(cross_down)
    )

    if direction == 'up':
        lf = lf.with_columns((pl.col(cross_up) & pl.col(same_symbol)).alias(column_name))
    elif direction == 'down':
        lf = lf.with_columns((pl.col(cross_down) & pl.col(same_symbol)).alias(column_name))
    else:
        lf = lf.with_columns(((pl.col(cross_up) | pl.col(cross_down)) & pl.col(same_symbol)).alias(column_name))

    df = lf.select(new_columns).collect()

    return IndicatorResult(df, column_name)


