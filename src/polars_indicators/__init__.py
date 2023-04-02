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

def simple_moving_average(df: pl.DataFrame | pl.LazyFrame, days: int, column: str='Close') -> IndicatorResult:
    """returns dataframe with simple moving average added as a column"""
    column_name = 'SMA' + str(days)
    if column_name in df.columns:
        pass
    elif SYMBOL_COLUMN in df.columns:
        df =  df.with_columns(pl.col(column).rolling_mean(days).over(SYMBOL_COLUMN).alias(column_name))
    else:
        df = df.with_columns(pl.col(column).rolling_mean(days).alias(column_name))
    return IndicatorResult(df, column_name)

def crossover_up(df: pl.DataFrame | pl.LazyFrame, column1: str, column2: str) -> IndicatorResult:
    """Adds column indicator crossover made by column1 over column2 in the upward direction
    This does not handle situations where the values are the same.
    I.e. the example below would not be considered a crossover
        day 1: column1 < column2 
        day 2: column1 == column2
        day 3: column1 > column2"""

    column_name = column1 + '_cross_up_' + column2

    if column_name in df.columns:
        return IndicatorResult(df, column_name)

    same_symbol = 'same_symbol'
    cross_up = 'cross_up'

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

    lf = lf.with_columns((pl.col(cross_up) & pl.col(same_symbol)).alias(column_name))
    lf = lf.select(new_columns)

    if isinstance(df, pl.LazyFrame):
        df = lf
    else:
        df = lf.collect()

    return IndicatorResult(df, column_name)

def crossover_down(df: pl.DataFrame | pl.LazyFrame, column1: str, column2: str) -> IndicatorResult:
    """Adds column indicator crossover made by column1 over column2 in the downward direction
    This does not handle situations where the values are the same.
    I.e. the example below would not be considered a crossover
        day 1: column1 < column2 
        day 2: column1 == column2
        day 3: column1 > column2"""
    column_name = column1 + '_cross_down_' + column2
    if column_name in df.columns:
        return IndicatorResult(df, column_name)

    
    cross_up = crossover_up(df, column2, column1) #cross up with columns flipped is the same as cross cown

    #this leaves a corner case where you can add a crossover up for col1 and col2 and then later add a crossover down for col2 and col1. This would rename the first column. The columns are identical though so just don't do that
    df = cross_up.df.rename({cross_up.column: column_name})
    return IndicatorResult(df, column_name)


def crossover(df: pl.DataFrame | pl.LazyFrame, column1: str, column2: str) -> IndicatorResult:
    """Adds column indicator crossover of input columns
    This does not handle situations where the values are the same.
    I.e. the example below would not be considered a crossover
        day 1: column1 < column2 
        day 2: column1 == column2
        day 3: column1 > column2"""

    column_name = column1 + '_cross_' + column2

    if column_name in df.columns:
        return IndicatorResult(df, column_name)

    new_columns = df.columns.copy()
    new_columns.append(column_name)

    lf = df.lazy()

    cross_up = crossover_up(lf, column1, column2)
    cross_down = crossover_down(cross_up.df, column1, column2)
    
    lf = cross_down.df.with_columns((pl.col(cross_up.column) | pl.col(cross_down.column)).alias(column_name))

    lf = lf.select(new_columns)

    if isinstance(df, pl.LazyFrame):
        df = lf
    else:
        df = lf.collect()

    return IndicatorResult(df, column_name)


def trailing_stop(df: pl.DataFrame | pl.LazyFrame, bars, column = "Low") -> IndicatorResult:
    """adds column of bool indicating when trailing stop hit"""
    column_name = f"{bars}_bars_{column}_stop"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    df = df.with_columns((pl.col(column) < pl.col(column).rolling_min(bars).shift(1)).alias(column_name))

    return IndicatorResult(df, column_name)



def create_trade_ids_old(df: pl.DataFrame | pl.LazyFrame, enter_column: str, exit_column: str) -> IndicatorResult:
    """increments count for each true value in input column
    intended to be used to create IDs for trades"""
    column_name = f"{enter_column}/{exit_column}"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    exit_ids = 'exit_ids'
    
    df = df.with_columns(pl.col(exit_column).cumsum().shift(1).alias(exit_ids))

    df = df.with_columns(pl.when(pl.col(enter_column).max().over(pl.col(exit_ids))).then(pl.col(exit_ids)).alias(column_name))
    return IndicatorResult(df, column_name)

def create_trade_ids(df: pl.DataFrame | pl.LazyFrame, enter_column: str, exit_column: str) -> IndicatorResult:
    """Adds a column that undiquely identifies each trade with an integer
    All bars of the trade will get a single value.
    This considers the bars of a trade to be from the first 'True' value in enter_column to the first
    True in the exit_column on the same or later bar
    Entries indicated in the enter_column will be ignored if there is already an active trade for that symbol
        i.e. this can only track one active trade per symbol at a time"""
    def add_exit_ids(df: pl.DataFrame | pl.LazyFrame, exit_column: str) -> IndicatorResult:
        """adds a column that counts up on each exit"""
        column_name = f"{exit_column}_exit_ids"
        if column_name in df.columns:
            return IndicatorResult(df, column_name)
        df = df.with_columns(pl.col(exit_column).cumsum().shift(1).alias(column_name))
        return IndicatorResult(df, column_name)
    def add_traded_column(df: pl.DataFrame | pl.LazyFrame, enter_column:str, exit_ids:str) -> IndicatorResult:
        """adds a column thats true for every bar that has an active trade"""
        column_name = f"{enter_column}/{exit_ids}_traded"
        if column_name in df.columns:
            return IndicatorResult(df, column_name)
        df = df.with_columns(
            pl.col(enter_column).max().over(pl.col(exit_ids)).alias(column_name)) #ensures we only count trades with entries
        return IndicatorResult(df, column_name)
    
    column_name = f"{enter_column}/{exit_column}"
    if column_name in df.columns:
            return IndicatorResult(df, column_name)
    
    new_columns = df.columns.copy()
    new_columns.append(column_name)
    
    lf = df.lazy()
    exit_ids = add_exit_ids(lf, exit_column)
    traded = add_traded_column(exit_ids.df, enter_column, exit_ids.column)

    lf = traded.df.with_columns(
        pl.when(
            pl.col(traded.column) #pl.col(enter_column).max().over(pl.col(exit_ids)) #ensures we only count trades with entries
            &
            (pl.col("Date") == pl.col("Date").min().over(exit_ids.column)) #ensures we only count a trade once per exit
        ).then(1).otherwise(0).alias(column_name))
    
    lf = lf.with_columns(pl.when(pl.col(traded.column)).then(pl.col(column_name).cumsum()))
    lf = lf.select(new_columns)

    df = lf if isinstance(df, pl.LazyFrame) else lf.collect()

    return IndicatorResult(df, column_name)
    
def summarize_trades(df: pl.DataFrame | pl.LazyFrame, trade_id_column: str) -> pl.DataFrame | pl.LazyFrame:
    """summarizes trade information given ids in input column
    PROTOTYPE. NEEDS MORE WORK AND MAY NOT BE THE DIRECTION I GO"""
    index_name = "index"
    df = df.with_row_count(index_name).groupby(trade_id_column).agg(
              pl.col("Date").min().alias("Start"),
              pl.col("Date").max().alias("End"),
              pl.col("Low").min().alias("Lowest"),
              pl.col("High").max().alias("Highest"),
              (pl.col(index_name).max() - pl.col(index_name).min() + 1).alias("length")
    ).sort(trade_id_column)

    return df
    

    


