"""
This calculates indictors given input polars DataFrame
Each indicator can handle a LazyFrame or a DataFrame and will return the input type
This handles DataFrames for a single symbol or Dataframes with multiple symbols and a column called "Symbol" that tracks them
Each indicator will also only be added if the column doesn't already exist in the DataFrame
    An indicator is assumed to exist if a column with the name for that indicator + parameters already exists inthe DataFrame
"""

from dataclasses import dataclass
import polars as pl


SYMBOL_COLUMN = "Symbol"
DATE_COLUMN = "Date"
LOW_COLUMN = "Low"
HIGH_COLUMN = "High"
OPEN_COLUMN = "Open"
CLOSE_COLUMN = "Close"
VOLUMNE_COLUMN = "Volume"

@dataclass
class IndicatorResult:
    """Holds the dataframe with the added column and the name of that column
    Every indicator returns this."""
    df: pl.DataFrame | pl.LazyFrame
    column: str

def simple_moving_average(df: pl.DataFrame | pl.LazyFrame, days: int, column: str='Close') -> IndicatorResult:
    """returns dataframe with simple moving average added as a column"""
    column_name = f"SMA{days}_{column}"
    if column_name in df.columns:
        pass
    elif SYMBOL_COLUMN in df.columns:
        df =  df.with_columns(pl.col(column).rolling_mean(days).over(SYMBOL_COLUMN).alias(column_name))
    else:
        df = df.with_columns(pl.col(column).rolling_mean(days).alias(column_name))
    return IndicatorResult(df, column_name)

def slope_up(df: pl.DataFrame | pl.LazyFrame, column: str='Close') -> IndicatorResult:
    """indicates if a column's current value is more than it's previous value"""
    column_name = column + '_slope_up'

    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    df = df.with_columns((pl.col(column) > pl.col(column).shift()).alias(column_name))
    return IndicatorResult(df, column_name)


def rolling_min_with_offset(df: pl.DataFrame | pl.LazyFrame, column: str='Low', bars: int=250, offset: int=5) -> IndicatorResult:
    """calculates rolling min but skipping the last 'offset' number of bars"""
    column_name = f"{column}_{bars}min_{offset}offset"

    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    if SYMBOL_COLUMN in df.columns:
        df = df.with_columns(pl.col(column).rolling_min(bars-offset).shift(offset + 1).over(SYMBOL_COLUMN).alias(column_name))
    else:
        df = df.with_columns(pl.col(column).rolling_min(bars-offset).shift(offset + 1).alias(column_name))

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


def relative_volume(df: pl.DataFrame | pl.LazyFrame, bars: int, column=VOLUMNE_COLUMN) -> IndicatorResult:
    """calculates volume relative to average of last bars"""
    column_name = f"{bars}_relative_volume%"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    average_volume = simple_moving_average(df, bars, column)
    df = average_volume.df

    #need to shift average so current bar value is divsor of previous bars' average
    sma_column = "temp"
    if SYMBOL_COLUMN in df.columns:
        df =  df.with_columns(pl.col(column).shift().rolling_mean(bars).over(SYMBOL_COLUMN).alias(sma_column))
    else:
        df = df.with_columns(pl.col(column).shift().rolling_mean(bars).alias(column_name))
    
    df = df.with_columns((
            (pl.col(column) - pl.col(average_volume.column)) / pl.col(average_volume.column) * 100
        ).alias(column_name))
    
    df = df.select(pl.exclude(average_volume.column))

    return IndicatorResult(df, column_name)



def trailing_stop(df: pl.DataFrame | pl.LazyFrame, bars: int) -> IndicatorResult:
    """adds column of exit values indicating when trailing stop hit"""
    column_name = f"{bars}_bar_trailing_stop"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    df = df.with_columns(pl.when(
        pl.col(LOW_COLUMN) < pl.col(LOW_COLUMN).rolling_min(bars).shift(1)).then( #when our low is less than the trailing stop
            pl.min(pl.col(LOW_COLUMN).rolling_min(bars).shift(1), pl.col(OPEN_COLUMN))).alias(column_name)) #set the value equal to the minimum of the Open and the trailing stop. This handles cases where we gap below the trailing stop

    return IndicatorResult(df, column_name)


def end_of_data_stop(df: pl.DataFrame | pl.LazyFrame) -> IndicatorResult:
    """adds a stop at the close of the last bar of the data"""
    column_name = f"EOD_Stops"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    index_name = "index"
    if SYMBOL_COLUMN in df.columns:
        df = df.with_row_count(index_name).with_columns(pl.when(pl.col(index_name) == pl.col(index_name).max().over(SYMBOL_COLUMN)).then(pl.col(CLOSE_COLUMN)).alias(column_name)).select(pl.exclude(index_name))
    else:
        df = df.with_row_count(index_name).with_columns(pl.when(pl.col(index_name) == pl.col(index_name).max()).then(pl.col(CLOSE_COLUMN)).alias(column_name)).select(pl.exclude(index_name))

    return IndicatorResult(df, column_name)


def entry_percentage_stop(df: pl.DataFrame | pl.LazyFrame, percentage: float, entry_column: str) -> IndicatorResult:
    """Allows for percentage of entry stop
    will add column if value (100-percentage) / 100 * entry_column hit on the entry bar
    THIS CAN CREATE A STOP BASED ON AN ENTRY THAT NEVER HAPPENS
    POTENTIALLY COULD ADDRESS THIS USING add_exit_ids AND ONLY CHECKING
    THE STOP ON THE FIRST TRADE OF AN EXIT_ID"""
    column_name = f"Entry_{percentage}%_Stops"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    df = df.with_columns(pl.when(
        (
            pl.col(entry_column) * ((100+percentage)/100)).is_between(pl.col(LOW_COLUMN), pl.col(HIGH_COLUMN))
        ).then(
            pl.col(entry_column) * ((100+percentage)/100)
        ).alias(column_name)
        )
    
    return IndicatorResult(df, column_name)


def close_above_open(df: pl.DataFrame | pl.LazyFrame, close=CLOSE_COLUMN, open=OPEN_COLUMN) -> IndicatorResult:
    """adds bool column indicating if close was greater than open"""
    column_name = f"{close}_above{open}"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    df = df.with_columns((pl.col(close) > pl.col(open)).alias(column_name))
    return IndicatorResult(df, column_name)


def fluctuation_percentage(df: pl.DataFrame | pl.LazyFrame, column1=LOW_COLUMN, column2=HIGH_COLUMN) -> IndicatorResult:
    """adds a column for the percentage movement between columns
    this is calculated using the following formula
    (column2 - column1) / column1 * 100"""
    column_name = f"{column1}/{column2}_fluctuation%"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    df = df.with_columns(((pl.col(column2) - pl.col(column1)) / pl.col(column1) * 100).alias(column_name))
    return IndicatorResult(df, column_name)


def targeted_value(df: pl.DataFrame | pl.LazyFrame, targets: str) -> IndicatorResult:
    """Given a column with target values, adds column with those values if they were hit
    useful for mocking limit orders"""
    column_name = f"{targets}_targets"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    df = df.with_columns(pl.when(pl.col(targets).is_between(pl.col(LOW_COLUMN), pl.col(HIGH_COLUMN))).then(pl.col(targets)).alias(column_name))
    return IndicatorResult(df, column_name)


def limit_entries(df: pl.DataFrame | pl.LazyFrame, bars: int, entries: str) -> IndicatorResult:
    """Forces a minimum number of bars between entries"""
    def group_by_bars(df: pl.DataFrame | pl.LazyFrame, nulls: int, column: str) -> IndicatorResult:
        """assigns an ID to each group of consecutive non-null values allowing for an
        input (nulls) number of nulls before considering a set a new group"""
        column_name = f"group_{nulls}"
        if column_name in df.columns:
            return IndicatorResult(df, column_name)
        
        tes = "tes"
        df = df.with_columns(pl.lit(True).alias(tes))
        for i in range(1, nulls+1):
            df = df.with_columns(pl.col(tes) & pl.col(column).shift(i).is_null().alias(tes))
        df = df.with_columns(pl.when(pl.col(tes)).then(1).otherwise(0).cumsum().alias(column_name)) 
        return IndicatorResult(df, column_name)
    
    column_name = f"{bars}_minimum_bars_between"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    new_columns = df.columns.copy()
    new_columns.append(column_name)

    lf = df.lazy()
    
    group = group_by_bars(lf, bars, entries)

    ones = "ones"

    lf = group.df

    sum = "sum"
    lf = group.df.with_columns(pl.lit(1).alias(ones)).with_columns(pl.col(ones).cumsum().over(group.column).alias(sum))

    lf = lf.with_columns(pl.when(pl.col(sum) % (bars+1) == 1).then(pl.col(entries)).alias(column_name))

    lf = lf.select(new_columns)
    df = lf if isinstance(df, pl.LazyFrame) else lf.collect()

    return IndicatorResult(df, column_name)


def group_by_amount(df: pl.DataFrame | pl.LazyFrame, column: str, amounts: list[float], smallest_unit: float=0.01) -> IndicatorResult:
    """groups values into ranges based on input amounts
    the column this adds is a float column
    amounts must be sorted lowest to highest"""
    column_name = f"{column}_grouped_values"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    cols = []
    for i in range(len(amounts)):
        amount = amounts[i]
        cols.append(pl.when(pl.col(column) < amount).then(amount))
    cols.append(amounts[-1] + smallest_unit)
    df =  df.with_columns(pl.coalesce(cols).alias(column_name))

    return IndicatorResult(df, column_name)

def group_by_amount_display(df: pl.DataFrame | pl.LazyFrame, column: str, amounts: list[float], smallest_unit: float=0.01) -> IndicatorResult:
    """groups values into ranges based on input amounts
    the column this adds is formatted for display like 1-1.99, 2-2.99
    amounts must be sorted lowest to highest"""
    column_name = f"{column}_grouped_values_d"
    if column_name in df.columns:
        return IndicatorResult(df, column_name)
    
    cols = []
    first_amt = amounts[0]
    cols.append(pl.when(pl.col(column) < first_amt).then(pl.lit("<" + str(first_amt))))
    for i in range(1, len(amounts)):
        last_amount = amounts[i-1]
        amount = amounts[i]
        display_amount = str(amount-smallest_unit)
        cols.append(pl.when(pl.col(column) < amount).then(pl.lit(str(last_amount) + "-" + display_amount)))
    last_amt = amounts[-1]
    cols.append(pl.lit(">" + str(last_amt)))
    df =  df.with_columns(pl.coalesce(cols).alias(column_name))

    return IndicatorResult(df, column_name)


def create_trade_ids_old(df: pl.DataFrame | pl.LazyFrame, enter_column: str, exit_column: str) -> IndicatorResult:
    """increments count for each true value in input column
    intended to be used to create IDs for trades
    DEAD CODE SHOULD PROBABLY BE REMOVED"""
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
        df = df.with_columns(pl.when(pl.col(exit_column).is_not_null()).then(1).otherwise(0).cumsum().shift(1).alias(column_name))
        return IndicatorResult(df, column_name)
    def add_traded_column(df: pl.DataFrame | pl.LazyFrame, enter_column:str, exit_ids:str) -> IndicatorResult:
        """adds a column thats true for every bar that has an active trade"""
        column_name = f"{enter_column}/{exit_ids}_traded"
        if column_name in df.columns:
            return IndicatorResult(df, column_name)
        df = df.with_columns(
            pl.when(pl.col(DATE_COLUMN) >= pl.when(pl.col(enter_column).is_not_null()).then(pl.col(DATE_COLUMN)).min().over(exit_ids)).then(True) \
            .alias(column_name))
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
            (pl.col(DATE_COLUMN) == pl.col(DATE_COLUMN).min().over(exit_ids.column, traded.column)) #ensures we only count a trade once per exit
        ).then(1).otherwise(0).alias(column_name))
    
    lf = lf.with_columns(pl.when(pl.col(traded.column)).then(pl.col(column_name).cumsum()))
    lf = lf.select(new_columns)

    df = lf if isinstance(df, pl.LazyFrame) else lf.collect()

    return IndicatorResult(df, column_name)