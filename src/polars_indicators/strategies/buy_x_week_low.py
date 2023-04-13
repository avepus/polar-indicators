from datetime import timedelta
import polars as pl
import polars_indicators as pi
from polars_indicators import IndicatorResult

def strategy(df: pl.DataFrame | pl.LazyFrame, lookback: timedelta) -> IndicatorResult:
    """generates trades on input df
    this will filter out all data from before we have a full min from the lookback period
    lookback is a string like 52w for 52 weeks
    see polars 'rolling_' documentation for all options"""

    weeks_min = f"{lookback}_week_min"
    df = df.with_columns(pl.col("Low").rolling_min(lookback, by=pi.DATE_COLUMN).over(pi.SYMBOL_COLUMN).alias(weeks_min))
    filter_datetime = df[pi.DATE_COLUMN].min() + lookback
    df = df.filter(pl.col(pi.DATE_COLUMN) > filter_datetime) #this filters out data that doesn't have the full lookback

    target = pi.targeted_value(df, weeks_min)
    enter_column = target.column
    df = target.df
    
    percentage = -5
    percentage_stop = pi.entry_percentage_stop(df, percentage, target.column)
    df = percentage_stop.df

    trail = pi.trailing_stop(df, bars=2) #magic number should be parameter

    eod = pi.end_of_data_stop(trail.df)
    df = eod.df

    #clear out trailing stop values on entry columns since that's handled by percentage stop
    df = df.with_columns(pl.when(pl.col(enter_column).is_not_null()).then(pl.lit(None)).otherwise(pl.col(trail.column)).alias(trail.column))

    exit_column = "exit_column"
    df = df.with_columns(pl.coalesce(pl.col(percentage_stop.column),pl.col(trail.column), pl.col(eod.column)).alias(exit_column))

    return pi.create_trade_ids(df, enter_column, exit_column)
    
        