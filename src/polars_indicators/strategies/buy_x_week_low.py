from datetime import timedelta
import polars as pl
import polars_indicators as pi
from polars_indicators.strategies.strategy_result import StrategyResult


def strategy(df: pl.DataFrame | pl.LazyFrame, lookback: timedelta, offset_percentage: int, entry_stop_percentage: int, trailing_stop_bars: int) -> StrategyResult:
    """Generates trades on input df

    This will filter out all data from before we have a full min from the lookback period
    see polars 'rolling_' documentation for all options

    Args:
        df: DataFrame instance with OHLCV columns
        lookback: how long the rolling min is
        offset_percentage: how much to offset the limit order from the min
        entry_stop_percentage: how much to offset the entry bar stop-loss value from
          the entry limit order price
        trailing_stop_bars: number of bars trailing stop considers

    Returns:
        A StategyResult representing the result of running the strategy
        on the input data
    """
    factor = (100 + offset_percentage) / 100

    weeks_min = f"{lookback}_week_min"
    df = df.with_columns((pl.col("Low") * factor).rolling_min(
        lookback, by=pi.indicators.DATE_COLUMN).over(
            pi.indicators.SYMBOL_COLUMN).alias(weeks_min))
    
    filter_datetime = df[pi.indicators.DATE_COLUMN].min() + lookback
    df = df.filter(pl.col(pi.indicators.DATE_COLUMN) > filter_datetime) #this filters out data that doesn't have the full lookback

    target = pi.indicators.targeted_value(df, weeks_min)
    enter_column = target.column
    df = target.df
    
    percentage_stop = pi.indicators.entry_percentage_stop(df, entry_stop_percentage, target.column)
    df = percentage_stop.df

    trail = pi.indicators.trailing_stop(df, bars=trailing_stop_bars)

    eod = pi.indicators.end_of_data_stop(trail.df)
    df = eod.df

    #clear out trailing stop values on entry columns since that's handled by percentage stop
    df = df.with_columns(pl.when(pl.col(enter_column).is_not_null()).then(pl.lit(None)).otherwise(pl.col(trail.column)).alias(trail.column))

    exit_column = "exit_column"
    df = df.with_columns(pl.coalesce(pl.col(percentage_stop.column),pl.col(trail.column), pl.col(eod.column)).alias(exit_column))

    ids = pi.indicators.create_trade_ids(df, enter_column, exit_column)

    df = ids.df

    return StrategyResult(df, enter_column, exit_column, ids.column )
    
        