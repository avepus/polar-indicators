import polars as pl
import polars_indicators as pi
from polars_indicators.strategies.strategy_result import StrategyResult


def strategy(df: pl.DataFrame | pl.LazyFrame, lookback_bars: int, offset_bars: float, offset_percentage: int, entry_stop_percentage: int, trailing_stop_bars: int, min_price: float, additional_criteria: str=None) -> StrategyResult:
    """Generates trades on input df

    This will filter out all data from before we have a full min from the lookback period
    see polars 'rolling_' documentation for all options

    Args:
        df: DataFrame instance with OHLCV columns
        lookback_bars: how long the rolling min is
        offset_bars: how many bars to offset the min by
        offset_percentage: how much to offset the limit order from the min
        entry_stop_percentage: how much to offset the entry bar stop-loss value from
          the entry limit order price
        trailing_stop_bars: number of bars trailing stop considers
        min_price: minimum price required to attempt a trade on a bar
        additional_criteria: a bool series that must be true in order to attempt a trade

    Returns:
        A StategyResult representing the result of running the strategy
        on the input data
    """
    df = df.lazy()

    roll_min = pi.indicators.rolling_min_with_offset(df, column=pi.indicators.LOW_COLUMN, bars=lookback_bars, offset=offset_bars)

    df = roll_min.df

    #min_price filter
    df = df.with_columns(pl.when(pl.col(roll_min.column) >= min_price).then(pl.col(roll_min.column)))

    #filter when bar in offset range was below minimum
    df = df.with_columns(pl.when(pl.col(roll_min.column) < pl.col(pi.indicators.LOW_COLUMN).rolling_min(offset_bars).shift().over(pi.indicators.SYMBOL_COLUMN)).then(
        pl.col(roll_min.column)))

    #take the offset_percentage into account
    factor = (100 + offset_percentage) / 100
    df = df.with_columns(pl.col(roll_min.column) * factor)

    target = pi.indicators.targeted_value(df, roll_min.column)
    enter_column = target.column
    df = target.df

    #take additional criteria into account
    if additional_criteria:
        df = df.with_columns(pl.when(pl.col(additional_criteria)).then(pl.col(enter_column)))
    
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

    df = ids.df.collect()


    return StrategyResult(df, enter_column, exit_column, ids.column)


def fluctuation_and_relative_volume(df: pl.DataFrame | pl.LazyFrame, fluctuation_sma_length: int, relative_volume_sma_length: int, lookback_bars: int=7, offset_bars: int=3, offset_percentage: float=1, entry_stop_percentage: int=-5, trailing_stop_bars: int=2, min_price: float=3) -> StrategyResult:
    """uses fluctuation and relative volume to determine when to trade"""
    fluctuation = pi.indicators.fluctuation_percentage(df)
    fluct_sma3 = pi.indicators.simple_moving_average(fluctuation.df, fluctuation_sma_length, fluctuation.column)
    grouped_fluctation = pi.indicators.group_by_amount(fluct_sma3.df, fluct_sma3.column, [1,2,3,5,7,10,20])
    df = grouped_fluctation.df

    relative_volume = pi.indicators.relative_volume(df, relative_volume_sma_length)
    grouped_relative_volume = pi.indicators.group_by_amount(relative_volume.df, relative_volume.column, [x * 25 for x in range(1,7)])
    df = grouped_relative_volume.df

    additional_criteria = "additional_criteria"

    df = df.with_columns(
        (
            ((pl.col(grouped_fluctation.column) >= 10) & (pl.col(grouped_relative_volume.column) == 25))
            |
            ((pl.col(grouped_fluctation.column) == 20.01) & (pl.col(grouped_relative_volume.column) == 50))
        ).alias(additional_criteria))
    
    #return df

    return strategy(df, lookback_bars, offset_bars, offset_percentage, entry_stop_percentage, trailing_stop_bars, min_price, additional_criteria)
    
        