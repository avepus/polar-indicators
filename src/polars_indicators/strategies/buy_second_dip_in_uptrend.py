from datetime import timedelta
import polars as pl
import polars_indicators as pi
from polars_indicators.strategies.strategy_result import StrategyResult


def strategy(df: pl.DataFrame | pl.LazyFrame, lookback_bars: int, offset_bars: int, offset_percentage: int, trailing_stop_bars: int, min_price: float) -> StrategyResult:
    """Generates trades on input df

    This will filter out all data from before we have a full min from the lookback period
    see polars 'rolling_' documentation for all options

    Args:
        df: DataFrame instance with OHLCV columns
        lookback_bars: how long the rolling min is
        offset_bars: how many bars to offset the min by and how many bars to look for the second dip
        offset_percentage: how far off the min the second dip can be to still be considered a dip
        trailing_stop_bars: number of bars trailing stop considers
        min_price: minimum price required to attempt a trade on a bar

    Returns:
        A StategyResult representing the result of running the strategy
        on the input data
    """
    df = df.lazy()

    
    roll_min = pi.indicators.rolling_min_with_offset(df, column=pi.indicators.LOW_COLUMN, bars=lookback_bars, offset=offset_bars)

    df = roll_min.df

    #min_price filter
    df = df.with_columns(pl.when(pl.col(roll_min.column) >= min_price).then(pl.col(roll_min.column)))

    #when last low was in the threshold range but not actually below low
    factor = (100 + offset_percentage) / 100
    near_low = "near_low"
    df = df.with_columns(
                        (
                            (pl.col(pi.indicators.LOW_COLUMN) < (pl.col(roll_min.column) * factor))
                            &
                            (pl.col(pi.indicators.LOW_COLUMN) > pl.col(roll_min.column))
                        ).alias(near_low)
                    )

    close_above_open = pi.indicators.close_above_open(df)
    df = close_above_open.df

    close_targets = "close_targets"
    df = df.with_columns(pl.when(pl.col(close_above_open.column) & pl.col(near_low)).then(pl.col(pi.indicators.CLOSE_COLUMN).shift()).alias(close_targets))

    target = pi.indicators.targeted_value(df, close_targets)
    enter_column = target.column
    df = target.df

    trail = pi.indicators.trailing_stop(df, bars=trailing_stop_bars)

    eod = pi.indicators.end_of_data_stop(trail.df)
    df = eod.df

    exit_column = "exit_column"
    df = df.with_columns(pl.coalesce(pl.col(trail.column), pl.col(eod.column)).alias(exit_column))

    ids = pi.indicators.create_trade_ids(df, enter_column, exit_column)

    df = ids.df.collect()

    return StrategyResult(df, enter_column, exit_column, ids.column)