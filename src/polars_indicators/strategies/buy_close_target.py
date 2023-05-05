import polars as pl
import polars_indicators as pi
from polars_indicators.strategies.strategy_result import StrategyResult


def strategy(df: pl.DataFrame | pl.LazyFrame, buy_column: str, trailing_stop_bars: int, entry_stop_percentage: int, target_percentage: float) -> StrategyResult:
    """Generates trades on input df

    This will filter out all data from before we have a full min from the lookback period
    see polars 'rolling_' documentation for all options

    Args:
        df: DataFrame instance with OHLCV columns
        buy_column: bool column in df that indicates when to buy
        trailing_stop_bars: number of bars trailing stop considers
        entry_stop_percentage: how much to offset the entry bar stop-loss value from
        target_percentage: how much to offset the target exit price

    Returns:
        A StategyResult representing the result of running the strategy
        on the input data
    """
    df = df.lazy()

    #calculate entries
    enter_column = "enter"
    df = df.with_columns(pl.when(pl.col(buy_column)).then(pl.col(pi.indicators.CLOSE_COLUMN)).alias(enter_column))

    #calculate exits
    eod = pi.indicators.end_of_data_stop()
    df = eod.df
    
    
    #clear out trailing stop values on entry columns since we're entering at close we can't hit it
    df = df.with_columns(pl.when(pl.col(enter_column).is_not_null()).then(pl.lit(None)).otherwise(pl.col(trail.column)).alias(trail.column))

    exit_column = "exit_column"
    df = df.with_columns(pl.coalesce(pl.col(trail.column), pl.col(eod.column)).alias(exit_column))



    ids = pi.indicators.create_trade_ids(df, enter_column, exit_column)

    df = ids.df.collect()
    
    return StrategyResult(df, enter_column, exit_column, ids.column)