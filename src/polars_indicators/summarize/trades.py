

import polars as pl
import polars_indicators as pi
from polars_indicators.strategies.strategy_result import StrategyResult

class Trades:
    """represents a summary of trades
    main purpose is to further summarize the trades"""

    @classmethod
    def from_strategy_result(cls, strategy_result: StrategyResult, additional_columns=[]):
        """constructs instance of class from a StrategyResult"""
        return cls(strategy_result.df, strategy_result.ids_column, strategy_result.entry_column, strategy_result.exit_column, additional_columns)


    def __init__(self, df: pl.DataFrame | pl.LazyFrame, trade_id_column: str, enter_column: str, exit_column: str, additional_columns: list[str] = []) -> pl.DataFrame | pl.LazyFrame:
        """summarizes trade information given ids in input column
        allows for keeping additional column values as they were at the start of the trade
        PROTOTYPE. NEEDS MORE WORK AND MAY NOT BE THE DIRECTION I GO"""
        index_name = "index"
        start = "Start"
        end = "End"
        entry_price = "Entry_Price"
        exit_price = "Exit_Price"
        self.length = "Length"
        df = df.drop_nulls(trade_id_column).with_row_count(index_name).groupby(trade_id_column).agg(
                [pl.when(pl.col(pi.indicators.DATE_COLUMN) == pl.col(pi.indicators.DATE_COLUMN).max()).then(pl.col(col)).min().alias(col) for col in additional_columns],
                pl.col(pi.indicators.SYMBOL_COLUMN).min().alias(pi.indicators.SYMBOL_COLUMN),
                pl.col(pi.indicators.DATE_COLUMN).min().alias(start),
                pl.col(pi.indicators.DATE_COLUMN).max().alias(end),
                pl.when(pl.col(pi.indicators.DATE_COLUMN) == pl.col(pi.indicators.DATE_COLUMN).min()).then(pl.col(enter_column)).min().alias(entry_price),
                pl.when(pl.col(pi.indicators.DATE_COLUMN) == pl.col(pi.indicators.DATE_COLUMN).max()).then(pl.col(exit_column)).min().alias(exit_price),
                pl.col(pi.indicators.LOW_COLUMN).min().alias(pi.indicators.LOW_COLUMN),
                pl.col(pi.indicators.HIGH_COLUMN).max().alias(pi.indicators.HIGH_COLUMN),
                (pl.col(index_name).max() - pl.col(index_name).min() + 1).alias(self.length)
        ).sort(trade_id_column)

        net_gain = "Gain/Loss"
        self.percent_change = "Gain/Loss%"
        self.highest_percent = "Highest%"
        self.lowest_percent = "Lowest%"
        df = df.with_columns(
            (pl.col(exit_price) - pl.col(entry_price)).alias(net_gain),
            ((pl.col(exit_price) - pl.col(entry_price)) / pl.col(entry_price) * 100).round(2).alias(self.percent_change),
            ((pl.col(pi.indicators.HIGH_COLUMN) - pl.col(entry_price)) / pl.col(entry_price) * 100).round(2).alias(self.highest_percent),
            ((pl.col(pi.indicators.   LOW_COLUMN) - pl.col(entry_price)) / pl.col(entry_price) * 100).round(2).alias(self.lowest_percent),
            )

        self.df = df
        self.trade_id_column = trade_id_column
        self.enter_column = enter_column
        self.exit_column = exit_column



    def summarize_strategy(self, group_column: str = None) -> pl.DataFrame | pl.LazyFrame:
        """Summarizes strategy by key metrics"""
        columns = [
            pl.col(self.percent_change).sum().alias("Gain/Loss%Sum"),
            (pl.col(self.percent_change) > 0).sum().alias("Winners"),
            (pl.col(self.percent_change) < 0).sum().alias("Losers"),
            pl.col(pi.indicators.SYMBOL_COLUMN).count().alias("Trades"),
            ((pl.col(self.percent_change) > 0).sum() / (pl.col(pi.indicators.SYMBOL_COLUMN).count())).alias("Win_Ratio"),
            pl.col(self.percent_change).mean().alias("Average_Gain/Loss%"),
            pl.col(self.length).mean().alias("Average_Length"),
            (pl.when(pl.col(self.percent_change) > 0).then(pl.col(self.percent_change))).max().alias("highest_win"),
            pl.col(self.highest_percent).max(),
            (pl.when(pl.col(self.percent_change) < 0).then(pl.col(self.percent_change))).min().alias("lowest_loss"),
            pl.col(self.lowest_percent).min()
            ]
        if group_column:
            return self.df.groupby(group_column).agg(
                columns
                )
        
        return self.df.select(columns
            
            )


def zzsummarize_trades(df: pl.DataFrame | pl.LazyFrame, trade_id_column: str, enter_column: str, exit_column: str, additional_columns: list[str]) -> pl.DataFrame | pl.LazyFrame:
    """summarizes trade information given ids in input column
    allows for keeping additional column values as they were at the start of the trade
    PROTOTYPE. NEEDS MORE WORK AND MAY NOT BE THE DIRECTION I GO"""
    index_name = "index"
    start = "Start"
    end = "End"
    entry_price = "Entry_Price"
    exit_price = "Exit_Price"
    length = "Length"
    df = df.drop_nulls(trade_id_column).with_row_count(index_name).groupby(trade_id_column).agg(
              [pl.when(pl.col(pi.indicators.DATE_COLUMN) == pl.col(pi.indicators.DATE_COLUMN).max()).then(pl.col(col)).min().alias(col) for col in additional_columns],
              pl.col(pi.indicators.SYMBOL_COLUMN).min().alias(pi.indicators.SYMBOL_COLUMN),
              pl.col(pi.indicators.DATE_COLUMN).min().alias(start),
              pl.col(pi.indicators.DATE_COLUMN).max().alias(end),
              pl.when(pl.col(pi.indicators.DATE_COLUMN) == pl.col(pi.indicators.DATE_COLUMN).min()).then(pl.col(enter_column)).min().alias(entry_price),
              pl.when(pl.col(pi.indicators.DATE_COLUMN) == pl.col(pi.indicators.DATE_COLUMN).max()).then(pl.col(exit_column)).min().alias(exit_price),
              pl.col(pi.indicators.LOW_COLUMN).min().alias(pi.indicators.LOW_COLUMN),
              pl.col(pi.indicators.HIGH_COLUMN).max().alias(pi.indicators.HIGH_COLUMN),
              (pl.col(index_name).max() - pl.col(index_name).min() + 1).alias(length)
    ).sort(trade_id_column)

    net_gain = "Gain/Loss"
    net_gain_percent = "Gain/Loss%"
    highest_percent = "Highest%"
    lowest_percent = "Lowest%"
    df = df.with_columns(
        (pl.col(exit_price) - pl.col(entry_price)).alias(net_gain),
        ((pl.col(exit_price) - pl.col(entry_price)) / pl.col(entry_price) * 100).round(2).alias(net_gain_percent),
        ((pl.col(HIGH_COLUMN) - pl.col(entry_price)) / pl.col(entry_price) * 100).round(2).alias(highest_percent),
        ((pl.col(LOW_COLUMN) - pl.col(entry_price)) / pl.col(entry_price) * 100).round(2).alias(lowest_percent),
        )

    return df