from dataclasses import dataclass
from polars import DataFrame, LazyFrame


@dataclass
class StrategyResult:
    """Holds the dataframe with the enter and exit column and the name of those columns
    Every indicator returns this."""
    df: DataFrame | LazyFrame
    entry_column: str
    exit_column: str
    ids_column: str