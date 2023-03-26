import polars as pl

def simple_moving_average(df: pl.DataFrame | pl.LazyFrame, days: int, column: str='Close') -> pl.DataFrame | pl.LazyFrame:
    """returns dataframe with simple moving average added as a column
    if the column is already present, it will return the same df"""
    column_name = 'SMA' + str(days)
    if column_name in df.columns:
        return df
    return df.with_columns(pl.col('Close').rolling_mean(3).over('Symbol').alias(column_name))
