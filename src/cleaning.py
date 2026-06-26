"""
cleaning.py
-----------
Data cleaning steps for the flight data. This covers the things the
assignment asks about: missing values, duplicates, formatting and outliers.

Each function takes a DataFrame and returns a new (cleaned) DataFrame, so we
can chain them in the notebook and see the row count after every step.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def parse_dates(df: DataFrame) -> DataFrame:
    """Turn the 'fl_date' text column into a real date type."""
    return df.withColumn("fl_date", F.to_date("fl_date", "yyyy-MM-dd"))


def drop_duplicates(df: DataFrame) -> DataFrame:
    """
    Remove fully duplicated rows.
    (same flight, same date, same times = a duplicate record)
    """
    return df.dropDuplicates()


def drop_cancelled_and_diverted(df: DataFrame) -> DataFrame:
    """
    For delay analysis we only want flights that actually flew on time-ish.
    Cancelled or diverted flights have no real arrival delay, so we drop them.
    We keep them counted separately in analysis.py if needed.
    """
    return df.filter((F.col("cancelled") == 0) & (F.col("diverted") == 0))


def drop_missing_delays(df: DataFrame) -> DataFrame:
    """
    Drop rows where the delay columns are null. From the data dictionary this
    is only about 1.6% of rows, so it is safe to remove them.
    """
    return df.dropna(subset=["dep_delay", "arr_delay"])


def remove_delay_outliers(df: DataFrame, low: int = -120, high: int = 1440) -> DataFrame:
    """
    Remove crazy outlier delays that are most likely data errors.
    - low  = -120 min  : a flight leaving more than 2 hours early is not realistic
    - high = 1440 min  : 24 hours, anything above is almost surely bad data
    We apply this to both departure and arrival delay.
    """
    return df.filter(
        (F.col("dep_delay").between(low, high))
        & (F.col("arr_delay").between(low, high))
    )


def clean_flights(df: DataFrame) -> DataFrame:
    """
    Run the full cleaning pipeline in order.
    This is the one function the notebook will usually call.
    """
    df = parse_dates(df)
    df = drop_duplicates(df)
    df = drop_cancelled_and_diverted(df)
    df = drop_missing_delays(df)
    df = remove_delay_outliers(df)
    return df


def missing_value_report(df: DataFrame) -> DataFrame:
    """
    Helper for the 'Dataset Understanding' section of the report.
    Returns one row with the count of nulls in every column.
    """
    exprs = [
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]
    return df.select(exprs)
