"""
analysis.py
-----------
The descriptive analytics: group the data and calculate delay statistics.
Each function returns a small Spark DataFrame (already aggregated) that we
can turn into a pandas DataFrame and plot in the notebook.

The assignment asks to analyse delays by airport, airline, route, month,
day and time period - and we now go quite a bit further than that with extra
"scenario" breakdowns (season, day-part, distance band, weekend vs weekday,
delay bands, cause frequency, etc).

The deeper / statistical stuff (correlations, propagation, percentiles,
network analysis) lives in advanced_analysis.py to keep this file readable.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# SHARED HELPER
# ---------------------------------------------------------------------------
def _delay_summary(df: DataFrame, group_col: str) -> DataFrame:
    """
    Group by one column and work out the usual numbers.
    - flights        = how many flights
    - avg_arr_delay  = average arrival delay (minutes)
    - avg_dep_delay  = average departure delay (minutes)
    - delayed_pct    = % of flights that were 15+ min late
    - big_delay_pct  = % that were 60+ min late
    - early_pct      = % that arrived early
    """
    return (
        df.groupBy(group_col)
        .agg(
            F.count("*").alias("flights"),
            F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
            F.round(F.avg("dep_delay"), 2).alias("avg_dep_delay"),
            F.round(100 * F.avg("is_delayed"), 2).alias("delayed_pct"),
            F.round(100 * F.avg("is_big_delay"), 2).alias("big_delay_pct"),
            F.round(100 * F.avg("is_early"), 2).alias("early_pct"),
        )
        .orderBy(F.desc("flights"))
    )


# ---------------------------------------------------------------------------
# THE ORIGINAL CORE BREAKDOWNS (kept, just richer now)
# ---------------------------------------------------------------------------
def delays_by_airline(df: DataFrame) -> DataFrame:
    return _delay_summary(df, "op_unique_carrier")


def delays_by_origin_airport(df: DataFrame, min_flights: int = 0) -> DataFrame:
    out = _delay_summary(df, "origin")
    return out.filter(F.col("flights") >= min_flights) if min_flights else out


def delays_by_dest_airport(df: DataFrame, min_flights: int = 0) -> DataFrame:
    out = _delay_summary(df, "dest")
    return out.filter(F.col("flights") >= min_flights) if min_flights else out


def delays_by_route(df: DataFrame, min_flights: int = 500) -> DataFrame:
    """Busy routes only, otherwise tiny routes give noisy averages."""
    out = _delay_summary(df, "route")
    return out.filter(F.col("flights") >= min_flights)


def delays_by_month(df: DataFrame) -> DataFrame:
    return _delay_summary(df, "month").orderBy("month")


def delays_by_day_of_week(df: DataFrame) -> DataFrame:
    return _delay_summary(df, "day_of_week").orderBy("day_of_week")


def delays_by_hour(df: DataFrame) -> DataFrame:
    return _delay_summary(df, "dep_hour").orderBy("dep_hour")


# ---------------------------------------------------------------------------
# NEW SCENARIO BREAKDOWNS
# ---------------------------------------------------------------------------
def delays_by_season(df: DataFrame) -> DataFrame:
    return _delay_summary(df, "season")


def delays_by_day_part(df: DataFrame) -> DataFrame:
    return _delay_summary(df, "day_part")


def delays_by_distance_band(df: DataFrame) -> DataFrame:
    return _delay_summary(df, "distance_band")


def delays_weekend_vs_weekday(df: DataFrame) -> DataFrame:
    out = _delay_summary(df, "is_weekend")
    # make the label readable
    return out.withColumn(
        "kind", F.when(F.col("is_weekend") == 1, "Weekend").otherwise("Weekday")
    )


def delays_by_state(df: DataFrame) -> DataFrame:
    """Average delay grouped by the ORIGIN state - feeds the choropleth map."""
    return _delay_summary(df, "origin_state_nm")


def delay_band_counts(df: DataFrame) -> DataFrame:
    """How many flights fall in each delay band (Early / On time / Minor...)."""
    return (
        df.groupBy("delay_band")
        .agg(F.count("*").alias("flights"))
        .orderBy(F.desc("flights"))
    )


def delays_by_redeye(df: DataFrame) -> DataFrame:
    out = _delay_summary(df, "is_redeye")
    return out.withColumn(
        "kind", F.when(F.col("is_redeye") == 1, "Red-eye").otherwise("Daytime")
    )


# ---------------------------------------------------------------------------
# DELAY CAUSES
# ---------------------------------------------------------------------------
def delay_cause_totals(df: DataFrame) -> DataFrame:
    """
    Sum up the five official delay-cause columns to see which cause is biggest.
    Returns one row with a total (in minutes) for each cause.
    """
    causes = [
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
    ]
    return df.select([F.sum(c).alias(c) for c in causes])


def delay_cause_frequency(df: DataFrame) -> DataFrame:
    """
    How OFTEN each cause shows up (not minutes, but % of flights where the cause
    was present). Needs the has_* flags from features.add_cause_flags.
    """
    causes = [
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
    ]
    exprs = [F.round(100 * F.avg(f"has_{c}"), 2).alias(c) for c in causes]
    return df.select(exprs)


def cause_share_by_month(df: DataFrame) -> DataFrame:
    """
    Total minutes of each cause per month - good for a stacked area/bar chart
    showing how the delay 'mix' changes through the year (e.g. weather in winter).
    """
    causes = [
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
    ]
    return (
        df.groupBy("month")
        .agg(*[F.sum(c).alias(c) for c in causes])
        .orderBy("month")
    )


# ---------------------------------------------------------------------------
# CANCELLATIONS & DIVERSIONS
# (these need the RAW data, before we drop cancelled/diverted rows)
# ---------------------------------------------------------------------------
def cancellation_overview(df_raw: DataFrame) -> DataFrame:
    """Overall cancelled / diverted rates. Pass the RAW (uncleaned) dataframe."""
    return df_raw.select(
        F.count("*").alias("total_flights"),
        F.round(100 * F.avg("cancelled"), 3).alias("cancelled_pct"),
        F.round(100 * F.avg("diverted"), 3).alias("diverted_pct"),
    )


def cancellations_by_reason(df_raw: DataFrame) -> DataFrame:
    """
    Counts per cancellation code.
    Codes: A=Carrier, B=Weather, C=National Air System, D=Security.
    """
    return (
        df_raw.filter(F.col("cancelled") == 1)
        .groupBy("cancellation_code")
        .agg(F.count("*").alias("cancellations"))
        .orderBy(F.desc("cancellations"))
    )


def cancellations_by_airline(df_raw: DataFrame) -> DataFrame:
    return (
        df_raw.groupBy("op_unique_carrier")
        .agg(
            F.count("*").alias("flights"),
            F.round(100 * F.avg("cancelled"), 3).alias("cancelled_pct"),
            F.round(100 * F.avg("diverted"), 3).alias("diverted_pct"),
        )
        .orderBy(F.desc("cancelled_pct"))
    )


def cancellations_by_month(df_raw: DataFrame) -> DataFrame:
    return (
        df_raw.groupBy("month")
        .agg(F.round(100 * F.avg("cancelled"), 3).alias("cancelled_pct"))
        .orderBy("month")
    )
