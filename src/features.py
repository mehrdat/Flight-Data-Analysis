"""
features.py
-----------
Feature engineering = making new helpful columns from the existing ones.
These new columns make the analysis, charts and the ML models easier.

This file got bigger because the analysis now goes much deeper. Features are
grouped by topic so it is easy to read:
  - delay flags        (is the flight late? how late? early?)
  - time features      (hour, part of day, red-eye, weekend, season)
  - route / distance   (route string, distance bucket, long/short haul)
  - cause flags        (which delay cause was present)
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# DELAY FLAGS
# ---------------------------------------------------------------------------
def add_delay_flags(df: DataFrame) -> DataFrame:
    """
    Add simple yes/no (1/0) flags about lateness.
    In the industry a flight counts as 'delayed' if it arrives 15+ min late.
    """
    return (
        df
        .withColumn("is_delayed", (F.col("arr_delay") >= 15).cast("int"))
        .withColumn("is_early", (F.col("arr_delay") < 0).cast("int"))
        .withColumn("is_on_time", (F.col("arr_delay") < 15).cast("int"))
        # a "big" delay = more than one hour late, useful for severe-delay analysis
        .withColumn("is_big_delay", (F.col("arr_delay") >= 60).cast("int"))
        # did the crew make up time in the air? (left late but arrived less late)
        .withColumn("made_up_time", (F.col("dep_delay") - F.col("arr_delay")))
    )


def add_delay_severity(df: DataFrame) -> DataFrame:
    """
    Put each flight in a delay "band". This is nicer for stacked bar charts
    than the raw minutes, because everyone understands buckets.
    """
    sev = (
        F.when(F.col("arr_delay") < 0, "Early")
        .when(F.col("arr_delay") < 15, "On time")
        .when(F.col("arr_delay") < 60, "Minor (15-60m)")
        .when(F.col("arr_delay") < 180, "Major (1-3h)")
        .otherwise("Severe (3h+)")
    )
    return df.withColumn("delay_band", sev)


# ---------------------------------------------------------------------------
# TIME FEATURES
# ---------------------------------------------------------------------------
def add_time_buckets(df: DataFrame) -> DataFrame:
    """
    Turn the scheduled departure time (hhmm, e.g. 1845) into a clean hour 0-23,
    then make a friendly "part of the day" label and a red-eye flag.
    """
    df = df.withColumn("dep_hour", (F.col("crs_dep_time") / 100).cast("int"))

    part = (
        F.when(F.col("dep_hour") < 6, "Night (0-6)")
        .when(F.col("dep_hour") < 12, "Morning (6-12)")
        .when(F.col("dep_hour") < 17, "Afternoon (12-17)")
        .when(F.col("dep_hour") < 21, "Evening (17-21)")
        .otherwise("Late (21-24)")
    )
    df = df.withColumn("day_part", part)
    # red-eye = leaves very late at night / very early morning
    df = df.withColumn(
        "is_redeye",
        ((F.col("dep_hour") >= 22) | (F.col("dep_hour") <= 5)).cast("int"),
    )
    return df


def add_calendar_features(df: DataFrame) -> DataFrame:
    """Weekend flag + season label from the month."""
    df = df.withColumn(
        "is_weekend",
        F.col("day_of_week").isin(6, 7).cast("int"),  # 6=Sat, 7=Sun in this data
    )
    season = (
        F.when(F.col("month").isin(12, 1, 2), "Winter")
        .when(F.col("month").isin(3, 4, 5), "Spring")
        .when(F.col("month").isin(6, 7, 8), "Summer")
        .otherwise("Autumn")
    )
    df = df.withColumn("season", season)
    # name of the weekday, easier to read on charts
    dow_name = (
        F.when(F.col("day_of_week") == 1, "Mon")
        .when(F.col("day_of_week") == 2, "Tue")
        .when(F.col("day_of_week") == 3, "Wed")
        .when(F.col("day_of_week") == 4, "Thu")
        .when(F.col("day_of_week") == 5, "Fri")
        .when(F.col("day_of_week") == 6, "Sat")
        .otherwise("Sun")
    )
    df = df.withColumn("dow_name", dow_name)
    return df


# ---------------------------------------------------------------------------
# ROUTE / DISTANCE FEATURES
# ---------------------------------------------------------------------------
def add_route(df: DataFrame) -> DataFrame:
    """Make a single 'route' column like 'JFK-LAX' from origin and dest."""
    return df.withColumn("route", F.concat_ws("-", F.col("origin"), F.col("dest")))


def add_distance_buckets(df: DataFrame) -> DataFrame:
    """
    Group flights by trip length. Short hops and long hauls behave differently,
    so this lets us compare them.
    """
    bucket = (
        F.when(F.col("distance") < 250, "Very short (<250mi)")
        .when(F.col("distance") < 600, "Short (250-600)")
        .when(F.col("distance") < 1200, "Medium (600-1200)")
        .when(F.col("distance") < 2500, "Long (1200-2500)")
        .otherwise("Very long (2500+)")
    )
    df = df.withColumn("distance_band", bucket)
    df = df.withColumn("is_longhaul", (F.col("distance") >= 1200).cast("int"))
    return df


# ---------------------------------------------------------------------------
# DELAY CAUSE FLAGS
# ---------------------------------------------------------------------------
def add_cause_flags(df: DataFrame) -> DataFrame:
    """
    The five cause columns are minutes. Add a 1/0 flag for "was this cause
    present at all", which makes it easy to count how often each cause happens.
    """
    causes = [
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
    ]
    for c in causes:
        df = df.withColumn(f"has_{c}", (F.col(c) > 0).cast("int"))
    return df


# ---------------------------------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------------------------------
def add_features(df: DataFrame) -> DataFrame:
    """Run all feature steps together. The notebook calls just this one."""
    df = add_delay_flags(df)
    df = add_delay_severity(df)
    df = add_time_buckets(df)
    df = add_calendar_features(df)
    df = add_route(df)
    df = add_distance_buckets(df)
    df = add_cause_flags(df)
    return df
