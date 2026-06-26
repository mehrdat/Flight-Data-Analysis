"""
advanced_analysis.py
--------------------
The deeper / more statistical analysis that goes beyond simple group-by:

  - distributions & percentiles  (how spread out are the delays)
  - correlations                 (which numeric columns move together)
  - delay propagation            (does leaving late => arriving late?)
  - hub / network analysis       (busiest airports as a flight network)
  - two-way heatmaps             (hour x day, airline x month ...)
  - on-time performance ranking  (a single score per airline)

Each function returns a small aggregated Spark or pandas object that the
notebook then plots.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# DISTRIBUTIONS & PERCENTILES
# ---------------------------------------------------------------------------
def delay_describe(df: DataFrame) -> DataFrame:
    """
    Summary statistics for arrival delay: count, mean, stddev, min, max.
    Quick way to show the shape of the data in the report.
    """
    return df.select(
        F.count("arr_delay").alias("count"),
        F.round(F.avg("arr_delay"), 2).alias("mean"),
        F.round(F.stddev("arr_delay"), 2).alias("stddev"),
        F.min("arr_delay").alias("min"),
        F.max("arr_delay").alias("max"),
    )


def delay_percentiles(df: DataFrame, col: str = "arr_delay") -> DataFrame:
    """
    Percentiles (P10, P25, median, P75, P90, P95, P99) of a delay column.
    Percentiles tell the real story better than the mean because delays are
    very skewed (a few huge delays drag the average up).
    """
    probs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    qs = df.approxQuantile(col, probs, 0.001)
    labels = ["P10", "P25", "P50", "P75", "P90", "P95", "P99"]
    spark = df.sparkSession
    return spark.createDataFrame(
        [(lab, float(v)) for lab, v in zip(labels, qs)],
        ["percentile", "minutes"],
    )


def delay_histogram(df: DataFrame, col: str = "arr_delay",
                    low: int = -60, high: int = 240, step: int = 15):
    """
    Build a histogram of delays as a pandas dataframe (bin, count).
    We bucket the minutes into fixed-width bins so matplotlib can draw bars.
    """
    binned = df.withColumn(
        "bin",
        (F.floor(F.col(col) / step) * step).cast("int"),
    ).filter(F.col("bin").between(low, high))
    out = (
        binned.groupBy("bin")
        .agg(F.count("*").alias("flights"))
        .orderBy("bin")
    )
    return out


def percentiles_by_airline(df: DataFrame) -> DataFrame:
    """
    Median and P90 arrival delay per airline. Using the SQL percentile function
    so it works in one group-by. P90 = the 'bad day' a typical flyer might see.
    """
    return (
        df.groupBy("op_unique_carrier")
        .agg(
            F.count("*").alias("flights"),
            F.round(F.expr("percentile_approx(arr_delay, 0.5)"), 1).alias("median_delay"),
            F.round(F.expr("percentile_approx(arr_delay, 0.9)"), 1).alias("p90_delay"),
        )
        .filter(F.col("flights") >= 1000)
        .orderBy("p90_delay")
    )


# ---------------------------------------------------------------------------
# CORRELATIONS
# ---------------------------------------------------------------------------
def correlation_matrix(df: DataFrame, cols: list[str] | None = None):
    """
    Pearson correlation between the main numeric columns.
    Returns a pandas dataframe (a square matrix) ready for a heatmap.
    """
    if cols is None:
        cols = [
            "dep_delay", "arr_delay", "taxi_out", "taxi_in",
            "distance", "air_time", "crs_elapsed_time", "dep_hour",
        ]
    # keep only columns that actually exist
    cols = [c for c in cols if c in df.columns]
    pdf = df.select(cols).dropna().toPandas()
    return pdf.corr(method="pearson")


# ---------------------------------------------------------------------------
# DELAY PROPAGATION  (departure delay -> arrival delay)
# ---------------------------------------------------------------------------
def propagation_scatter_sample(df: DataFrame, n: int = 5000):
    """
    Take a small random sample of (dep_delay, arr_delay) for a scatter plot.
    We sample because plotting millions of points is pointless and slow.
    """
    frac = min(1.0, n / max(df.count(), 1))
    return (
        df.select("dep_delay", "arr_delay")
        .sample(withReplacement=False, fraction=frac, seed=42)
        .limit(n)
        .toPandas()
    )


def propagation_by_depdelay_band(df: DataFrame) -> DataFrame:
    """
    Group flights by how late they DEPARTED, then show the average ARRIVAL
    delay for each band. Shows how much of a late departure is recovered.
    """
    band = (
        F.when(F.col("dep_delay") <= 0, "On time / early")
        .when(F.col("dep_delay") < 15, "1-15 late")
        .when(F.col("dep_delay") < 30, "15-30 late")
        .when(F.col("dep_delay") < 60, "30-60 late")
        .when(F.col("dep_delay") < 120, "60-120 late")
        .otherwise("120+ late")
    )
    return (
        df.withColumn("dep_band", band)
        .groupBy("dep_band")
        .agg(
            F.count("*").alias("flights"),
            F.round(F.avg("dep_delay"), 1).alias("avg_dep_delay"),
            F.round(F.avg("arr_delay"), 1).alias("avg_arr_delay"),
            F.round(F.avg("made_up_time"), 1).alias("avg_minutes_recovered"),
        )
    )


# ---------------------------------------------------------------------------
# TWO-WAY HEATMAPS
# ---------------------------------------------------------------------------
def heatmap_hour_vs_dow(df: DataFrame):
    """
    Average delay for every (hour, day-of-week) combination.
    Returned as a pivoted pandas table -> perfect for a heatmap.
    """
    grouped = (
        df.groupBy("dep_hour", "day_of_week")
        .agg(F.round(F.avg("arr_delay"), 2).alias("avg_delay"))
        .toPandas()
    )
    pivot = grouped.pivot(index="dep_hour", columns="day_of_week", values="avg_delay")
    return pivot.sort_index()


def heatmap_airline_vs_month(df: DataFrame):
    """Average delay for every (airline, month) - another heatmap."""
    grouped = (
        df.groupBy("op_unique_carrier", "month")
        .agg(F.round(F.avg("arr_delay"), 2).alias("avg_delay"))
        .toPandas()
    )
    pivot = grouped.pivot(index="op_unique_carrier", columns="month", values="avg_delay")
    return pivot


# ---------------------------------------------------------------------------
# HUB / NETWORK ANALYSIS
# ---------------------------------------------------------------------------
def airport_traffic(df: DataFrame, top: int = 20) -> DataFrame:
    """
    Total movements per airport = departures + arrivals.
    This is the 'how big a hub is it' number used for the map bubble size.
    """
    deps = df.groupBy("origin").agg(F.count("*").alias("departures")) \
             .withColumnRenamed("origin", "airport")
    arrs = df.groupBy("dest").agg(F.count("*").alias("arrivals")) \
             .withColumnRenamed("dest", "airport")
    joined = (
        deps.join(arrs, "airport", "outer")
        .fillna(0, ["departures", "arrivals"])
        .withColumn("total_movements", F.col("departures") + F.col("arrivals"))
        .orderBy(F.desc("total_movements"))
    )
    return joined.limit(top) if top else joined


def busiest_routes_volume(df: DataFrame, top: int = 25) -> DataFrame:
    """Top routes by number of flights - feeds the route map lines."""
    return (
        df.groupBy("origin", "dest", "route")
        .agg(
            F.count("*").alias("flights"),
            F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
            F.first("distance").alias("distance"),
        )
        .orderBy(F.desc("flights"))
        .limit(top)
    )


# ---------------------------------------------------------------------------
# ON-TIME PERFORMANCE RANKING (a single score)
# ---------------------------------------------------------------------------
def airline_scorecard(df: DataFrame, min_flights: int = 1000) -> DataFrame:
    """
    One tidy 'scorecard' row per airline with the headline numbers, ranked by
    on-time %. Good for a final summary table in the report.
    """
    return (
        df.groupBy("op_unique_carrier")
        .agg(
            F.count("*").alias("flights"),
            F.round(100 * F.avg("is_on_time"), 1).alias("on_time_pct"),
            F.round(F.avg("arr_delay"), 1).alias("avg_arr_delay"),
            F.round(F.expr("percentile_approx(arr_delay, 0.9)"), 1).alias("p90_delay"),
            F.round(100 * F.avg("is_big_delay"), 2).alias("big_delay_pct"),
        )
        .filter(F.col("flights") >= min_flights)
        .orderBy(F.desc("on_time_pct"))
    )
