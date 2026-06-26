"""
clustering.py
-------------
Unsupervised learning: group airports into "performance segments" with K-Means.

Idea: build one row per airport that describes how it behaves (how busy it is,
average delay, % delayed, % cancelled, average taxi-out time). Then K-Means
finds natural groups, e.g. "big busy hubs with high delays" vs "small quiet
airports that run on time". This is useful for the report because it turns
thousands of airports into a few easy-to-explain types.

We also compute the silhouette score for a few values of k so we can justify
the number of clusters we picked (instead of guessing).
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# the airport "profile" columns we cluster on
PROFILE_COLS = ["flights", "avg_arr_delay", "delayed_pct", "avg_taxi_out", "pct_longhaul"]


def build_airport_profiles(df: DataFrame, min_flights: int = 1000) -> DataFrame:
    """
    One row per ORIGIN airport summarising its behaviour.
    Only keep airports with enough flights so the numbers are stable.
    """
    profiles = (
        df.groupBy("origin")
        .agg(
            F.count("*").alias("flights"),
            F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
            F.round(100 * F.avg("is_delayed"), 2).alias("delayed_pct"),
            F.round(F.avg("taxi_out"), 2).alias("avg_taxi_out"),
            F.round(100 * F.avg("is_longhaul"), 2).alias("pct_longhaul"),
        )
        .filter(F.col("flights") >= min_flights)
        .dropna()
    )
    return profiles


def _make_scaled_features(profiles: DataFrame):
    """Assemble the profile columns into one vector and standardise them."""
    assembler = VectorAssembler(inputCols=PROFILE_COLS, outputCol="raw_features")
    assembled = assembler.transform(profiles)
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features",
        withMean=True, withStd=True,
    )
    scaler_model = scaler.fit(assembled)
    return scaler_model.transform(assembled)


def choose_k(profiles: DataFrame, k_values=(2, 3, 4, 5, 6)) -> DataFrame:
    """
    Try several k and report the silhouette score for each.
    Higher silhouette (closer to 1) = better separated clusters.
    Returns a pandas dataframe (k, silhouette).
    """
    scaled = _make_scaled_features(profiles).cache()
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    import pandas as pd

    rows = []
    for k in k_values:
        km = KMeans(featuresCol="features", k=k, seed=42)
        model = km.fit(scaled)
        score = evaluator.evaluate(model.transform(scaled))
        rows.append({"k": k, "silhouette": round(score, 4)})
    scaled.unpersist()
    return pd.DataFrame(rows)


def cluster_airports(profiles: DataFrame, k: int = 4) -> DataFrame:
    """
    Fit K-Means with the chosen k and attach a 'cluster' label to each airport.
    Returns a pandas dataframe with the profile + cluster (small, ready to plot
    or map).
    """
    scaled = _make_scaled_features(profiles)
    km = KMeans(featuresCol="features", k=k, seed=42)
    model = km.fit(scaled)
    labelled = model.transform(scaled).select(
        "origin", *PROFILE_COLS, "prediction"
    ).withColumnRenamed("prediction", "cluster")
    return labelled.toPandas()


def cluster_profiles_summary(clustered_pdf):
    """
    Average profile of each cluster (pandas in, pandas out).
    This is the table that explains what each cluster 'means'.
    """
    return (
        clustered_pdf.groupby("cluster")[PROFILE_COLS]
        .mean()
        .round(2)
        .assign(n_airports=clustered_pdf.groupby("cluster").size())
        .reset_index()
    )
