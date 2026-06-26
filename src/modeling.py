"""
modeling.py
-----------
Predictive modelling with Spark MLlib (the machine-learning part).

Two jobs:
  1. CLASSIFICATION - will a flight be delayed (15+ min) or not? (yes/no)
     We try Logistic Regression and Random Forest and compare them.
  2. REGRESSION     - how many minutes late will the arrival be? (a number)
     We use Random Forest Regressor.

Important: to avoid 'cheating', the model must NOT use columns that are only
known AFTER the flight (like dep_delay, taxi_out, arr_time). We only use things
known BEFORE departure: airline, origin, dest, month, day, hour, distance.

Everything is built with a Spark ML Pipeline so the same steps run on train
and test data the same way.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)


# columns the model is allowed to use (all known before the flight leaves)
CAT_COLS = ["op_unique_carrier", "origin", "dest", "season", "day_part"]
NUM_COLS = ["month", "day_of_week", "dep_hour", "distance", "is_weekend", "is_redeye"]


def _build_feature_stages(cat_cols, num_cols):
    """
    Make the pipeline steps that turn text columns into numbers and then glue
    all features into one vector column called 'features'.
    handleInvalid='keep' so unseen categories in the test set don't crash it.
    """
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh")
        for c in cat_cols
    ]
    assembler_inputs = [f"{c}_oh" for c in cat_cols] + num_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs, outputCol="features", handleInvalid="keep"
    )
    return indexers + encoders + [assembler]


def prepare_model_data(df: DataFrame, max_rows: int | None = None) -> DataFrame:
    """
    Select only the columns we need + the label, drop nulls.
    max_rows lets you train on a sub-sample so it is fast while developing.
    """
    cols = CAT_COLS + NUM_COLS + ["is_delayed", "arr_delay"]
    cols = [c for c in cols if c in df.columns]
    data = df.select(cols).dropna()
    if max_rows:
        total = data.count()
        if total > max_rows:
            data = data.sample(False, max_rows / total, seed=42)
    return data


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION  (delayed yes/no)
# ---------------------------------------------------------------------------
def train_classifiers(df: DataFrame, train_frac: float = 0.8) -> dict:
    """
    Train Logistic Regression and Random Forest to predict 'is_delayed'.
    Returns a dict of metrics (AUC, accuracy, F1) for each model so we can
    compare them in a small table / bar chart.
    """
    data = df.withColumnRenamed("is_delayed", "label")
    train, test = data.randomSplit([train_frac, 1 - train_frac], seed=42)

    stages = _build_feature_stages(CAT_COLS, NUM_COLS)

    auc_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    results = {}
    models = {
        "LogisticRegression": LogisticRegression(maxIter=20),
        "RandomForest": RandomForestClassifier(numTrees=40, maxDepth=8, maxBins=512),
    }
    fitted = {}
    for name, clf in models.items():
        pipe = Pipeline(stages=stages + [clf])
        model = pipe.fit(train)
        pred = model.transform(test)
        results[name] = {
            "AUC": round(auc_eval.evaluate(pred), 4),
            "accuracy": round(acc_eval.evaluate(pred), 4),
            "f1": round(f1_eval.evaluate(pred), 4),
        }
        fitted[name] = model

    return {"metrics": results, "models": fitted, "test": test}


def confusion_counts(model, test: DataFrame):
    """Confusion matrix counts (TP/FP/FN/TN) for a trained classifier model."""
    pred = model.transform(test)
    cm = (
        pred.groupBy("label", "prediction")
        .agg(F.count("*").alias("n"))
        .toPandas()
    )
    return cm


def feature_importance(model, top: int = 15):
    """
    Pull feature importances out of a fitted Random Forest pipeline.
    Returns a pandas dataframe (feature, importance) sorted high->low.
    """
    import pandas as pd

    rf = model.stages[-1]
    if not hasattr(rf, "featureImportances"):
        raise ValueError("This model has no featureImportances (use Random Forest).")

    vals = rf.featureImportances.toArray()

    # The one-hot columns expand into many vector slots, so the assembler input
    # names usually don't line up 1-to-1 with the importance values. We try to
    # use the real names, and if the lengths don't match we fall back to f0,f1...
    assembler = model.stages[-2]
    names = list(assembler.getInputCols())
    if len(names) != len(vals):
        names = [f"f{i}" for i in range(len(vals))]

    df_imp = pd.DataFrame({"feature": names, "importance": vals})
    df_imp = df_imp.groupby("feature", as_index=False)["importance"].sum()
    return df_imp.sort_values("importance", ascending=False).head(top)


# ---------------------------------------------------------------------------
# 2. REGRESSION  (predict arrival delay minutes)
# ---------------------------------------------------------------------------
def train_regressor(df: DataFrame, train_frac: float = 0.8) -> dict:
    """
    Random Forest Regressor to predict arr_delay (minutes).
    Returns RMSE, MAE and R2 on the test set + the fitted model.
    """
    data = df.withColumnRenamed("arr_delay", "label")
    train, test = data.randomSplit([train_frac, 1 - train_frac], seed=42)

    stages = _build_feature_stages(CAT_COLS, NUM_COLS)
    rf = RandomForestRegressor(numTrees=40, maxDepth=8, maxBins=512)
    pipe = Pipeline(stages=stages + [rf])
    model = pipe.fit(train)
    pred = model.transform(test)

    def _m(metric):
        return RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName=metric
        ).evaluate(pred)

    metrics = {
        "RMSE": round(_m("rmse"), 3),
        "MAE": round(_m("mae"), 3),
        "R2": round(_m("r2"), 4),
    }
    return {"metrics": metrics, "model": model, "test": test}
