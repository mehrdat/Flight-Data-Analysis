"""
data_loader.py
--------------
Reads the flight CSV into a Spark DataFrame.

We give Spark a fixed schema instead of using inferSchema=True. Reasons:
1. inferSchema reads the whole 1.3 GB file twice (slow).
2. a fixed schema makes the column types stable and predictable.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src import config

# the schema is based on the data dictionary (flight_data_2024_data_dictionary.csv)
FLIGHT_SCHEMA = StructType([
    StructField("year", IntegerType()),
    StructField("month", IntegerType()),
    StructField("day_of_month", IntegerType()),
    StructField("day_of_week", IntegerType()),
    StructField("fl_date", StringType()),            # parse to date later
    StructField("op_unique_carrier", StringType()),  # airline code, e.g. AA
    StructField("op_carrier_fl_num", DoubleType()),
    StructField("origin", StringType()),             # airport code, e.g. JFK
    StructField("origin_city_name", StringType()),
    StructField("origin_state_nm", StringType()),
    StructField("dest", StringType()),
    StructField("dest_city_name", StringType()),
    StructField("dest_state_nm", StringType()),
    StructField("crs_dep_time", IntegerType()),      # scheduled dep time (hhmm)
    StructField("dep_time", DoubleType()),           # actual dep time (hhmm)
    StructField("dep_delay", DoubleType()),          # minutes, can be negative
    StructField("taxi_out", DoubleType()),
    StructField("wheels_off", DoubleType()),
    StructField("wheels_on", DoubleType()),
    StructField("taxi_in", DoubleType()),
    StructField("crs_arr_time", IntegerType()),
    StructField("arr_time", DoubleType()),
    StructField("arr_delay", DoubleType()),          # minutes, can be negative
    StructField("cancelled", IntegerType()),         # 0/1
    StructField("cancellation_code", StringType()),  # mostly null
    StructField("diverted", IntegerType()),          # 0/1
    StructField("crs_elapsed_time", DoubleType()),
    StructField("actual_elapsed_time", DoubleType()),
    StructField("air_time", DoubleType()),
    StructField("distance", DoubleType()),
    StructField("carrier_delay", IntegerType()),
    StructField("weather_delay", IntegerType()),
    StructField("nas_delay", IntegerType()),
    StructField("security_delay", IntegerType()),
    StructField("late_aircraft_delay", IntegerType()),
])


def load_flights(spark: SparkSession, sample: bool = False) -> DataFrame:
    """
    Load the flight data.

    sample=True  -> small 10k file (use this while writing/testing code).
    sample=False -> full 1.3 GB file (use this for the real run).
    """
    path = config.SAMPLE_DATA if sample else config.FULL_DATA
    df = (
        spark.read
        .option("header", True)
        .schema(FLIGHT_SCHEMA)
        .csv(str(path))
    )
    return df
