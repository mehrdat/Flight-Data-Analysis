"""
config.py
---------
All the project settings live here in one place (paths + spark session).
So if a path or a setting changes, we only edit it here.
"""

import os
import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# PROJECT_ROOT = the "assignment 2" folder (one level up from this src file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# the two data files we care about
FULL_DATA = DATA_DIR / "flight_data_2024.csv"          # the big 1.3 GB file
SAMPLE_DATA = DATA_DIR / "flight_data_2024_sample.csv"  # small 10k rows for quick tests

# make sure the output folders exist (does nothing if already there)
for _d in (OUTPUT_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SPARK SESSION
# ---------------------------------------------------------------------------
# NOTE: the old notebook crashed with "Could not initialize class
# ByteArrayMethods". That happens on newer Java (17+) because Spark needs
# some extra "--add-opens" flags to be allowed to use internal java memory.
# The lines below add those flags so the session starts cleanly.
_JAVA_OPENS = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED"
)


def _java_home_17() -> str | None:
    """Return a local Java 17 home on macOS if one is installed."""
    try:
        result = subprocess.run(
            ["/usr/libexec/java_home", "-v", "17"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.environ.get("JAVA_HOME")


_JAVA_HOME = _java_home_17()
if _JAVA_HOME:
    os.environ["JAVA_HOME"] = _JAVA_HOME


def get_spark(app_name: str = "flight-delay-2024", memory: str = "8g"):
    """
    Build (or reuse) a local Spark session.

    app_name : name shown in the Spark UI.
    memory   : how much RAM to give the driver. 4g is fine for the full file
            son a normal laptop, raise it if you have more RAM.
    """
    # make python use the same interpreter for driver and workers
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .master("local[*]")              # use all CPU cores on this machine
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.extraJavaOptions", _JAVA_OPENS)
        .config("spark.executor.extraJavaOptions", _JAVA_OPENS)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    # less noisy logs (only show warnings and errors)
    spark.sparkContext.setLogLevel("WARN")
    return spark
