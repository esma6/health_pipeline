# src/spark_pipeline.py
# Katman 1: Apache Spark ile buyuk veri isleme

import os
import pyspark

# Spark icin Java yolu
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot"

# Spark'in kendi klasorunu bul
os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)

print("JAVA_HOME:", os.environ["JAVA_HOME"])
print("SPARK_HOME:", os.environ["SPARK_HOME"])


import time
import json
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from memory_profiler import memory_usage


# ---------------------------------------------------
# PROJE DIZINLERI
# ---------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# SPARK SESSION
# ---------------------------------------------------

def create_spark_session():
    """16 GB RAM icin optimize edilmis Spark oturumu"""

    print("JAVA_HOME:", os.environ.get("JAVA_HOME"))

    spark = (
        SparkSession.builder
        .appName("HealthPipeline")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    print("[Spark] Oturum baslatildi.")
    return spark


# ---------------------------------------------------
# ANOMALI TANIMLARI
# ---------------------------------------------------

ANOMALY_ITEMS = {
    50813: ("Lactate", ">", 2.0),
    50882: ("Bicarbonate", "<", 18.0),
    50912: ("Creatinine", ">", 2.0),
    51222: ("Hemoglobin", "<", 7.0),
    50971: ("Potassium", ">", 5.5),
    50983: ("Sodium", "<", 130.0),
    51006: ("BUN", ">", 40.0),
    50902: ("Chloride", "<", 95.0),
}


# ---------------------------------------------------
# LAB EVENTS PROCESSING
# ---------------------------------------------------

def process_lab_events(spark):

    print("[Spark] labevents.csv okunuyor...")
    t_start = time.time()

    lab_path = str(RAW_DIR / "hosp" / "labevents.csv")
    print("[Spark] Dosya:", lab_path)

    df_lab = spark.read.csv(
        lab_path,
        header=True,
        inferSchema=True
    )

    total_rows = df_lab.count()
    print(f"[Spark] Toplam satir sayisi: {total_rows:,}")

    anomaly_ids = list(ANOMALY_ITEMS.keys())

    df_filtered = df_lab.filter(
        df_lab.itemid.isin(anomaly_ids) &
        df_lab.valuenum.isNotNull()
    )

    from pyspark.sql.functions import col

    anomaly_conditions = []

    for itemid, (_, op, threshold) in ANOMALY_ITEMS.items():

        if op == ">":
            anomaly_conditions.append(
                (col("itemid") == itemid) &
                (col("valuenum") > threshold)
            )
        else:
            anomaly_conditions.append(
                (col("itemid") == itemid) &
                (col("valuenum") < threshold)
            )

    final_condition = anomaly_conditions[0]

    for cond in anomaly_conditions[1:]:
        final_condition = final_condition | cond

    df_anomalies = df_filtered.filter(final_condition)

    anomaly_count = df_anomalies.count()

    print(f"[Spark] Anormal deger sayisi: {anomaly_count:,}")

    output_path = str(PROCESSED_DIR / "lab_anomalies.parquet")

    df_anomalies.write.mode("overwrite").parquet(output_path)

    t_end = time.time()

    print(f"[Spark] Lab anomalileri yazildi: {output_path}")
    print(f"[Spark] Sure: {t_end - t_start:.1f} saniye")

    return {
        "total_rows": total_rows,
        "anomaly_rows": anomaly_count,
        "reduction_ratio": round((1 - anomaly_count / total_rows) * 100, 1),
        "duration_sec": round(t_end - t_start, 1)
    }


# ---------------------------------------------------
# JOIN PATIENT DATA
# ---------------------------------------------------

def join_patient_data(spark):

    print("[Spark] Hasta birlesim islemi basliyor...")

    df_anomalies = spark.read.parquet(
        str(PROCESSED_DIR / "lab_anomalies.parquet")
    )

    df_patients = spark.read.csv(
        str(RAW_DIR / "hosp" / "patients.csv"),
        header=True,
        inferSchema=True
    )

    df_admissions = spark.read.csv(
        str(RAW_DIR / "hosp" / "admissions.csv"),
        header=True,
        inferSchema=True
    )

    df_labitems = spark.read.csv(
        str(RAW_DIR / "hosp" / "d_labitems.csv"),
        header=True,
        inferSchema=True
    )

    df_joined = (
        df_anomalies
        .join(
            df_admissions.select(
                "hadm_id",
                "admittime",
                "hospital_expire_flag"
            ),
            on="hadm_id",
            how="inner"
        )
        .join(
            df_patients.select(
                "subject_id",
                "gender",
                "anchor_age"
            ),
            on="subject_id",
            how="left"
        )
        .join(
            df_labitems.select(
                "itemid",
                "label"
            ),
            on="itemid",
            how="left"
        )
    )

    output_path = str(PROCESSED_DIR / "patient_anomalies.parquet")

    df_joined.write.mode("overwrite").parquet(output_path)

    print(f"[Spark] Birlesik veri yazildi: {output_path}")

    return df_joined.count()


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

def run_spark_pipeline():

    print("=" * 60)
    print("APACHE SPARK PIPELINE BASLIYOR")
    print("=" * 60)

    spark = create_spark_session()

    results = {}

    lab_stats = process_lab_events(spark)
    results["lab_processing"] = lab_stats

    joined_count = join_patient_data(spark)
    results["joined_records"] = joined_count

    print("[Spark] Oturum kapatiliyor...")
    spark.stop()

    print("[Spark] Pipeline tamamlandi!")

    with open(RESULTS_DIR / "spark_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[Spark] Sonuclar kaydedildi.")

    return results


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------

if __name__ == "__main__":
    run_spark_pipeline()