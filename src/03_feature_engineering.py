from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("dropout-features")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

log = spark.read.csv("data/train/log_train.csv", header=True, inferSchema=True)
enrollment = spark.read.csv("data/train/enrollment_train.csv", header=True, inferSchema=True)
date = spark.read.csv("data/date.csv", header=True, inferSchema=True)

# --- 1. basic activity aggregates ---
basic = log.groupBy("enrollment_id").agg(
    F.count("*").alias("total_events"),
    F.countDistinct(F.to_date("time")).alias("active_days"),
    F.min("time").alias("first_event_time"),
    F.max("time").alias("last_event_time"),
    F.countDistinct("object").alias("distinct_objects_touched"),
)

# --- 2. event-type counts (pivot) ---
event_counts = (
    log.groupBy("enrollment_id")
    .pivot("event", ["access", "problem", "page_close", "navigate", "video", "discussion", "wiki"])
    .count()
    .fillna(0)
)
# rename pivoted columns to be explicit
for col in ["access", "problem", "page_close", "navigate", "video", "discussion", "wiki"]:
    event_counts = event_counts.withColumnRenamed(col, f"{col}_count")

# --- 3. source split ---
source_counts = (
    log.groupBy("enrollment_id")
    .pivot("source", ["server", "browser"])
    .count()
    .fillna(0)
    .withColumnRenamed("server", "server_events")
    .withColumnRenamed("browser", "browser_events")
)

# --- 4. join enrollment -> course -> course end date, compute recency ---
enroll_with_course = enrollment.join(date, on="course_id", how="left")

features = (
    basic
    .join(event_counts, on="enrollment_id", how="left")
    .join(source_counts, on="enrollment_id", how="left")
    .join(enroll_with_course.select("enrollment_id", "course_id", "to"), on="enrollment_id", how="left")
)

features = features.withColumn(
    "days_before_course_end",
    F.datediff(F.col("to"), F.to_date("last_event_time"))
)

features = features.withColumn(
    "session_span_days",
    F.datediff(F.to_date("last_event_time"), F.to_date("first_event_time"))
)

features = features.withColumn(
    "events_per_active_day",
    F.col("total_events") / F.col("active_days")
)

print("=== feature schema ===")
features.printSchema()
print("=== sample ===")
features.show(5, truncate=False)
print("=== row count ===")
print(features.count())

# --- save ---
features.write.mode("overwrite").parquet("data/features_train.parquet")
print("Saved to data/features_train.parquet")

spark.stop()