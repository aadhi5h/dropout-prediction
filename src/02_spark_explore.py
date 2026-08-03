from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("dropout-explore")
    .config("spark.driver.memory", "4g")   # bump if you have RAM to spare
    .getOrCreate()
)

log = spark.read.csv("data/train/log_train.csv", header=True, inferSchema=True)

print("=== schema ===")
log.printSchema()

print("=== row count ===")
print(log.count())

print("=== sample rows ===")
log.show(5, truncate=False)

print("=== distinct event types ===")
log.groupBy("event").count().orderBy(F.desc("count")).show()

print("=== distinct source values ===")
log.groupBy("source").count().show()

print("=== enrollment_id coverage ===")
print("Distinct enrollment_ids in log:", log.select("enrollment_id").distinct().count())

print("=== time range ===")
log.select(F.min("time"), F.max("time")).show()

spark.stop()