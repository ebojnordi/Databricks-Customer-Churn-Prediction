# Databricks notebook source
from pyspark.sql import functions as F

SOURCE_TABLE = "churn_prediction.bronze.telco_customers"
TARGET_TABLE = "churn_prediction.silver.telco_customers"

# Read Bronze data
df = spark.table(SOURCE_TABLE)

print(f"Rows loaded: {df.count()}")

# COMMAND ----------

# Check the schema
df.printSchema()

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# Check missing values
display(
    df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])
)

# COMMAND ----------

# Check duplicates
print("Total rows:", df.count())
print("Duplicate rows:", df.count() - df.dropDuplicates().count())

# COMMAND ----------

# Cleaning and standardizing the data
silver_df = (
    df
    .withColumn("customerID", F.trim(F.col("customerID")))
    .withColumn("TotalCharges", F.coalesce(F.expr("try_cast(trim(TotalCharges) AS DOUBLE)"), F.lit(0.0)))
    .withColumn("Churn", F.when(F.col("Churn") == "Yes", 1)
                          .when(F.col("Churn") == "No", 0)
                          .otherwise(None))
)

# COMMAND ----------

# Basic validation
print(f"Rows after transformation: {silver_df.count()}")
print(f"Columns: {len(silver_df.columns)}")

print(f"Null TotalCharges: {silver_df.filter(F.col('TotalCharges').isNull()).count()}")

print(f"Null Churn: {silver_df.filter(F.col('Churn').isNull()).count()}")

# COMMAND ----------

display(
    silver_df
    .filter(F.col("TotalCharges").isNull())
    .select(
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    )
)

# COMMAND ----------

# Write Silver Delta table
(
    silver_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TARGET_TABLE)
)