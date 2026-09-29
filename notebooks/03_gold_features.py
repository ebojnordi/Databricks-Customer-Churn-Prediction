# Databricks notebook source
from pyspark.sql import functions as F

SOURCE_TABLE = "churn_prediction.silver.telco_customers"
TARGET_TABLE = "churn_prediction.gold.churn_features"

# Read Silver data
df = spark.table(SOURCE_TABLE)
print(f"Rows loaded: {df.count()}")

# COMMAND ----------

# Create business-oriented features

gold_df = (
    df
    # Customer tenure in years
    .withColumn(
        "tenure_years",
        F.round(F.col("tenure") / 12, 2)
    )

    # Whether the customer has internet service
    .withColumn(
        "has_internet",
        F.when(F.col("InternetService") == "No", 0).otherwise(1)
    )

    # Whether the customer has technical support
    .withColumn(
        "has_tech_support",
        F.when(F.col("TechSupport") == "Yes", 1).otherwise(0)
    )

    # Whether the customer has online security
    .withColumn(
        "has_online_security",
        F.when(F.col("OnlineSecurity") == "Yes", 1).otherwise(0)
    )

    # Whether the customer is on a month-to-month contract
    .withColumn(
        "is_month_to_month",
        F.when(F.col("Contract") == "Month-to-month", 1).otherwise(0)
    )
)

# COMMAND ----------

# validate gold features

print(f"Rows: {gold_df.count()}")
print(f"Columns: {len(gold_df.columns)}")

display(
    gold_df.select(
        "customerID",
        "tenure",
        "tenure_years",
        "Contract",
        "is_month_to_month",
        "InternetService",
        "has_internet",
        "TechSupport",
        "has_tech_support",
        "OnlineSecurity",
        "has_online_security",
        "Churn"
    ).limit(10)
)


# COMMAND ----------

display(
    gold_df
    .groupBy("Churn")
    .count().alias("count")
    .withColumn(
        "percentage",
        F.round(F.col("count") / df.count() * 100, 2)
    )
)

# COMMAND ----------

display(
    gold_df
    .groupBy("Contract")
    .agg(
        F.count("*").alias("customers"),
        F.round(F.avg("Churn") * 100, 2).alias("churn_rate_percent")
    )
    .orderBy(F.desc("churn_rate_percent"))
)

# COMMAND ----------

display(
    gold_df
    .groupBy("tenure")
    .agg(
        F.count("*").alias("customers"),
        F.round(F.avg("Churn") * 100, 2).alias("churn_rate_percent")
    )
    .orderBy("tenure")
)

# COMMAND ----------

# Write Gold Delta table
(
    gold_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TARGET_TABLE)
)

print(f"Gold table created: {TARGET_TABLE}")