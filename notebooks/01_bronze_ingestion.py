# Databricks notebook source

SOURCE_PATH = "/Volumes/churn_prediction/bronze/raw/Telco_Customer_Churn.csv"
TARGET_TABLE = "churn_prediction.bronze.telco_customers"

# Read raw csv
bronze_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(SOURCE_PATH)
)



# COMMAND ----------

# Write Bronze Delta table
(
    bronze_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TARGET_TABLE)
)



# COMMAND ----------

# Basic validation
print(f"Rows loaded: {bronze_df.count()}")
print(f"Columns loaded: {len(bronze_df.columns)}")
print(f"Bronze Table: {TARGET_TABLE}")