# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator



SOURCE_TABLE = "churn_prediction.gold.churn_features"
TARGET_COL = "Churn"

RANDOM_SEED = 42


# Load Gold features
df = spark.table(SOURCE_TABLE)

print("Rows:", df.count())
print("Columns:", len(df.columns))


# COMMAND ----------

# Define target and model features

#       We have duplicated information in some features e.g. Contract and is_month_to_month, so
#       for our first ML model, let's establish a clean baseline using the original customer attributes.

numeric_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

feature_cols = numeric_cols + categorical_cols

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Total features: {len(feature_cols)}")

# COMMAND ----------

# Split data into train and test sets

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=RANDOM_SEED
)

print("Training rows:", train_df.count())
print("Test rows:", test_df.count())

# COMMAND ----------

##########################################
# Numeric: --> 1) VectorAssembler
##########################################
# Categorical: -->
# 1) StringIndexer === Converts categories into numeric indexes ==e.g.==> Month-to-month → 0, One year → 1, Two year → 2
# 2) OneHotEncoder === Represents those categories without implying that 2 > 1 > 0 ==e.g.==> Month-to-month → [1,0,0], One year → [0,1,0], Two year → [0,0,1]
# 3) VectorAssembler
##########################################
# Encode categorical features and assemble feature vector

indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_index",
        handleInvalid="keep"
    )
    for col in categorical_cols
]

encoder = OneHotEncoder(
    inputCols=[f"{col}_index" for col in categorical_cols],
    outputCols=[f"{col}_encoded" for col in categorical_cols]
)

encoded_cols = [f"{col}_encoded" for col in categorical_cols]

assembler = VectorAssembler(
    inputCols=numeric_cols + encoded_cols,
    outputCol="features"
)


# COMMAND ----------

# First baseline model: Logistic Regression 

lr = LogisticRegression(
    featuresCol="features",
    labelCol=TARGET_COL,
    maxIter=50
)

lr_pipeline = Pipeline(
    stages=indexers + [encoder, assembler, lr]
)

lr_model = lr_pipeline.fit(train_df)

# Generate Logistic Regression predictions

lr_predictions = lr_model.transform(test_df)

display(
    lr_predictions.select(
        "customerID",
        TARGET_COL,
        "prediction",
        "probability"
    ).limit(10)
)

# COMMAND ----------

# Evaluate Logistic Regression

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol=TARGET_COL,
    predictionCol="prediction",
    metricName="accuracy"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol=TARGET_COL,
    predictionCol="prediction",
    metricName="f1"
)

roc_auc_evaluator = BinaryClassificationEvaluator(
    labelCol=TARGET_COL,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

lr_accuracy = accuracy_evaluator.evaluate(lr_predictions)
lr_f1 = f1_evaluator.evaluate(lr_predictions)
lr_roc_auc = roc_auc_evaluator.evaluate(lr_predictions)

print(f"Accuracy: {lr_accuracy:.4f}")
print(f"F1: {lr_f1:.4f}")
print(f"ROC-AUC: {lr_roc_auc:.4f}")

# COMMAND ----------

# Calculate churn-specific metrics

tp = lr_predictions.filter(
    (F.col(TARGET_COL) == 1) & (F.col("prediction") == 1)
).count()

fp = lr_predictions.filter(
    (F.col(TARGET_COL) == 0) & (F.col("prediction") == 1)
).count()

fn = lr_predictions.filter(
    (F.col(TARGET_COL) == 1) & (F.col("prediction") == 0)
).count()

churn_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
churn_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
churn_f1 = (
    2 * churn_precision * churn_recall /
    (churn_precision + churn_recall)
    if (churn_precision + churn_recall) > 0
    else 0
)

print(f"Churn Precision: {churn_precision:.4f}")
print(f"Churn Recall: {churn_recall:.4f}")
print(f"Churn F1: {churn_f1:.4f}")

# COMMAND ----------

# Train Random Forest

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol=TARGET_COL,
    numTrees=100,
    maxDepth=8,
    seed=RANDOM_SEED
)

rf_pipeline = Pipeline(
    stages=indexers + [encoder, assembler, rf]
)

rf_model = rf_pipeline.fit(train_df)

# COMMAND ----------

# Generate Random Forest predictions

rf_predictions = rf_model.transform(test_df)

display(
    rf_predictions.select(
        "customerID",
        TARGET_COL,
        "prediction",
        "probability"
    ).limit(10)
)

# COMMAND ----------

# Evaluate Random Forest

rf_accuracy = accuracy_evaluator.evaluate(rf_predictions)
rf_f1 = f1_evaluator.evaluate(rf_predictions)
rf_roc_auc = roc_auc_evaluator.evaluate(rf_predictions)

tp = rf_predictions.filter(
    (F.col(TARGET_COL) == 1) & (F.col("prediction") == 1)
).count()

fp = rf_predictions.filter(
    (F.col(TARGET_COL) == 0) & (F.col("prediction") == 1)
).count()

fn = rf_predictions.filter(
    (F.col(TARGET_COL) == 1) & (F.col("prediction") == 0)
).count()

rf_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
rf_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
rf_churn_f1 = (
    2 * rf_precision * rf_recall /
    (rf_precision + rf_recall)
    if (rf_precision + rf_recall) > 0
    else 0
)

print(f"Accuracy: {rf_accuracy:.4f}")
print(f"F1: {rf_f1:.4f}")
print(f"ROC-AUC: {rf_roc_auc:.4f}")
print(f"Churn Precision: {rf_precision:.4f}")
print(f"Churn Recall: {rf_recall:.4f}")
print(f"Churn F1: {rf_churn_f1:.4f}")

# COMMAND ----------

# Compare model performance

comparison = spark.createDataFrame([
    (
        "Logistic Regression",
        lr_accuracy,
        lr_roc_auc,
        churn_precision,
        churn_recall,
        churn_f1
    ),
    (
        "Random Forest",
        rf_accuracy,
        rf_roc_auc,
        rf_precision,
        rf_recall,
        rf_churn_f1
    )
], [
    "Model",
    "Accuracy",
    "ROC_AUC",
    "Churn_Precision",
    "Churn_Recall",
    "Churn_F1"
])

display(comparison)