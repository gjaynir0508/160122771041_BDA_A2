from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("TelcoChurnClassification").getOrCreate()

# Load and prepare dataset
df = spark.read.csv("datasets/Telco-Customer-Churn.csv", header=True, inferSchema=True)

# Drop customerID and rows with missing values
df = df.drop("customerID").dropna()

# Index categorical columns
cat_cols = [col for (col, dtype) in df.dtypes if dtype == "string" and col != "Churn"]
for col in cat_cols + ["Churn"]:
    indexer = StringIndexer(inputCol=col, outputCol=col + "_idx")
    df = indexer.fit(df).transform(df)

# Assemble features
feature_cols = [c + "_idx" for c in cat_cols] + [c for (c, t) in df.dtypes if t in ['double', 'int'] and c != "Churn"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df).withColumnRenamed("Churn_idx", "label")

# Train-test split
train, test = df.randomSplit([0.7, 0.3], seed=42)

# Train classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
model = rf.fit(train)

# Predict and evaluate
predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Churn Prediction Accuracy: {accuracy:.4f}")

# Stop Spark
spark.stop()
