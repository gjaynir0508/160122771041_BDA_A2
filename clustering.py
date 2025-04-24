from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("MallCustomerClustering").getOrCreate()

# Load dataset
df = spark.read.csv("Mall_Customers.csv", header=True, inferSchema=True)

# Drop CustomerID and Name
df = df.drop("CustomerID")

# Encode Gender
indexer = StringIndexer(inputCol="Gender", outputCol="Gender_idx")
df = indexer.fit(df).transform(df).drop("Gender")

# Assemble features
feature_cols = ["Gender_idx", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df)

# Fit KMeans
kmeans = KMeans(k=5, seed=1, featuresCol="features", predictionCol="prediction")
model = kmeans.fit(data)
predictions = model.transform(data)

# Evaluate with silhouette score
evaluator = ClusteringEvaluator()
score = evaluator.evaluate(predictions)

print(f"Mall Customer Clustering Silhouette Score: {score:.4f}")

# Stop Spark
spark.stop()
