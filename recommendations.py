from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("MovieLensRecommendation").getOrCreate()

# Load MovieLens dataset
ratings = spark.read.csv("ml-1m/u.data", sep="\t", header=False, inferSchema=True)
ratings = ratings.select(
    col("_c0").alias("userId"),
    col("_c1").alias("movieId"),
    col("_c2").alias("rating")
)

# Load movie titles (optional, for displaying movie names)
movies = spark.read.csv("ml-1m/u.item", sep="|", header=False, inferSchema=True)
movies = movies.select(
    col("_c0").alias("movieId"),
    col("_c1").alias("title")
)

# Split data into training and testing sets
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

# Build ALS model
als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train)

# Make predictions
predictions = model.transform(test)

# Evaluate the model using RMSE (Root Mean Squared Error)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse:.4f}")

# Generate top 10 movie recommendations for each user
user_recs = model.recommendForAllUsers(10)

# Show top 10 recommendations for a specific user (userId = 1)
user_1_recs = user_recs.filter(user_recs.userId == 1).select("recommendations").collect()
print("Top 10 recommendations for User 1:")
for rec in user_1_recs[0].recommendations:
    movie_id = rec['movieId']
    movie_title = movies.filter(movies.movieId == movie_id).select("title").collect()[0]["title"]
    print(f"Movie ID: {movie_id}, Title: {movie_title}")

# Stop Spark session
spark.stop()
