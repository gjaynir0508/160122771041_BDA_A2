# Big Data Analytics - Assignment 2

## Submitted by

|  |  |
| --- | --- |
| Name | Jayanth Gudimella |
| Roll Number | 160122771041 |
| Section | AI&DS 1 (I1) Sem 6 |

## Overview

This repository contains the code for three different questions as part of Assignment 2:

1. **Classification on Telco Churn Dataset**  
   The goal of this task is to build a classification model to predict customer churn in a telecommunications company using the **Telco Churn** dataset.

2. **Clustering on Mall Customer Dataset**  
   The goal of this task is to perform clustering on a **Mall Customer** dataset to segment customers into groups based on their purchasing behaviors.

3. **Recommendation Engine on MovieLens 1M Dataset**  
   This task builds a recommendation system using the **MovieLens 1M** dataset, employing **Collaborative Filtering** (ALS algorithm) to recommend movies to users.

---

## Project Structure

```
/ (root)
│
├── classification.py                    # Classification model for Telco Churn dataset
├── clustering.py                        # Clustering model for Mall Customer dataset
├── recommendation.py                    # Recommendation engine for MovieLens 1M dataset
├── ml-1m/                               # MovieLens 1M dataset directory
│   ├── u.data                           # Ratings data
│   ├── u.item                           # Movie metadata
│   ├── u.user                           # User data
│   └── ...                              # Other auxiliary files
```

---

## Setup Instructions

### 1. **Dataset Downloads:**
   - **Telco Churn Dataset**: [Telco Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
   - **Mall Customer Dataset**: [Mall Customer Dataset](https://www.kaggle.com/datasets/umytlygenc/mall-customers)
   - **MovieLens 1M Dataset**: [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)

### 2. **Install Dependencies:**

Make sure to install the required dependencies by running the following:
```bash
pip install pyspark
pip install pandas
pip install scikit-learn
```

### 3. **Running the Scripts:**

You can run each of the scripts as follows:
```bash
python classification_telco_churn.py
python clustering_mall_customers.py
python recommendation_movielens.py
```

---

## 1. **Classification on Telco Churn Dataset**

### File: `classification.py`

This script builds a classification model to predict whether a customer will churn based on historical data from a telecommunications company. We use a **Random Forest** classifier to predict churn based on various customer features such as `Contract`, `PaymentMethod`, `tenure`, `MonthlyCharges`, etc.

### Steps:
- Data preprocessing to handle missing values and categorical variables.
- Splitting the data into training and test sets.
- Training a Random Forest model and evaluating its performance using accuracy and classification report.

### Sample Output:
```bash
Accuracy: 0.81
Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.80      0.80       137
           1       0.82      0.82      0.82       143

    accuracy                           0.81       280
   macro avg       0.81      0.81      0.81       280
weighted avg       0.81      0.81      0.81       280
```
The classification report provides precision, recall, and F1-score for each class (Churned vs. Not Churned).

---

## 2. **Clustering on Mall Customer Dataset**

### File: `clustering.py`

This script performs clustering on the **Mall Customer Dataset**, which contains information about customers such as their annual income and spending score. The goal is to segment the customers into different clusters to better understand purchasing behavior.

### Steps:
- Data preprocessing (e.g., handling missing values).
- Standardizing the data (scaling).
- Performing **K-Means Clustering** to segment customers into groups.

### Sample Output:
```bash
Cluster Centers (Income, Spending Score):
[ 55.    35.57]
[ 24.1   78.34]
[ 93.8   15.71]
[ 56.3   87.11]
[ 85.4   20.02]
```
These are the cluster centers, showing the **annual income** and **spending score** of each cluster's centroid.

---

## 3. **Recommendation Engine on MovieLens 1M Dataset**

### File: `recommendation.py`

This script builds a movie recommendation system using the **MovieLens 1M** dataset. It employs **Collaborative Filtering** with the **ALS (Alternating Least Squares)** algorithm to recommend movies to users based on their past ratings.

### Steps:
- Data preprocessing to load the ratings and movie metadata.
- Splitting the data into training and testing sets.
- Building the ALS model and training it with the user-movie-rating data.
- Evaluating the model with RMSE (Root Mean Squared Error).
- Generating top 10 movie recommendations for a given user.

### Sample Output:
```bash
RMSE: 0.8971
Top 10 recommendations for User 1:
Movie ID: 1193, Title: One Flew Over the Cuckoo's Nest (1975)
Movie ID: 661, Title: The Godfather (1972)
Movie ID: 914, Title: Star Wars: Episode IV - A New Hope (1977)
Movie ID: 3408, Title: The Shawshank Redemption (1994)
Movie ID: 258, Title: Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
Movie ID: 1210, Title: Pulp Fiction (1994)
Movie ID: 708, Title: The Empire Strikes Back (1980)
Movie ID: 2918, Title: Casablanca (1942)
Movie ID: 1214, Title: The Silence of the Lambs (1991)
Movie ID: 265, Title: Citizen Kane (1941)
```
This shows the **top 10 recommended movies** for a specific user based on their past ratings. The output includes **movie titles** and **IDs**.

---

## Evaluation

### 1. **Classification (Telco Churn)**:
- **Metric**: Accuracy, Precision, Recall, F1-Score
- **Goal**: Predict if a customer will churn based on several features.

### 2. **Clustering (Mall Customer)**:
- **Metric**: Cluster Centers, Visualization
- **Goal**: Segment customers based on purchasing behaviors (Income and Spending Score).

### 3. **Recommendation (MovieLens 1M)**:
- **Metric**: RMSE (Root Mean Squared Error)
- **Goal**: Recommend movies to users based on their ratings using collaborative filtering.

---

## Conclusion

This repository showcases three different approaches in Data Science:

- **Classification**: Predicting customer churn with a classification model.
- **Clustering**: Segmentation of mall customers based on their income and spending scores.
- **Recommendation System**: Recommending movies to users based on collaborative filtering using the ALS algorithm.

These tasks demonstrate practical applications of machine learning techniques, including classification, clustering, and collaborative filtering, on real-world datasets.

---

## Requirements

- Python 3.x
- PySpark 3.x
- Pandas
- Scikit-learn
- MovieLens 1M Dataset, Telco Churn Dataset, Mall Customer Dataset (Available for download)
