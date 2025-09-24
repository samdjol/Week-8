# Unsupervised Learning Analysis: Wine, Poultry Feed, and Crime Data

This repository contains a Jupyter Notebook demonstrating various unsupervised learning techniques applied to three distinct datasets. The analysis covers classification with k-Nearest Neighbors (k-NN), a content-based recommendation system, and comparative clustering with K-Means and Gaussian Mixture Models (GMM).

---

## Project Overview

This project is divided into three independent analyses:

1.  **Wine Cultivar Classification:** A k-NN model is trained to classify wine cultivars based on their chemical properties. The model is optimized using `RandomizedSearchCV` and `GridSearchCV`, achieving perfect accuracy on the test set. Principal Component Analysis (PCA) is used for dimensionality reduction and visualization.

2.  **Poultry Feed Recommendation System:** A content-based recommendation system is built to suggest similar poultry feeds. PCA is used to reduce the dimensionality of the feed's nutritional data, and **Cosine Similarity** is calculated on the principal components to determine feed similarity.

3.  **US Arrests Data Clustering:** K-Means and GMM algorithms are used to segment U.S. states into clusters based on crime rates. The analysis compares the performance of clustering on the original scaled data versus PCA-reduced data, with PCA leading to better cluster separation as measured by the **Silhouette Score**.

---

## 1. Wine Cultivar Classification using k-NN

This analysis aims to classify different types of wine into one of three target classes based on 13 chemical features.

### Methodology
* **Data Preparation:** The wine dataset (178 samples) was loaded, scaled using `StandardScaler`, and split into training and test sets.
* **Dimensionality Reduction:** PCA was applied to the training data. It was determined that **10 principal components** were needed to explain 95% of the variance.
    
* **Model Tuning:** A k-NN classifier was tuned first with `RandomizedSearchCV` to explore a wide range of hyperparameters and then with `GridSearchCV` to fine-tune the best-performing options.
* **Evaluation:** The final model was evaluated on the test set using a classification report and confusion matrix.

### Results
The optimized k-NN model achieved **perfect accuracy (1.0)** on the held-out test data.

* **Best Parameters:** `{'metric': 'euclidean', 'n_neighbors': 26, 'weights': 'distance'}`
* **Test Set Performance:** The model correctly classified all 36 samples in the test set.



The visualization of the first two principal components shows a clear separation between the wine classes, which explains the model's high performance. This model could be effectively used for automated quality control and inventory management.

---

## 2. Poultry Feed Recommendation System

This analysis builds a content-based recommendation system to help users find poultry feeds with similar nutritional profiles.

### Methodology
* **Data Preparation:** The poultry feed dataset (100 samples) was loaded and the 15 numerical features were scaled using `StandardScaler`.
* **Dimensionality Reduction:** To capture the most important nutritional information while reducing noise, PCA was used. It was determined that **8 principal components** were needed to explain over 90% of the data's variance.
* **Similarity Calculation:** **Cosine Similarity** was computed on the 8-component PCA-transformed data to create a similarity matrix between all feeds.
* **Recommendation Function:** A function was created to take a feed name as input and return the top 5 most similar feeds based on their cosine similarity scores.

### Results
The system can successfully recommend feeds with similar nutritional and physical properties. For example, the top recommendation for 'FirstPeck' is 'FeatherUp Complete' with a similarity score of 0.85. The similarity heatmap provides a comprehensive view of relationships across the entire product line.



This tool can be used to suggest alternatives to customers if a product is out of stock or to help them discover new products that meet their specific needs.

---

## 3. Clustering of US Arrests Data

This analysis segments U.S. states into clusters based on rates for four types of violent crimes. Both K-Means and Gaussian Mixture Model (GMM) algorithms were used and compared.

### Methodology
* **Data Preparation:** The arrests dataset (50 samples) was loaded and the four crime rate features were scaled using `StandardScaler`.
* **Determining Optimal Clusters:** The Elbow Method (for K-Means) and the Bayesian Information Criterion (BIC) (for GMM) were used to determine the optimal number of clusters. Both methods pointed to **k=4** clusters.
* **Clustering and Evaluation:** The models were trained on both the original 4D scaled data and on data reduced to 2 principal components by PCA. Cluster quality was evaluated using the **Silhouette Score**.



### Results
The analysis showed that applying **PCA before clustering resulted in better-defined clusters** for both algorithms.

| Model | Data | Silhouette Score |
| :--- | :--- | :--- |
| **K-Means** | **2D (PCA)** | **0.4424** |
| GMM | 2D (PCA) | 0.4129 |
| K-Means | 4D (Original) | 0.3441 |
| GMM | 4D (Original) | 0.3035 |

**K-Means on the PCA-reduced data** provided the highest silhouette score, indicating the most distinct and well-separated clusters. The resulting clusters group states with similar crime profiles, which could be used by law enforcement or sociologists to identify regional patterns.

---
