import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# -------------------------------
# Step 1: Load Dataset
# -------------------------------
dataset = pd.read_csv(&quot;Mall_Customers.csv&quot;)
print(dataset.head())
# -------------------------------
# Step 2: Select Features
# Basis: Income vs Spending
# -------------------------------
X = dataset.iloc[:, [3, 4]].values
# -------------------------------
# Step 3: Elbow Method
# -------------------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init=&#39;k-means++&#39;, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker=&#39;o&#39;)
plt.title(&#39;Elbow Method&#39;)
plt.xlabel(&#39;Number of Clusters&#39;)
plt.ylabel(&#39;WCSS&#39;)
plt.show()
# -------------------------------
# Step 4: Train Model
# -------------------------------
kmeans = KMeans(n_clusters=5, init=&#39;k-means++&#39;, random_state=42)
y_kmeans = kmeans.fit_predict(X)
# -------------------------------
# Step 5: Silhouette Score
# -------------------------------
score = silhouette_score(X, y_kmeans)
print(&quot;Silhouette Score :&quot;, score)
# -------------------------------

# Step 6: Cluster Names
# -------------------------------
cluster_names = {
    0: &quot;Premium Customers&quot;,
    1: &quot;Careful Customers&quot;,
    2: &quot;Impulsive Customers&quot;,
    3: &quot;Budget Customers&quot;,
    4: &quot;Standard Customers&quot;
}
# Add cluster column
dataset[&#39;Cluster&#39;] = y_kmeans
dataset[&#39;Cluster Name&#39;] = dataset[&#39;Cluster&#39;].map(cluster_names)
print(dataset.head())
# -------------------------------
# Step 7: Visualization
# -------------------------------
plt.figure(figsize=(10,7))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, label=cluster_names[0])
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, label=cluster_names[1])
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, label=cluster_names[2])
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, label=cluster_names[3])
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, label=cluster_names[4])
# Centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300,
            label=&#39;Centroids&#39;)
plt.title(&#39;Customer Segmentation&#39;)
plt.xlabel(&#39;Annual Income (k$)&#39;)
plt.ylabel(&#39;Spending Score (1–100)&#39;)
plt.legend()
plt.show()
