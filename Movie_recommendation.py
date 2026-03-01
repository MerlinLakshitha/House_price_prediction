import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------
data = pd.read_csv(&quot;Tamil_movies_dataset.csv&quot;)
print(&quot;Dataset Preview:&quot;)
print(data.head())
# -----------------------------------------------------
# 2. Create Dummy User Ratings
# (Lab purpose – since dataset has no userId)
# -----------------------------------------------------
# Select required columns
movies = data[[&#39;MovieName&#39;, &#39;Rating&#39;]]
# Create 10 dummy users
np.random.seed(42)
user_ratings = {
f&#39;User_{i}&#39;: np.random.randint(1, 11, len(movies))
for i in range(1, 11)
}
user_movie_matrix = pd.DataFrame(
user_ratings,
index=movies[&#39;MovieName&#39;]
).T
print(&quot;\nUser-Movie Matrix:&quot;)
print(user_movie_matrix.head())
# -----------------------------------------------------
# 3. Compute User Similarity
# -----------------------------------------------------
similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(
similarity,
index=user_movie_matrix.index,
columns=user_movie_matrix.index
)

print(&quot;\nUser Similarity Matrix:&quot;)
print(similarity_df.head())
# -----------------------------------------------------
# 4. Visualization – Similarity Heatmap
# -----------------------------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(similarity_df,
annot=True,
cmap=&quot;YlGnBu&quot;)
plt.title(&quot;User Similarity Heatmap&quot;)
plt.show()
# -----------------------------------------------------
# 5. Recommendation Function
# -----------------------------------------------------
def recommend_movies(user_name, n=5):
similar_users = similarity_df[user_name] \
.sort_values(ascending=False) \
.drop(user_name) \
.head(3) \
.index
# Average ratings of similar users
movies_mean = user_movie_matrix.loc[similar_users].mean()
recommendations = movies_mean.sort_values(
ascending=False
).head(n)
return recommendations
# -----------------------------------------------------
# 6. User Input
# -----------------------------------------------------
user_name = input(&quot;\nEnter User Name (User_1 to User_10): &quot;)
recommendations = recommend_movies(user_name)
print(f&quot;\nTop Recommendations for {user_name}:\n&quot;)
print(recommendations)
# -----------------------------------------------------
# 7. Visualization – Recommended Movies
# -----------------------------------------------------
plt.figure(figsize=(10,5))
recommendations.plot(

kind=&#39;bar&#39;
)
plt.title(f&quot;Top Recommended Movies for {user_name}&quot;)
plt.xlabel(&quot;Movies&quot;)
plt.ylabel(&quot;Predicted Rating&quot;)
plt.show()
