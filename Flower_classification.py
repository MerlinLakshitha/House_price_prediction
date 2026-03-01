# flower_decision_tree.py
# Program: Flower Classification using Decision Tree Algorithm

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------- Step 1: Load dataset ----------
iris = load_iris()

# Convert to DataFrame for clarity
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df[&#39;species&#39;] = iris.target
df[&#39;species&#39;] = df[&#39;species&#39;].replace({0:&#39;setosa&#39;, 1:&#39;versicolor&#39;, 2:&#39;virginica&#39;})

print(&quot;Sample Data:&quot;)
print(df.head())

# ---------- Step 2: Split into features and labels ----------
X = df[iris.feature_names]
y = df[&#39;species&#39;]

# ---------- Step 3: Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ---------- Step 4: Train Decision Tree model ----------

model = DecisionTreeClassifier(criterion=&#39;entropy&#39;, random_state=1)
model.fit(X_train, y_train)

# ---------- Step 5: Make predictions ----------
y_pred = model.predict(X_test)

# ---------- Step 6: Evaluate model ----------
acc = accuracy_score(y_test, y_pred)
print(f&quot;\nModel Accuracy: {acc*100:.2f}%&quot;)

# ---------- Step 7: Visualize Decision Tree ----------
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=iris.feature_names,
class_names=iris.target_names)
plt.title(&quot;Decision Tree for Iris Flower Classification&quot;)
plt.show()

# ---------- Step 8: Predict for user input ----------
print(&quot;\n------ Predict Flower Type ------&quot;)
sepal_length = float(input(&quot;Enter Sepal Length (cm): &quot;))
sepal_width  = float(input(&quot;Enter Sepal Width (cm): &quot;))
petal_length = float(input(&quot;Enter Petal Length (cm): &quot;))
petal_width  = float(input(&quot;Enter Petal Width (cm): &quot;))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
print(f&quot;\n The predicted flower species is: {prediction.capitalize()}&quot;)
