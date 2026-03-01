import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
# --------------------------------------------
# Step 1 — Load Dataset
# --------------------------------------------
data = pd.read_csv(&quot;spam_dataset.csv&quot;)
# Convert labels to numeric
data[&#39;label&#39;] = data[&#39;label&#39;].map({&#39;ham&#39;: 0, &#39;spam&#39;: 1})
# --------------------------------------------
# Step 2 — Split Data
# --------------------------------------------
X = data[&#39;message&#39;]
y = data[&#39;label&#39;]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
# --------------------------------------------
# Step 3 — TF-IDF Vectorization
# --------------------------------------------
vectorizer = TfidfVectorizer(stop_words=&#39;english&#39;)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# --------------------------------------------
# Step 4 — Train SVM Model
# --------------------------------------------
svm_model = SVC(kernel=&#39;linear&#39;, probability=True)

svm_model.fit(X_train_vec, y_train)
# --------------------------------------------
# Step 5 — Test Prediction
# --------------------------------------------
y_pred = svm_model.predict(X_test_vec)
# --------------------------------------------
# Step 6 — Evaluation Metrics
# --------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(&quot;\nMODEL PERFORMANCE&quot;)
print(&quot;----------------------&quot;)
print(f&quot;Accuracy :{accuracy*100:.2f}%&quot;)
print(f&quot;Error Rate :{error*100:.2f}%&quot;)
print(f&quot;Precision:{precision:.2f}&quot;)
print(f&quot;Recall :{recall:.2f}&quot;)
print(f&quot;F1 Score:{f1:.2f}&quot;)
print(&quot;\nConfusion Matrix:\n&quot;, cm)
# --------------------------------------------
# Step 7 — ROC Curve
# --------------------------------------------
y_prob = svm_model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=&quot;AUC = %0.2f&quot; % roc_auc)
plt.plot([0,1], [0,1], linestyle=&#39;--&#39;)
plt.xlabel(&quot;False Positive Rate&quot;)
plt.ylabel(&quot;True Positive Rate&quot;)
plt.title(&quot;ROC Curve - SVM Spam Classifier&quot;)
plt.legend()
plt.show()
# --------------------------------------------

# Step 8 — USER INPUT PREDICTION
# --------------------------------------------
print(&quot;\nEMAIL SPAM DETECTOR&quot;)
print(&quot;----------------------&quot;)
user_email = input(&quot;Enter the Email Subject/Message: &quot;)
# Transform input using same vectorizer
user_vec = vectorizer.transform([user_email])
prediction = svm_model.predict(user_vec)[0]
print(&quot;\nPrediction Result:&quot;)
if prediction == 1:
print(&quot;This Email is SPAM&quot;)
else:
print(&quot;This Email is NOT SPAM&quot;)
