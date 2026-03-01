# customer_churn_rf.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# ---------- Step 1: Load Excel file ----------
df = pd.read_excel(&quot;customer_churn.xlsx&quot;)
# ---------- Step 2: Select features and target ----------
X = df[[&#39;CustomerAge&#39;, &#39;MonthlyCharges&#39;, &#39;Tenure&#39;,
        &#39;ContractType&#39;, &#39;InternetService&#39;,
        &#39;SupportCalls&#39;, &#39;TotalSpend&#39;]]
y = df[&#39;Churn&#39;]
# ---------- Step 3: Train Random Forest ----------
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X, y)
# ---------- Step 4: Prediction ----------
y_pred = model.predict(X)
# ---------- Step 5: Accuracy and Error ----------
accuracy = accuracy_score(y, y_pred)
error = 1 - accuracy
print(&quot;Accuracy:&quot;, round(accuracy, 2))
print(&quot;Error Rate:&quot;, round(error, 2))
# ---------- Step 6: Confusion Matrix ----------
print(&quot;\nConfusion Matrix:&quot;)
print(confusion_matrix(y, y_pred))
# ---------- Step 7: User Input ----------
print(&quot;\n--- Customer Churn Prediction ---&quot;)
age = float(input(&quot;Customer Age: &quot;))
mc = float(input(&quot;Monthly Charges: &quot;))
tenure = float(input(&quot;Tenure (months): &quot;))
contract = int(input(&quot;Contract Type (1=Long, 0=Monthly): &quot;))
internet = int(input(&quot;Internet Service (1=Yes, 0=No): &quot;))
calls = int(input(&quot;Support Calls: &quot;))
spend = float(input(&quot;Total Spend: &quot;))
result = model.predict([[age, mc, tenure, contract, internet, calls, spend]])
if result[0] == 1:

    print(&quot;Customer is likely to CHURN&quot;)
else:
    print(&quot;Customer is likely to STAY&quot;)
