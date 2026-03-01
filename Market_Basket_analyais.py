import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv(&quot;groceries.csv&quot;)
print(&quot;Dataset Preview:&quot;)
print(data.head())
# -------------------------------
# 2. Convert Rows into Transactions
# -------------------------------
# Drop &#39;Item(s)&#39; column if present
data_items = data.drop(columns=[&#39;Item(s)&#39;])
# Convert each row to list &amp; remove NaN
transactions = data_items.apply(
lambda row: row.dropna().tolist(),
axis=1
).tolist()
print(&quot;\nSample Transactions:&quot;)
print(transactions[:5])
# -------------------------------
# 3. One-Hot Encoding
# -------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(&quot;\nEncoded Data Shape:&quot;, df.shape)
# -------------------------------
# 4. Frequent Itemsets
# -------------------------------
frequent_itemsets = apriori(
df,
min_support=0.01,
use_colnames=True
)

print(&quot;\nFrequent Itemsets:&quot;)
print(frequent_itemsets.head())
# -------------------------------
# 5. Association Rules
# -------------------------------
rules = association_rules(
frequent_itemsets,
metric=&quot;confidence&quot;,
min_threshold=0.3
)
print(&quot;\nAssociation Rules:&quot;)
print(rules[[&#39;antecedents&#39;, &#39;consequents&#39;,
&#39;support&#39;, &#39;confidence&#39;, &#39;lift&#39;]].head())
# -------------------------------
# 6. Top 10 Frequent Items
# -------------------------------
item_frequencies = df.sum().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(
x=item_frequencies.head(10).values,
y=item_frequencies.head(10).index
)
plt.title(&quot;Top 10 Frequent Items&quot;)
plt.xlabel(&quot;Frequency&quot;)
plt.ylabel(&quot;Items&quot;)
plt.tight_layout()
plt.show()
# -------------------------------
# 7. Confidence Heatmap
# -------------------------------
rules[&#39;antecedents_str&#39;] = rules[&#39;antecedents&#39;] \
.apply(lambda x: &#39;, &#39;.join(list(x)))
rules[&#39;consequents_str&#39;] = rules[&#39;consequents&#39;] \
.apply(lambda x: &#39;, &#39;.join(list(x)))
top_ants = rules.groupby(&#39;antecedents_str&#39;)[&#39;support&#39;] \
.sum().nlargest(10).index
top_cons = rules.groupby(&#39;consequents_str&#39;)[&#39;support&#39;] \
.sum().nlargest(10).index
filtered = rules[
(rules[&#39;antecedents_str&#39;].isin(top_ants)) &amp;
(rules[&#39;consequents_str&#39;].isin(top_cons))
]

heatmap_data = filtered.pivot_table(
index=&#39;antecedents_str&#39;,
columns=&#39;consequents_str&#39;,
values=&#39;confidence&#39;
)
plt.figure(figsize=(12,8))
sns.heatmap(
heatmap_data,
annot=True,
cmap=&quot;YlGnBu&quot;,
linewidths=0.5,
cbar_kws={&#39;label&#39;: &#39;Confidence&#39;}
)
plt.title(&quot;Heatmap of Confidence for Top Rules&quot;)
plt.xlabel(&quot;Consequents&quot;)
plt.ylabel(&quot;Antecedents&quot;)
plt.tight_layout()
plt.show()
