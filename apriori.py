from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

data = {
  "Transaction": ["T1", "T2", "T3", "T4", "T5"],
    "Items": ["milk, bread, butter", "bread, butter", "milk, bread", "milk, bread, butter", "milk, bread"]

}

#create a dataframe
df = pd.DataFrame(data)

# Transform the data into a one-hot encoded format
df_encoded = df["Items"].str.get_dummies(sep=", ")

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

# Generate association rules
association_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules)
