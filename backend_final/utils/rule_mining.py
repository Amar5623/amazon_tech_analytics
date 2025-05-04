# utils/rule_mining.py

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Function to generate frequent itemsets using the Apriori algorithm
def apriori_rule_mining(data: pd.DataFrame, min_support: float = 0.05, min_threshold: float = 0.7) -> pd.DataFrame:
    # Convert the specifications data to a binary matrix for apriori mining
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    return rules

# Function to preprocess the specifications into a one-hot encoded format
def preprocess_specs_for_mining(specs_df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encoding the spec values
    specs_onehot = pd.get_dummies(specs_df, columns=["spec_name", "spec_value"])
    return specs_onehot