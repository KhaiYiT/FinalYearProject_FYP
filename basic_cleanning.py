import pandas as pd

# Read the 2022 parquet file (smaller file size for faster processing)
df = pd.read_parquet("data/Combined_Flights_2022.parquet")

# check original dataset
print("\nOriginal shape:" , df.shape)

# Remove cancelled flights row
df = df[df["Cancelled"] == 0]

# Remove missing row of target variable
df = df.dropna(subset=["ArrDel15"])

# Check Basic Cleaning Result
print("\nAfter cleaning:", df.shape)

# Check class distribution for sampling strategy (class balancing)
print("\nClass distribution:", df["ArrDel15"].value_counts())

# Separate classes
#df_0 = df[df["ArrDel15"] == 0]
#df_1 = df[df["ArrDel15"] == 1]

# Random sample from each class
#df_0_sample = df_0.sample(n=5000, random_state=42)
#df_1_sample = df_1.sample(n=5000, random_state=42)

# Combine both classes
#df_balanced = pd.concat([df_0_sample, df_1_sample])

# Shuffle rows
#df_balanced = df_balanced.sample(frac=1, random_state=42)

# Check balanced distribution
#print("\nBalanced class distribution:", df_balanced["ArrDel15"].value_counts())

# Save parquet for Python
#df_balanced.to_parquet("balanced_2022(10k).parquet", index=False)

# Save CSV for Orange
#df_balanced.to_csv("balanced_2022(10k).csv", index=False)