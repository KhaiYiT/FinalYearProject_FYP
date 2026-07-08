import pandas as pd
import glob

# Find all parquet files
files = glob.glob("data/*.parquet")
summary = []

for file in files:
    df = pd.read_parquet(file)
    rows, cols = df.shape
    summary.append({
        "File Name": file.split("\\")[-1],
        "Rows": rows,
        "Columns": cols
    })

summary_df = pd.DataFrame(summary)

# Print table
print(summary_df)