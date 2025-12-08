import pandas as pd

# Read the CSV file
df = pd.read_csv('../output/LapRLS_pre_gpcr_test.csv')

# 1. Remove records where pre value is 0
df = df[df['pre'] != 0]

# 2. Sort by pre value
df = df.sort_values(by='pre')

# 3. Calculate 10% thresholds
total_records = len(df)
top_10_percent = int(total_records * 0.1)
bottom_10_percent = int(total_records * 0.1)

# 4. Get top 10% and bottom 10% records
top_10_df = df.tail(top_10_percent).copy()
bottom_10_df = df.head(bottom_10_percent).copy()

# 5. Add labels (1 for top 10%, 0 for bottom 10%)
top_10_df['label'] = 1
bottom_10_df['label'] = 0

# 6. Combine the results
result_df = pd.concat([top_10_df, bottom_10_df])

# 7. Write to Excel
result_df.to_excel('../output/labeled_map_gpcr.xlsx', index=False)
