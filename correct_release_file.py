import pandas as pd

csv_path = 'public_release_timeseries.csv'  # Replace with your actual file path

# 1. Read the CSV file
df = pd.read_csv(csv_path)

# 2. Update the 'filename' column
# This replaces every occurrence of ':' with '-' in that column
df['filename'] = df['filename'].str.replace(':', '-', regex=False)

# 3. Save the changes back to the CSV
# index=False prevents pandas from adding a new column for row numbers
df.to_csv(csv_path, index=False)