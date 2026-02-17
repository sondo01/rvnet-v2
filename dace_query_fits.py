import os
from dace_query.sun import Sun

# 1. Configuration
start_date = "2015-07-29"
end_date = "2018-05-24"
output_dir = f"DATA"

# Ensure output directory exists to avoid FileNotFoundError
os.makedirs(output_dir, exist_ok=True)

# 2. Define Range Filter
# [cite_start]Using 'min' and 'max' allows you to query a range of dates [cite: 1]
query_filters = {
    "date_night": {
        "min": start_date,
        "max": end_date
    }
}

print(f"Querying DACE database for range: {start_date} to {end_date}...")

# 3. Query the Database
query_result = Sun.query_database(
    filters=query_filters,
    output_format='pandas'
)

if query_result.empty:
    print("No observations found in this date range.")
else:
    # [cite_start]Extract the file identifiers (file_rootpath) [cite: 1]
    files_to_download = query_result["file_rootpath"].tolist()
    total_files = len(files_to_download)
    
    print(f"Found {total_files} observations.")
    print(f"Downloading files to folder: '{output_dir}'...")

    # 4. Download
    # [cite_start]This downloads the CCF files (Cross Correlation Function) [cite: 18]
    Sun.download_files(
        files=files_to_download,
        file_type="ccf", 
        output_directory=output_dir
    )
    
    print("✅ Download complete.")