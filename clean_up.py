import os
import glob
import pandas as pd


def get_bad_file():
    """
    Reads a CSV file and returns a list of filenames where:
    - drs_quality is "FALSE" (or False)
    - OR obs_quality < 0.9
    
    Args:
        csv_file_path (str): Path to the public_release_timeseries.csv file.
        
    Returns:
        list: A list of filenames matching the criteria.
    """
    csv_path = "public_release_timeseries.csv"
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Ensure column names are stripped of whitespace (just in case)
        df.columns = df.columns.str.strip()
        
        # Define the conditions
        # 1. drs_quality = "FALSE". 
        # We convert to string and upper case to handle variations like "False", "FALSE", or boolean False.
        cond_drs = df['drs_quality'].astype(str).str.upper() == 'FALSE'
        
        # 2. obs_quality < 0.9
        # We ensure it's numeric, coercing errors to NaN (which won't be < 0.9)
        cond_obs = pd.to_numeric(df['obs_quality'], errors='coerce') < 0.99
        
        # Apply the filter with OR logic (|)
        bad_files_df = df[cond_drs | cond_obs]
        
        # Extract the 'filename' column as a list
        bad_file_list = bad_files_df['filename'].tolist()
        
        return bad_file_list

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return []

def change_file_extension(f):
    # print(f"Full file name: {f}")
    new_name = f.replace( ".fits", ".stif")
    os.rename(f, new_name)

def clean_up_events(target_dates):
    base_data_dir = "DATA"
    filter_out_list = get_bad_file()
    count = 0
    total = 0
    for date in target_dates:
        folder_path = os.path.join(base_data_dir, date)
        file_list = glob.glob(os.path.join(folder_path, "*.fits"))
        for f in file_list:
            filename = f.replace( "_CCF_A.fits", ".fits").replace(folder_path + '/', '')
            total += 1      
            if filename in filter_out_list:
                count += 1
                # change .fits to .stif
                change_file_extension(f)

    print(f"Filter out {count} files / {total} files")
    return True


if __name__ == "__main__":
    # # Filter not good observations by changing file extension to .stif
    base_data_dir = "DATA"
    quiet_date = "2018-03-29"
    all_subdirs = sorted([d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))])
    target_dates = [d for d in all_subdirs if d != quiet_date]
    clean_up_events(target_dates)
