import os
import pandas as pd
import numpy as np
import glob

# Path to the 'Yearly' directory
data_root = "Data/CRSP/Yearly"
START_YEAR = 2000
END_YEAR = 2021

def load_gzipped_data(data_root):
    """
    Loads all .csv.gz daily files from Yearly/2000 through Yearly/2015.
    Extracts the date from the filename and adds it as a column.
    """
    all_files = []
    for year in range(START_YEAR, END_YEAR + 1):
        pattern = os.path.join(data_root, str(year), "*.csv.gz")
        files = sorted(glob.glob(pattern))
        
        for f in files:
            try:
                df_day = pd.read_csv(f, compression='gzip')
                date_str = os.path.basename(f).replace(".csv.gz", "")
                df_day["date"] = pd.to_datetime(date_str, format="%Y%m%d")
                all_files.append(df_day)
            except Exception as e:
                print(f"Skipping {f}: {e}")

    df = pd.concat(all_files, ignore_index=True)
    return df

def main():
    # Define the folder path
    folder_path = 'Data/'
    
    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Define the file path to save the DataFrame
    file_path = os.path.join(folder_path, 'crsp_df.csv')
    
    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")

if __name__ == "__main__":

    df = load_gzipped_data(data_root)  
    main()