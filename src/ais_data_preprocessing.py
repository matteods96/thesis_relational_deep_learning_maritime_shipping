import os
import pandas as pd
import pyarrow.parquet as pq
import duckdb

PARQUET_DIR = "parquet_files"
PROCESSED_PARQUET_DIR="processed_parquet_files"
os.makedirs(PROCESSED_PARQUET_DIR, exist_ok=True)

# Load distinct AIS ports
distincts_ports_filename='distinct_ais_ports_locodes_based.csv'
distinct_ports_df=pd.read_csv(distincts_ports_filename)
distinct_ports=distinct_ports_df['ais_port'].values.tolist()
relevant_ship_types=['Tanker', 'Cargo']
relevant_type_of_mobile=['Class A']



# Iterate over parquet files
for file in os.listdir(PARQUET_DIR):
    if file.endswith(".parquet"):
        path = os.path.join(PARQUET_DIR, file)
        df = duckdb.execute( """ SELECT * FROM read_parquet(?) WHERE type_of_mobile IN ? AND destination IN ? AND "shiptype" IN ? """, [path,relevant_type_of_mobile, distinct_ports, relevant_ship_types] ).df()
        out_name = file.replace(".parquet", "_processed.parquet") 
        out_path = os.path.join(PROCESSED_PARQUET_DIR, out_name) 
        df.to_parquet(out_path)

print('Finishing preprocessing files saved in processed_parquet_files directory')