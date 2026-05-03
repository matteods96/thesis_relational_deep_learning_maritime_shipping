#Libraries needed
import geopandas as gpd
import pandas as pd
import os
import pyarrow.parquet as pq





DUCKDB_DIR = "duckdb_files"
PARQUET_DIR = "parquet_files"
port_data_filename='port_locodes.csv'
col_name='destination' # column in the parquet files and duckdb

#Loading official port data from World Port Index
port_data = pd.read_csv(
    'port_locodes.csv',
    sep=';',
    header=None,
    names=['port_name', 'port_code', 'polygon']
)

# Extract distinct port names
#distinct_ports = set(gdf["port_name"].dropna().unique().tolist())
distinct_ports = set(port_data["port_code"])

print(distinct_ports)
print("Number of distinct ports in port_locodes.csv file:", len(distinct_ports))

distinct_ais_ports={}
destinations=[]

for filename in os.listdir(PARQUET_DIR):
    if filename.endswith(".parquet"):
        filepath = os.path.join(PARQUET_DIR, filename)

        table = pq.read_table(filepath)

        if col_name in table.column_names:
            col_values = table[col_name].to_pylist()
            unique_vals = set(v for v in col_values if v is not None)

            # APPEND EACH VALUE IMMEDIATELY
            for val in unique_vals:
                destinations.append(val)


distinct_destinations=set(destinations)
print("Distinct AIS destinations:", len(distinct_destinations))
distinct_ais_ports=distinct_ports.intersection(distinct_destinations)
print("Matched AIS ports:", len(distinct_ais_ports)) 
print(distinct_ais_ports)

# Create a sorted list for consistent ordering 
sorted_ports = sorted(list(distinct_ais_ports)) 
#  Building a  DataFrame 
df = pd.DataFrame({ "row_number": range(1, len(sorted_ports) + 1), "ais_port": sorted_ports }) 
# Save to Excel 
output_file = "distinct_ais_ports_locodes_based.csv" 
df.to_csv(output_file, index=False) 
print(f"Saved Excel file: {output_file}")