import duckdb
import pandas as pd
import relbench
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.json

from relbench.base import Database, Dataset, Table
from relbench.datasets import get_dataset, get_dataset_names, register_dataset
from relbench.utils import unzip_processor

class MaritimeShippingAISDataset(Dataset):
    ################################################################################
    # Choose the val_timestamp and test_timestamp carefully
    ################################################################################
    val_timestamp = pd.Timestamp("2025-01-25")
    test_timestamp = pd.Timestamp("2025-02-14")

    def make_db(self) -> Database:
        DB_PATH = "maritime_shipping_ais.db" 
        con=duckdb.connect(DB_PATH)

        #Loading  duckdb database 

        vessels=con.execute("""
        select * 
        FROM vessel
        """).df().drop_duplicates()

        print("The shape  of the vessel table is",vessels.shape)
        print("The first 5 rows of the vessel table are:")
        print(vessels.head())


        vessels_details=con.execute("""
        select * 
        FROM vessel_details
        """).df().drop_duplicates()
        vessels_details['timestamp'] = vessels_details['timestamp'].astype('datetime64[s]')


        print("The shape  of the vessel details table is",vessels_details.shape)
        print("The first 5 rows of the vessel details table are:")
        print(vessels_details.head())

        voyages=con.execute("""
        select *
        FROM voyage;
        """).df().drop_duplicates()
        voyages['timestamp'] = voyages['timestamp'].astype('datetime64[s]')

        print("The shape  of the voyage table is",voyages.shape)
        print("The first 5 rows of the voyage table are:")
        print(voyages.head())

        positions=con.execute("""
        select *
        FROM position;
        """).df().drop_duplicates()
        positions['timestamp'] = positions['timestamp'].astype('datetime64[s]')

        print("The shape  of the position table is",positions.shape)
        print("The first 5 rows of the position table are:")
        print(positions.head())



        ports=con.execute("""
        select *
        FROM port;
        """).df()

        print("The shape  of the port table is",ports.shape)
        print("The first 5 rows of the port table are:")
        print(ports.head())
        for p in set(ports['country_name']):
            print('Country name: ',p)

        def print_table_info(df, table_name):
            """Print column names, data types and missing values"""
            print(f"\n{table_name} table - Column : Type : Na")
            print("-" * 50)
            for col in df.columns:
                missing = df[col].isna().sum()
                print(f"{col} : {df[col].dtype} : {missing}")


        print_table_info(vessels, "vessel")
        print_table_info(vessels_details, "vessel_details")
        print_table_info(voyages, "voyage") 
        print_table_info(positions, "position")
        print_table_info(ports, "port")


     
        #Collecting all tables in the database as relbench.base.Table objects.

        tables = {}

        tables["vessels"] = Table(
            df=pd.DataFrame(vessels),
            fkey_col_to_pkey_table={},
            pkey_col='mmsi',
            time_col=None,
        )

        tables["vessels_details"] = Table(
            df=pd.DataFrame(vessels_details),
            fkey_col_to_pkey_table={'mmsi':'vessels'},
            pkey_col='vessel_details_id',
            time_col='timestamp',
        )

        tables["ports"] = Table(
            df=pd.DataFrame(ports),
            fkey_col_to_pkey_table={},
            pkey_col="port_code",
            time_col=None,
        )

        

        tables["positions"] = Table(
            df=pd.DataFrame(positions),
            fkey_col_to_pkey_table={"mmsi":'vessels'},
            pkey_col='position_id',
            time_col='timestamp',
        )

        tables["voyages"] = Table(
            df=pd.DataFrame(voyages),
            fkey_col_to_pkey_table={"mmsi":'vessels',"destination":"ports"},
            pkey_col='voyage_id',
            time_col="timestamp",
        )

        return Database(tables)



maritime_shipping_ais_dataset = MaritimeShippingAISDataset()
print(maritime_shipping_ais_dataset)
maritime_shipping_ais_full_db = maritime_shipping_ais_dataset.make_db()