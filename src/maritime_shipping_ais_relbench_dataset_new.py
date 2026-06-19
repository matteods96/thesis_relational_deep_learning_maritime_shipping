import duckdb
import pandas as pd
from relbench.base import Database, Dataset, Table


class MaritimeShippingAISDatasetNew(Dataset):

    # Choose timestamps for RelBench splits
    val_timestamp = pd.Timestamp("2025-04-14")
    test_timestamp = pd.Timestamp("2025-04-18")

    def make_db(self) -> Database:

        #DB_PATH = "maritime_shipping_ais_half_year_new.db"
        DB_PATH ='maritime_shipping_ais_7_to_20_april_new.db'
        con = duckdb.connect(DB_PATH)

       
        # Load tables from DuckDB
       

        vessels = con.execute("""
            SELECT
                mmsi,
                imo,
                callsign,
                shipname,
                CASE
                    WHEN shiptype IN ('Cargo', 'Tanker') THEN 'Cargo/Tanker'
                    ELSE 'Other'
                END AS shiptype,
                median_to_bow,
                median_to_stern,
                median_to_port,
                median_to_starboard,
                mode_to_bow,
                mode_to_stern,
                mode_to_port,
                mode_to_starboard
            FROM vessel
        """).df().drop_duplicates(subset=["mmsi"])

        vessel_details = con.execute("""
            SELECT *
            FROM vessel_details
        """).df().drop_duplicates(subset=["vessel_details_id"])
        vessel_details["timestamp"] = vessel_details["timestamp"].astype("datetime64[s]")

        positions = con.execute("""
            SELECT *
            FROM position
        """).df().drop_duplicates(subset=["position_id"])
        positions["timestamp"] = positions["timestamp"].astype("datetime64[s]")

        voyages = con.execute("""
            SELECT *
            FROM voyage
        """).df().drop_duplicates(subset=["voyage_id"])
        voyages["timestamp"] = voyages["timestamp"].astype("datetime64[s]")

        ports = con.execute("""
            SELECT *
            FROM port
        """).df()

        con.close()

       
        # Build RelBench tables
       

        tables = {}

        tables["vessels"] = Table(
            df=vessels,
            fkey_col_to_pkey_table={},
            pkey_col="mmsi",
            time_col=None,
        )

        tables["vessel_details"] = Table(
            df=vessel_details,
            fkey_col_to_pkey_table={"mmsi": "vessels"},
            pkey_col="vessel_details_id",
            time_col="timestamp",
        )

        tables["positions"] = Table(
            df=positions,
            fkey_col_to_pkey_table={"mmsi": "vessels"},
            pkey_col="position_id",
            time_col="timestamp",
        )

        tables["voyages"] = Table(
            df=voyages,
            fkey_col_to_pkey_table={"mmsi": "vessels", "destination": "ports"},
            pkey_col="voyage_id",
            time_col="timestamp",
        )

        tables["ports"] = Table(
            df=ports,
            fkey_col_to_pkey_table={},
            pkey_col="port_code",
            time_col=None,
        )

        return Database(tables)


# Instantiate dataset
maritime_dataset = MaritimeShippingAISDatasetNew()
db = maritime_dataset.make_db()
print("Dataset loaded:", maritime_dataset)
