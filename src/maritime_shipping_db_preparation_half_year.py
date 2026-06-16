import duckdb
import os

db_path='maritime_shipping_ais_half_year.db'
con=duckdb.connect(db_path)

PARQUET_GLOB = "processed_parquet_files/*.parquet"

# ------------------------------------------------------------
# Load AIS RAW
# ------------------------------------------------------------
con.execute("""
            
CREATE TABLE IF NOT EXISTS ais_raw AS
SELECT
    mmsi,
    strptime(timestamp, '%d/%m/%Y %H:%M:%S') AS timestamp,
    lat,
    lon,
    speed,
    course,
    heading,
    turn,
    status,
    draught,
    destination,
    strptime(eta, '%d/%m/%Y %H:%M:%S') AS eta,
    TRY_CAST(imo AS INTEGER) AS imo,
    callsign,
    shipname,
    shiptype,
    TRY_CAST(to_bow AS INTEGER) AS to_bow,
    TRY_CAST(to_stern AS INTEGER) AS to_stern,
    TRY_CAST(to_port AS INTEGER) AS to_port,
    TRY_CAST(to_starboard AS INTEGER) AS to_starboard
FROM parquet_scan($glob)
WHERE strptime(timestamp, '%d/%m/%Y %H:%M:%S')
      < (SELECT MIN(strptime(timestamp, '%d/%m/%Y %H:%M:%S'))
         FROM parquet_scan($glob)) + INTERVAL 180 DAY;
""", {"glob": PARQUET_GLOB})

print("Loaded first 180 days of AIS data into ais_raw")


#Creating empty tables with defined schema by the user
#print("Creating vessel table ")


print("Creating vessel table with static info)") 
con.execute(""" CREATE TABLE IF NOT EXISTS vessel (
    mmsi INTEGER NOT NULL, 
    imo INTEGER,
    callsign TEXT,
    shipname TEXT,
    shiptype TEXT ); 
""")


print("Creating vessel_details table (dynamic offsets only)")
con.execute(""" CREATE TABLE IF NOT EXISTS vessel_details ( 
    vessel_details_id TEXT,
    mmsi INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    to_bow INTEGER,
    to_stern INTEGER,
    to_port INTEGER,
    to_starboard INTEGER);
 """)

print("Creating voyage table ")
con.execute("""
CREATE TABLE IF NOT EXISTS voyage (
    voyage_id TEXT,
    mmsi INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    draught INTEGER NOT NULL,
    destination TEXT,
    month INTEGER,
    day INTEGER,
    hour INTEGER,
    minute INTEGER
);
""")
print("Creating position table ")

con.execute("""
    CREATE TABLE IF NOT EXISTS position (
    position_id TEXT,
    mmsi INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    status TEXT,
    turn INTEGER,
    speed INTEGER,
    lon INTEGER,
    lat INTEGER,
    course INTEGER,
    heading INTEGER
);
""")
print('Installing spatial library')
con.execute("INSTALL spatial;")
con.execute("LOAD spatial;")
print("Inserting table for port")

print("Creating port table ")
con.execute("""
CREATE TABLE IF NOT EXISTS port (
    port_id INTEGER,
    port_code TEXT,
    port_name TEXT,
    country_code TEXT,
    country_name TEXT,
    polygon TEXT
);
""")




print("Inserting table for vessel")
con.execute("""
INSERT INTO vessel
SELECT
    mmsi,
    imo,
    callsign,
    shipname,
    shiptype
FROM (
    SELECT
        mmsi,
        imo,
        callsign,
        shipname,
        shiptype,
        ROW_NUMBER() OVER (
            PARTITION BY mmsi
            ORDER BY timestamp ASC
        ) AS rn
    FROM ais_raw
    WHERE mmsi IS NOT NULL
) t
WHERE rn = 1;
""")

print("Inserted table for vessel")

print("Inserting into vessel_details")
con.execute("""
INSERT INTO vessel_details
SELECT
    mmsi || '_' || strftime(timestamp, '%Y%m%d%H%M%S') AS vessel_details_id,
    mmsi,
    timestamp,
    to_bow,
    to_stern,
    to_port,
    to_starboard
FROM (
    SELECT
        mmsi,
        timestamp,
        to_bow,
        to_stern,
        to_port,
        to_starboard,
        ROW_NUMBER() OVER (
            PARTITION BY mmsi, timestamp
            ORDER BY timestamp
        ) AS rn
    FROM ais_raw
    WHERE mmsi IS NOT NULL
      AND (to_bow IS NOT NULL OR to_stern IS NOT NULL OR
           to_port IS NOT NULL OR to_starboard IS NOT NULL)
)
WHERE rn = 1
ORDER BY mmsi, timestamp;
""")
print("Inserted table for vessel_details")





print("Inserting table for position")
print("Inserting into position")
con.execute("""
INSERT INTO position
SELECT
    mmsi || '_' || strftime(timestamp, '%Y%m%d%H%M%S') AS position_id,
    mmsi,
    timestamp,
    status,
    turn,
    speed,
    lon,
    lat,
    course,
    heading
FROM (
    SELECT
        mmsi,
        timestamp,
        status,
        turn,
        speed,
        lon,
        lat,
        course,
        heading,
        ROW_NUMBER() OVER (
            PARTITION BY mmsi, timestamp
            ORDER BY timestamp
        ) AS rn
    FROM ais_raw
    WHERE mmsi IS NOT NULL
)
WHERE rn = 1
ORDER BY mmsi, timestamp;
""")
print("Inserted table for position")


print("Inserting table for voyage")
con.execute("""
INSERT INTO voyage
SELECT
    mmsi || '_' || strftime(timestamp, '%Y%m%d%H%M%S') AS voyage_id,
    mmsi,
    timestamp,
    draught,
    destination,
    EXTRACT(month FROM eta),
    EXTRACT(day FROM eta),
    EXTRACT(hour FROM eta),
    EXTRACT(minute FROM eta)
FROM (
    SELECT
        mmsi,
        timestamp,
        draught,
        destination,
        eta,
        ROW_NUMBER() OVER (
            PARTITION BY mmsi, timestamp
            ORDER BY timestamp
        ) AS rn
    FROM ais_raw
    WHERE mmsi IS NOT NULL
      AND draught IS NOT NULL
)
WHERE rn = 1
ORDER BY mmsi, timestamp;
""")





print("Inserting  table for port")

con.execute("""
INSERT INTO port 
SELECT
    port_id,
    port_code,
    port_name,
    country_code,
    country_name,
    polygon
FROM  read_csv_auto('port_data.csv');
""")

print("Inserted  table for port")







con.close()

print('Tables for maritime db created')