import duckdb
import os

db_path = 'maritime_shipping_ais_7_to_20_april_new.db'
con = duckdb.connect(db_path)

PARQUET_GLOB = "processed_parquet_files/*.parquet"

period_start = "2025-04-04 00:00:00"
period_end   = "2025-04-21 00:00:00"

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
      >= strptime($start, '%Y-%m-%d %H:%M:%S')
  AND strptime(timestamp, '%d/%m/%Y %H:%M:%S')
      <  strptime($end,   '%Y-%m-%d %H:%M:%S');
""", {
    "glob": PARQUET_GLOB,
    "start": period_start,
    "end": period_end
})

print("Loaded AIS data between 4 April and 20 April 2025 into ais_raw")

print("Creating vessel table")
con.execute("""
CREATE TABLE IF NOT EXISTS vessel (
    mmsi INTEGER NOT NULL,
    imo INTEGER,
    callsign TEXT,
    shipname TEXT,
    shiptype TEXT
);
""")

print("Creating vessel_details table")
con.execute("""
CREATE TABLE IF NOT EXISTS vessel_details (
    vessel_details_id TEXT,
    mmsi INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    to_bow INTEGER,
    to_stern INTEGER,
    to_port INTEGER,
    to_starboard INTEGER
);
""")

print("Creating voyage table")
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

print("Creating position table")
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

print("Installing spatial library")
con.execute("INSTALL spatial;")
con.execute("LOAD spatial;")

print("Creating port table")
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

print("Inserting into vessel")
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
print("Inserted vessel")

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
print("Inserted vessel_details")

print("Computing median and mode ship dimensions per vessel")
con.execute("""
CREATE OR REPLACE TABLE vessel_geom_stats AS
WITH base AS (
    SELECT mmsi, to_bow, to_stern, to_port, to_starboard
    FROM vessel_details
),

med AS (
    SELECT
        mmsi,
        median(to_bow)       AS median_to_bow,
        median(to_stern)     AS median_to_stern,
        median(to_port)      AS median_to_port,
        median(to_starboard) AS median_to_starboard
    FROM base
    GROUP BY mmsi
),

mode_bow AS (
    SELECT mmsi, to_bow AS mode_to_bow
    FROM (
        SELECT mmsi, to_bow, COUNT(*) AS freq,
               ROW_NUMBER() OVER (PARTITION BY mmsi ORDER BY COUNT(*) DESC) AS rn
        FROM base WHERE to_bow IS NOT NULL
        GROUP BY mmsi, to_bow
    ) WHERE rn = 1
),

mode_stern AS (
    SELECT mmsi, to_stern AS mode_to_stern
    FROM (
        SELECT mmsi, to_stern, COUNT(*) AS freq,
               ROW_NUMBER() OVER (PARTITION BY mmsi ORDER BY COUNT(*) DESC) AS rn
        FROM base WHERE to_stern IS NOT NULL
        GROUP BY mmsi, to_stern
    ) WHERE rn = 1
),

mode_port AS (
    SELECT mmsi, to_port AS mode_to_port
    FROM (
        SELECT mmsi, to_port, COUNT(*) AS freq,
               ROW_NUMBER() OVER (PARTITION BY mmsi ORDER BY COUNT(*) DESC) AS rn
        FROM base WHERE to_port IS NOT NULL
        GROUP BY mmsi, to_port
    ) WHERE rn = 1
),

mode_starboard AS (
    SELECT mmsi, to_starboard AS mode_to_starboard
    FROM (
        SELECT mmsi, to_starboard, COUNT(*) AS freq,
               ROW_NUMBER() OVER (PARTITION BY mmsi ORDER BY COUNT(*) DESC) AS rn
        FROM base WHERE to_starboard IS NOT NULL
        GROUP BY mmsi, to_starboard
    ) WHERE rn = 1
)

SELECT
    med.mmsi,
    med.median_to_bow,
    med.median_to_stern,
    med.median_to_port,
    med.median_to_starboard,
    mode_bow.mode_to_bow,
    mode_stern.mode_to_stern,
    mode_port.mode_to_port,
    mode_starboard.mode_to_starboard
FROM med
LEFT JOIN mode_bow USING (mmsi)
LEFT JOIN mode_stern USING (mmsi)
LEFT JOIN mode_port USING (mmsi)
LEFT JOIN mode_starboard USING (mmsi);
""")
print("Computed vessel_geom_stats")

print("Adding geometry columns to vessel")
con.execute("""
ALTER TABLE vessel ADD COLUMN IF NOT EXISTS median_to_bow INTEGER;
ALTER TABLE vessel ADD COLUMN IF NOT EXISTS median_to_stern INTEGER;
ALTER TABLE vessel ADD COLUMN IF NOT EXISTS median_to_port INTEGER;
ALTER TABLE vessel ADD COLUMN IF NOT EXISTS median_to_starboard INTEGER;

ALTER TABLE vessel ADD COLUMN IF NOT EXISTS mode_to_bow INTEGER;
ALTER TABLE vessel ADD COLUMN IF NOT EXISTS mode_to_stern INTEGER;
ALTER TABLE vessel ADD COLUMN IF NOT EXISTS mode_to_port INTEGER;
ALTER TABLE vessel ADD COLUMN IF NOT EXISTS mode_to_starboard INTEGER;
""")

print("Updating vessel with geometry stats")
con.execute("""
UPDATE vessel v
SET
    median_to_bow       = s.median_to_bow,
    median_to_stern     = s.median_to_stern,
    median_to_port      = s.median_to_port,
    median_to_starboard = s.median_to_starboard,
    mode_to_bow         = s.mode_to_bow,
    mode_to_stern       = s.mode_to_stern,
    mode_to_port        = s.mode_to_port,
    mode_to_starboard   = s.mode_to_starboard
FROM vessel_geom_stats s
WHERE v.mmsi = s.mmsi;
""")
print("Updated vessel with geometry features")

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
print("Inserted position")

print("Inserting into voyage")
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
print("Inserted voyage")

print("Inserting into port")
con.execute("""
INSERT INTO port
SELECT *
FROM read_csv_auto('port_data.csv');
""")
print("Inserted port")

con.close()
print("Tables for maritime DB created")
