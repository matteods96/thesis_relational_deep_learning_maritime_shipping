import duckdb
import requests
import zipfile
import os
import shutil
from datetime import date, timedelta

# -----------------------------
# CONFIGURATION
# -----------------------------
def aisdk_url(d: date) -> str:
    cutoff = date(2025, 2, 27)
    datestr = d.strftime("%Y-%m-%d")
    if d < cutoff:
        return f"http://aisdata.ais.dk/{d.year}/aisdk-{datestr}.zip"
    else:
        return f"http://aisdata.ais.dk/aisdk-{datestr}.zip"

start_date = date(2025, 1, 1)
end_date   = date(2025, 12, 31)

TMP_DIR = "tmp_csv"
DUCKDB_DIR = "duckdb_files"
PARQUET_DIR = "parquet_files"

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(DUCKDB_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

# -----------------------------
# BUILD ZIP URL LIST
# -----------------------------
zip_urls = []
d = start_date
while d <= end_date:
    zip_urls.append((d, aisdk_url(d)))
    d += timedelta(days=1)

# -----------------------------
# PROCESS EACH ZIP
# -----------------------------
for d, url in zip_urls:
    datestr = d.strftime("%Y-%m-%d")
    print(f"\nDownloading ZIP: {url}")

    zip_filename = os.path.basename(url)
    local_zip_path = os.path.join(TMP_DIR, zip_filename)

    # Downloading ZIP containing data
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_zip_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    print(f"  Saved ZIP to: {local_zip_path}")

    # Extracting  CSV file from zip file
    with zipfile.ZipFile(local_zip_path, "r") as zf:
        csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError(f"Expected 1 CSV in {zip_filename}, found: {csv_files}")

        csv_name = csv_files[0]
        local_csv_path = os.path.join(TMP_DIR, csv_name)
        zf.extract(csv_name, TMP_DIR)

    # Delete ZIP immediately
    os.remove(local_zip_path)

    # -----------------------------
    # 3. DuckDB + Parquet
    # -----------------------------
    db_file = os.path.join(DUCKDB_DIR, f"{datestr}_ais.duckdb")
    parquet_path = os.path.join(PARQUET_DIR, f"{datestr}_ais.parquet")

    con = duckdb.connect(db_file)

    print(f"  Creating DuckDB table + Parquet for {datestr}...")

    con.execute(f"""
        CREATE TABLE aisdk AS
        SELECT
            "# Timestamp" AS timestamp,
            "Type of mobile" AS type_of_mobile,
            TRY_CAST(MMSI AS BIGINT) AS mmsi,
            TRY_CAST(Latitude AS DOUBLE) AS lat,
            TRY_CAST(Longitude AS DOUBLE) AS lon,
            "Navigational status" AS status,
            TRY_CAST(ROT AS DOUBLE) AS turn,
            TRY_CAST(SOG AS DOUBLE) AS speed,
            TRY_CAST(COG AS DOUBLE) AS course,
            TRY_CAST(Heading AS DOUBLE) AS heading,
            IMO AS imo,
            Callsign AS callsign,
            Name AS shipname,
            "Ship type" AS shiptype,
            "Cargo type" AS cargotype,
            TRY_CAST(Width AS DOUBLE) AS width,
            TRY_CAST(Length AS DOUBLE) AS length,
            "Type of position fixing device" AS type_of_position_fixing,
            TRY_CAST(NULLIF(Draught, 'Undefined') AS DOUBLE) AS draught,
            Destination as destination,
            ETA AS eta,
            "Data source type" AS data_source_type,
            TRY_CAST(A AS DOUBLE) AS to_bow,
            TRY_CAST(B AS DOUBLE) AS to_stern,
            TRY_CAST(C AS DOUBLE) AS to_port,
            TRY_CAST(D AS DOUBLE) AS to_starboard
        FROM read_csv_auto(
            '{local_csv_path}',
            header = TRUE,
            all_varchar = TRUE,
            ignore_errors = TRUE,
            strict_mode = FALSE,
            null_padding = TRUE,
            parallel = FALSE
        );
    """)

    # Write Parquet
    con.execute(f"COPY aisdk TO '{parquet_path}' (FORMAT PARQUET);")
    con.close()

    # Delete CSV immediately after processing
    os.remove(local_csv_path)

    print(f"Finished {datestr} — DuckDB + Parquet stored locally.")

print("\nAll AIS files processed successfully.")
