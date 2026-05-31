import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import os
from datetime import timedelta


def load_nth_dimension_features(n):
    con = duckdb.connect("maritime_shipping_ais.db")

    period_end = "2025-03-31"

    df = con.execute(f"""
        WITH vessel_geom AS (
            SELECT
                mmsi,
                to_bow,
                to_stern,
                to_port,
                to_starboard
            FROM vessel_details
        ),

        nth_ts AS (
            SELECT
                mmsi,
                nth_value(timestamp, {n}) OVER (
                    PARTITION BY mmsi ORDER BY timestamp
                ) AS nth_timestamp
            FROM position
            WHERE timestamp < '{period_end}'
        ),

        distinct_nth AS (
            SELECT DISTINCT mmsi, nth_timestamp
            FROM nth_ts
            WHERE nth_timestamp IS NOT NULL
        )

        SELECT
            d.mmsi,
            v.shiptype,
            g.to_bow,
            g.to_stern,
            g.to_port,
            g.to_starboard,
            d.nth_timestamp
        FROM distinct_nth d
        JOIN vessel v USING (mmsi)
        JOIN vessel_geom g USING (mmsi)
    """).df()

    con.close()

    df = df.dropna()

    # keep only Cargo and Tanker
   # df = df[df["shiptype"].isin(["Cargo", "Tanker"])] no need as the db only contains Cargo and Tankers

    # encode: Cargo = 1, Tanker = 0
    df["shiptype"] = (df["shiptype"] == "Cargo").astype(int)

    df["nth_timestamp"] = pd.to_datetime(df["nth_timestamp"])
    print(f"Shape of the full data in our experiments is {df.shape[0]/1_000_000} million rows")

    return df


def temporal_split(df, val_timestamp, test_timestamp):
    val_timestamp = pd.to_datetime(val_timestamp)
    test_timestamp = pd.to_datetime(test_timestamp)

    train = df[df["nth_timestamp"] < val_timestamp]
    val   = df[(df["nth_timestamp"] >= val_timestamp) &
               (df["nth_timestamp"] < test_timestamp)]
    test  = df[df["nth_timestamp"] >= test_timestamp]

    return train, val, test


df = load_nth_dimension_features(n=500)
print('Earliest date',df["nth_timestamp"].min())
print('Latest date',df["nth_timestamp"].max())

val_timestamp="2025-01-25"
test_timestamp="2025-02-14"


df_train, df_val, df_test = temporal_split(
    df,
    val_timestamp,
    test_timestamp
)

print(
    "Training period:    ",
    df["nth_timestamp"].min().date(),
    " - ",
    (pd.to_datetime(val_timestamp) - timedelta(days=1)).date()
)

print(
    "Validation period:  ",
    pd.to_datetime(val_timestamp).date(),
    " - ",
    (pd.to_datetime(test_timestamp) - timedelta(days=1)).date()
)

print(
    "Test period:        ",
    pd.to_datetime(test_timestamp).date(),
    " - ",
    df["nth_timestamp"].max().date()
)



num_ships_train=len(set(df_train['mmsi']))
num_ships_val=len(set(df_val['mmsi']))
num_ships_test=len(set(df_test['mmsi']))

print(f'Distinct ships for training: {num_ships_train}')
print(f'Distinct shipsfor validation: {num_ships_val}')
print(f'Distinct ships type for testing: {num_ships_test}')


geom_train = pd.DataFrame({
    "length": df_train["to_bow"] + df_train["to_stern"],
    "width":  df_train["to_port"] + df_train["to_starboard"],
    "shiptype": df_train["shiptype"]
})

geom_val = pd.DataFrame({
    "length": df_val["to_bow"] + df_val["to_stern"],
    "width":  df_val["to_port"] + df_val["to_starboard"],
    "shiptype": df_val["shiptype"]
})

geom_test = pd.DataFrame({
    "length": df_test["to_bow"] + df_test["to_stern"],
    "width":  df_test["to_port"] + df_test["to_starboard"],
    "shiptype": df_test["shiptype"]
})


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Cargo vs Tanker — Length vs Width", fontsize=16)

color_map = {
    0: "steelblue",   # Tanker
    1: "darkorange"   # Cargo
}

# TRAIN
axes[0].scatter(
    geom_train["length"],
    geom_train["width"],
    c=geom_train["shiptype"].map(color_map),
    s=10,
    alpha=0.6
)
axes[0].set_title(f"Train — n° observations: {len(geom_train)}")
axes[0].set_xlabel("length")
axes[0].set_ylabel("width")
axes[0].grid(True)
axes[0].scatter([], [], c="darkorange", label="Cargo")
axes[0].scatter([], [], c="steelblue", label="Tanker")
axes[0].legend(loc="upper right")

# VALIDATION
axes[1].scatter(
    geom_val["length"],
    geom_val["width"],
    c=geom_val["shiptype"].map(color_map),
    s=10,
    alpha=0.6
)
axes[1].set_title(f"Validation — n° observations: {len(geom_val)}")
axes[1].set_xlabel("length")
axes[1].set_ylabel("width")
axes[1].grid(True)
axes[1].scatter([], [], c="darkorange", label="Cargo")
axes[1].scatter([], [], c="steelblue", label="Tanker")
axes[1].legend(loc="upper right")

# TEST
axes[2].scatter(
    geom_test["length"],
    geom_test["width"],
    c=geom_test["shiptype"].map(color_map),
    s=10,
    alpha=0.6
)
axes[2].set_title(f"Test — n° observations: {len(geom_test)}")
axes[2].set_xlabel("length")
axes[2].set_ylabel("width")
axes[2].grid(True)
axes[2].scatter([], [], c="darkorange", label="Cargo")
axes[2].scatter([], [], c="steelblue", label="Tanker")
axes[2].legend(loc="upper right")

plt.tight_layout()

output_dir = "ship type prediction/visual_plot_for_linear_separability"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(
    output_dir,
    "linear_separability_between_length_and_width.png"
)

plt.savefig(save_path, dpi=300)
plt.close()


#Creating Histograms length and width 

# integer‑aligned bins for discrete ship dimensions lenghts and width
length_bins = range(
    int(df["to_bow"].min() + df["to_stern"].min()),
    int(df["to_bow"].max() + df["to_stern"].max()) + 2
)

width_bins = range(
    int(df["to_port"].min() + df["to_starboard"].min()),
    int(df["to_port"].max() + df["to_starboard"].max()) + 2
)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle("Histograms of Length and Width — Cargo vs Tanker", fontsize=16)

#  histograms for ship's length
axes2[0, 0].hist(geom_train["length"], bins=length_bins, color="darkorange", alpha=0.7)
axes2[0, 0].set_title(f"Train Length (n={len(geom_train)})")
axes2[0, 0].set_xlabel("length")
axes2[0, 0].set_ylabel("count")

axes2[0, 1].hist(geom_val["length"], bins=length_bins, color="darkorange", alpha=0.7)
axes2[0, 1].set_title(f"Validation Length (n={len(geom_val)})")
axes2[0, 1].set_xlabel("length")
axes2[0, 1].set_ylabel("count")

axes2[0, 2].hist(geom_test["length"], bins=length_bins, color="darkorange", alpha=0.7)
axes2[0, 2].set_title(f"Test Length (n={len(geom_test)})")
axes2[0, 2].set_xlabel("length")
axes2[0, 2].set_ylabel("count")

# histograms for ships' width
axes2[1, 0].hist(geom_train["width"], bins=width_bins, color="steelblue", alpha=0.7)
axes2[1, 0].set_title(f"Train Width (n={len(geom_train)})")
axes2[1, 0].set_xlabel("width")
axes2[1, 0].set_ylabel("count")

axes2[1, 1].hist(geom_val["width"], bins=width_bins, color="steelblue", alpha=0.7)
axes2[1, 1].set_title(f"Validation Width (n={len(geom_val)})")
axes2[1, 1].set_xlabel("width")
axes2[1, 1].set_ylabel("count")

axes2[1, 2].hist(geom_test["width"], bins=width_bins, color="steelblue", alpha=0.7)
axes2[1, 2].set_title(f"Test Width (n={len(geom_test)})")
axes2[1, 2].set_xlabel("width")
axes2[1, 2].set_ylabel("count")

plt.tight_layout()

save_path_hist = os.path.join(
    output_dir,
    "histograms_length_width.png"
)
plt.savefig(save_path_hist, dpi=300)
plt.close()


#Creating  a single plot that only contains the data for cargo ships, but containing all the training, validation and test data for the Cargo ships.
#  Using different colors for the three datasets.

cargo_train = geom_train[geom_train["shiptype"] == 1]
cargo_val   = geom_val[geom_val["shiptype"] == 1]
cargo_test  = geom_test[geom_test["shiptype"] == 1]

plt.figure(figsize=(10, 8))
plt.title("Cargo Ships — Length vs Width ( Train / Val / Test)", fontsize=16)

plt.scatter(
    cargo_train["length"], cargo_train["width"],
    s=8, alpha=0.5, color="darkorange", label="Train"
)

plt.scatter(
    cargo_val["length"], cargo_val["width"],
    s=8, alpha=0.5, color="green", label="Validation"
)

plt.scatter(
    cargo_test["length"], cargo_test["width"],
    s=12, alpha=0.8, color="steelblue", label="Test"
)

plt.xlabel("length")
plt.ylabel("width")
plt.grid(True)
plt.legend(loc="upper left")

save_path_single = os.path.join(
    output_dir,
    "cargo_only_train_val_test.png"
)
plt.savefig(save_path_single, dpi=300)
plt.close()

