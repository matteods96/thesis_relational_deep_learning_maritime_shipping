import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import os

def print_class_proportion(name, data):
    total = data.shape[0]
    class1 = (data["shiptype"] == 1).sum()
    class0 = (data["shiptype"] == 0).sum()

    print(f"\n{name}")
    print(f"total: {total}")
    print(f"cargo/tanker: {class1} ({class1 / total:.2%})")
    print(f"other:        {class0} ({class0 / total:.2%})")

# LOAD DATA

def load_nth_geometry_features(n, period_start, period_end):
    con = duckdb.connect("maritime_shipping_ais_half_year.db")

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
            WHERE timestamp >= '{period_start}'
              AND timestamp <  '{period_end}'
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
    df["shiptype"] = df["shiptype"].apply(lambda x: 1 if x in ["Cargo", "Tanker"] else 0)
    df["nth_timestamp"] = pd.to_datetime(df["nth_timestamp"])

    return df



# TEMPORAL SPLIT

def temporal_split(df, val_timestamp, test_timestamp):
    val_timestamp = pd.to_datetime(val_timestamp)
    test_timestamp = pd.to_datetime(test_timestamp)

    train = df[df["nth_timestamp"] < val_timestamp]
    val   = df[(df["nth_timestamp"] >= val_timestamp) &
               (df["nth_timestamp"] < test_timestamp)]
    test  = df[df["nth_timestamp"] >= test_timestamp]

    return train, val, test



# VISUALIZATION

def visualize_length_width(train_df, val_df, test_df, output_dir="ship type prediction/vis_log_regr_cargotanker_vs_other500"):

    os.makedirs(output_dir, exist_ok=True)

    # compute length and width
    for df in (train_df, val_df, test_df):
        df["length"] = df["to_bow"] + df["to_stern"]
        df["width"]  = df["to_port"] + df["to_starboard"]

    # ---------------- FIG 1: SCATTER ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Length vs Width — Train / Validation / Test", fontsize=16)

    for ax, data, title in zip(
        axes,
        [train_df, val_df, test_df],
        ["train", "validation", "test"]
    ):
        ax.scatter(
            data["length"], data["width"],
            c=data["shiptype"].map({1: "darkorange", 0: "steelblue"}),
            s=10, alpha=0.6
        )
        ax.set_title(f"{title} — n={len(data)}")
        ax.set_xlabel("length")
        ax.set_ylabel("width")
        ax.grid(True)

    axes[0].legend(handles=[
        plt.Line2D([0], [0], color="darkorange", lw=6, label="cargo/tanker"),
        plt.Line2D([0], [0], color="steelblue", lw=6, label="other")
    ])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG1_scatter_binary.png"), dpi=300)
    plt.close()

    # ---------------- FIG 2: HISTOGRAMS ----------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("Length & Width Distributions — Train / Val / Test", fontsize=16)

    datasets = [train_df, val_df, test_df]
    titles = ["train", "validation", "test"]

    for i, (data, title) in enumerate(zip(datasets, titles)):
        axes2[0, i].hist(data["length"], bins=40, color="darkorange", alpha=0.7)
        axes2[0, i].set_title(f"{title} — length")

        axes2[1, i].hist(data["width"], bins=40, color="steelblue", alpha=0.7)
        axes2[1, i].set_title(f"{title} — width")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG2_histograms_length_width.png"), dpi=300)
    plt.close()

    # ---------------- FIG 3: CLASS-CONDITIONED HISTOGRAMS ----------------
    fig3, axes3 = plt.subplots(3, 4, figsize=(26, 18))
    fig3.suptitle("Cargo/Tanker vs Other — Length & Width Distributions", fontsize=20)

    splits = [
        (train_df, "train"),
        (val_df,   "validation"),
        (test_df,  "test")
    ]

    for row, (df, name) in enumerate(splits):
        c_or_t = df[df["shiptype"] == 1]
        oth    = df[df["shiptype"] == 0]

        axes3[row, 0].hist(c_or_t["length"], bins=40, color="darkorange", alpha=0.7)
        axes3[row, 0].set_title(f"{name} cargo/tanker length")

        axes3[row, 1].hist(oth["length"], bins=40, color="steelblue", alpha=0.7)
        axes3[row, 1].set_title(f"{name} other length")

        axes3[row, 2].hist(c_or_t["width"], bins=40, color="darkorange", alpha=0.7)
        axes3[row, 2].set_title(f"{name} cargo/tanker width")

        axes3[row, 3].hist(oth["width"], bins=40, color="steelblue", alpha=0.7)
        axes3[row, 3].set_title(f"{name} other width")

    fig3.legend(handles=[
        plt.Line2D([0], [0], color="darkorange", lw=6, label="cargo/tanker"),
        plt.Line2D([0], [0], color="steelblue", lw=6, label="other")
    ], loc="lower center", ncol=2, fontsize=16, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "FIG3_binary_length_width.png"), dpi=300)
    plt.close()

    # ---------------- FIG 4: CARGO/TANKER ONLY ----------------
    plt.figure(figsize=(10, 8))
    plt.title("Cargo/Tanker — Length vs Width")

    plt.scatter(train_df[train_df["shiptype"] == 1]["length"],
                train_df[train_df["shiptype"] == 1]["width"],
                s=8, alpha=0.5, color="darkorange", label="train")

    plt.scatter(val_df[val_df["shiptype"] == 1]["length"],
                val_df[val_df["shiptype"] == 1]["width"],
                s=8, alpha=0.5, color="green", label="validation")

    plt.scatter(test_df[test_df["shiptype"] == 1]["length"],
                test_df[test_df["shiptype"] == 1]["width"],
                s=12, alpha=0.8, color="steelblue", label="test")

    plt.xlabel("length")
    plt.ylabel("width")
    plt.grid(True)
    plt.legend(loc="upper left")

    plt.savefig(os.path.join(output_dir, "FIG4_cargo_or_tanker_only.png"), dpi=300)
    plt.close()

    # ---------------- FIG 5: OTHER ONLY ----------------
    plt.figure(figsize=(10, 8))
    plt.title("Other Ships — Length vs Width")

    plt.scatter(train_df[train_df["shiptype"] == 0]["length"],
                train_df[train_df["shiptype"] == 0]["width"],
                s=8, alpha=0.5, color="steelblue", label="train")

    plt.scatter(val_df[val_df["shiptype"] == 0]["length"],
                val_df[val_df["shiptype"] == 0]["width"],
                s=8, alpha=0.5, color="green", label="validation")

    plt.scatter(test_df[test_df["shiptype"] == 0]["length"],
                test_df[test_df["shiptype"] == 0]["width"],
                s=12, alpha=0.8, color="darkorange", label="test")

    plt.xlabel("length")
    plt.ylabel("width")
    plt.grid(True)
    plt.legend(loc="upper left")

    plt.savefig(os.path.join(output_dir, "FIG5_other_only.png"), dpi=300)
    plt.close()



# MAIN

def main():
    n = 500
    period_start = "2025-04-07"
    period_end   = "2025-04-21"

    val_timestamp  = "2025-04-14"
    test_timestamp = "2025-04-18"

    df = load_nth_geometry_features(n, period_start, period_end)

    train_df, val_df, test_df = temporal_split(df, val_timestamp, test_timestamp)

    print_class_proportion("TRAIN", train_df)
    print_class_proportion("VALIDATION", val_df)
    print_class_proportion("TEST", test_df)


    visualize_length_width(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
