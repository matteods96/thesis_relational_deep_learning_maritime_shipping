import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

db_path = "maritime_shipping_ais_half_year.db"
con = duckdb.connect(db_path)

period_start = "2025-01-01"
period_end   = "2025-06-30"
step = 500

# filter every 30th position per vessel in db
df = con.execute(f"""
    WITH ranked AS (
        SELECT
            mmsi,
            timestamp,
            ROW_NUMBER() OVER (
                PARTITION BY mmsi ORDER BY timestamp
            ) AS rn
        FROM position
        WHERE timestamp >= '{period_start}'
          AND timestamp <  '{period_end}'
    ),
    every_500 AS (
        SELECT
            mmsi,
            timestamp AS nth_timestamp
        FROM ranked
        WHERE rn % {step} = 0
    )
    SELECT
        e.mmsi,
        e.nth_timestamp,
        v.shiptype
    FROM every_500 e
    JOIN vessel v USING (mmsi)
    ORDER BY e.mmsi, e.nth_timestamp;
""").fetchdf()

df["nth_timestamp"] = pd.to_datetime(df["nth_timestamp"])

min_date = df["nth_timestamp"].min()
max_date = df["nth_timestamp"].max()

print(f"minimum nth-500 timestamp: {min_date}")
print(f"maximum nth-500 timestamp: {max_date}")

# weekly aggregation based on nth_timestamp
df["week"] = df["nth_timestamp"].dt.isocalendar().week.astype(int)

weekly_counts = (
    df.groupby(["shiptype", "week"])
      .size()
      .reset_index(name="count")
)

pivot_df = weekly_counts.pivot(index="week", columns="shiptype", values="count").fillna(0)

# top 2 classes
total_counts = pivot_df.sum(axis=0).sort_values(ascending=False)
top4 = total_counts.head(4)

print("\nbiggest 4 shiptype classes (every 30th position):")
print(top4)

# create output folder
output_folder = "ship type prediction/visual_ship_type_over_first_180_days_year_2025_500thpos"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "weekly_shiptype_counts_500th.png")

# colormap with fallback
try:
    cmap = plt.colormaps["glasbey"]
except KeyError:
    cmap = plt.colormaps["tab20"]

colors = cmap(np.linspace(0, 1, len(pivot_df.columns)))

# plot with real week numbers on x-axis
plt.figure(figsize=(16, 8))

for idx, shiptype in enumerate(pivot_df.columns):
    plt.plot(
        pivot_df.index,
        pivot_df[shiptype],
        label=shiptype,
        color=colors[idx],
        linewidth=2,
    )

weeks = pivot_df.index.tolist()
plt.xticks(weeks, weeks, rotation=45)

plt.xlabel("week number (ww)")
plt.ylabel("number of ais observations (every 500th position)")
plt.title(
    f"weekly ais observations per shiptype class (every 500th position)\n({min_date.date()} - {max_date.date()})"
)

plt.legend(
    title="shiptype", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8
)

plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nplot saved to: {output_path}")

plt.show()
