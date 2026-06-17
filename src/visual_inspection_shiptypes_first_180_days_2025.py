import os
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

db_path = "maritime_shipping_ais_half_year.db"
con = duckdb.connect(db_path)

#Get distinct shiptype classes
shiptypes = con.execute(
    """
    SELECT DISTINCT shiptype
    FROM vessel
    WHERE shiptype IS NOT NULL
    ORDER BY shiptype;
"""
).fetchdf()

shiptype_list = shiptypes["shiptype"].tolist()
print("Distinct shiptype classes:")
print(shiptype_list)

# Get min/max dates efficiently
date_bounds = con.execute(
    """
    SELECT MIN(timestamp), MAX(timestamp) FROM position
"""
).fetchone()
min_date, max_date = pd.to_datetime(date_bounds[0]), pd.to_datetime(
    date_bounds[1]
)

print(f"\nMinimum timestamp in AIS dataset: {min_date}")
print(f"Maximum timestamp in AIS dataset: {max_date}")

weekly_counts = con.execute(
    """
    SELECT 
        v.shiptype,
        strftime(p.timestamp, '%G-%V') AS year_week, 
        strftime(p.timestamp, '%V') AS week_number,
        COUNT(*) AS count
    FROM position p
    JOIN vessel v USING (mmsi)
    WHERE v.shiptype IS NOT NULL
    GROUP BY 1, 2, 3
    ORDER BY year_week, v.shiptype;
"""
).fetchdf()

# Pivot using 'year_week' to guarantee the timeline flows chronologically
pivot_df = (
    weekly_counts.pivot(index="year_week", columns="shiptype", values="count")
    .fillna(0)
    .sort_index()
)

# Map the chronological index back to just the isolated 'ww' strings for display
# This drops the 'YYYY-' prefix while keeping the exact chronological order intact
week_mapping = (
    weekly_counts[["year_week", "week_number"]]
    .drop_duplicates()
    .set_index("year_week")["week_number"]
    .to_dict()
)
display_weeks = [week_mapping[yw] for yw in pivot_df.index]

# Compute biggest 4 classes
total_counts = pivot_df.sum(axis=0).sort_values(ascending=False)
top4 = total_counts.head(4)
print("\nBiggest 4 shiptype classes (by total AIS observations):")
print(top4)

# Create output folder
output_folder = "ship type prediction/visual_ship_type_over_first_180_days_year_2025"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "weekly_shiptype_counts.png")

# Setup colormap safely
try:
    cmap = plt.colormaps["glasbey"]
except KeyError:
    cmap = plt.colormaps["tab20"]

colors = cmap(np.linspace(0, 1, len(pivot_df.columns)))

# Plot
plt.figure(figsize=(16, 8))

# We use an integer range for the x-axis to keep lines continuous and uninterrupted
for idx, shiptype in enumerate(pivot_df.columns):
    plt.plot(
        range(len(pivot_df)),
        pivot_df[shiptype],
        label=shiptype,
        color=colors[idx],
        linewidth=2,
    )

# Format X-axis ticks using ONLY the 'ww' format strings
plt.xticks(range(len(display_weeks)), display_weeks, rotation=45)

plt.xlabel("Week Number (ww)")
plt.ylabel("Number of AIS observations")
plt.title(
    f"Weekly AIS Observations Per Shiptype Class\n({min_date.date()} - {max_date.date()})"
)

plt.legend(
    title="Shiptype", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8
)

plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to: {output_path}")

plt.show()