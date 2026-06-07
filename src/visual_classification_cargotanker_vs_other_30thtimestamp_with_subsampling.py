import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import os
from datetime import timedelta

def print_class_proportion(name, data):
    total = data.shape[0]
    class1 = (data["shiptype"] == 1).sum()
    class0 = (data["shiptype"] == 0).sum()

    print(f"\n{name}")
    print(f"total: {total}")
    print(f"cargo/tanker: {class1} ({class1 / total:.2%})")
    print(f"other:        {class0} ({class0 / total:.2%})")


def load_nth_dimension_features(n):
    con = duckdb.connect("maritime_shipping_ais_half_year.db")
    period_start="2025-03-01"
    period_end = "2025-03-31"

    df = con.execute(f"""
        with vessel_geom as (
            select
                mmsi,
                to_bow,
                to_stern,
                to_port,
                to_starboard
            from vessel_details
        ),

        nth_ts as (
            select
                mmsi,
                nth_value(timestamp, {n}) over (
                    partition by mmsi order by timestamp
                ) as nth_timestamp
            from position
            where timestamp>= '{period_start}' AND timestamp < '{period_end}'
        ),

        distinct_nth as (
            select distinct mmsi, nth_timestamp
            from nth_ts
            where nth_timestamp is not null
        )

        select
            d.mmsi,
            v.shiptype,
            g.to_bow,
            g.to_stern,
            g.to_port,
            g.to_starboard,
            d.nth_timestamp
        from distinct_nth d
        join vessel v using (mmsi)
        join vessel_geom g using (mmsi)
    """).df()

    con.close()

    df = df.dropna()

    df["shiptype"] = df["shiptype"].apply(lambda x: 1 if x in ["Cargo", "Tanker"] else 0)
    df["nth_timestamp"] = pd.to_datetime(df["nth_timestamp"])

    print(f"shape of the full data in our experiments is {df.shape[0]/1_000_000} million rows")

    return df


def temporal_split(df, val_timestamp, test_timestamp):
    val_timestamp = pd.to_datetime(val_timestamp)
    test_timestamp = pd.to_datetime(test_timestamp)

    train = df[df["nth_timestamp"] < val_timestamp]
    val   = df[(df["nth_timestamp"] >= val_timestamp) &
               (df["nth_timestamp"] < test_timestamp)]
    test  = df[df["nth_timestamp"] >= test_timestamp]

    return train, val, test


def downsample_cargo_tanker(df, keep_fraction=0.4, random_state=42):
    df_major = df[df["shiptype"] == 1]
    df_minor = df[df["shiptype"] == 0]

    df_major_sub = df_major.sample(
        frac=keep_fraction,
        random_state=random_state
    )

    return pd.concat([df_major_sub, df_minor], axis=0).sample(frac=1, random_state=random_state)


df = load_nth_dimension_features(n=500)
print('earliest date', df["nth_timestamp"].min())
print('latest date', df["nth_timestamp"].max())

val_timestamp="2025-03-16"
test_timestamp="2025-03-24"

df_train, df_val, df_test = temporal_split(df, val_timestamp, test_timestamp)

print("training period:", df["nth_timestamp"].min().date(), "-", (pd.to_datetime(val_timestamp) - timedelta(days=1)).date())
print("validation period:", pd.to_datetime(val_timestamp).date(), "-", (pd.to_datetime(test_timestamp) - timedelta(days=1)).date())
print("test period:", pd.to_datetime(test_timestamp).date(), "-", df["nth_timestamp"].max().date())

print_class_proportion("train (original)", df_train)
print_class_proportion("validation (original)", df_val)
print_class_proportion("test (original)", df_test)

df_train = downsample_cargo_tanker(df_train, keep_fraction=0.4)
df_val   = downsample_cargo_tanker(df_val,   keep_fraction=0.4)
df_test  = downsample_cargo_tanker(df_test,  keep_fraction=0.4)

print_class_proportion("train (downsampled)", df_train)
print_class_proportion("validation (downsampled)", df_val)
print_class_proportion("test (downsampled)", df_test)

num_ships_train=len(set(df_train['mmsi']))
num_ships_val=len(set(df_val['mmsi']))
num_ships_test=len(set(df_test['mmsi']))

print(f'distinct ships for training: {num_ships_train}')
print(f'distinct ships for validation: {num_ships_val}')
print(f'distinct ships for testing: {num_ships_test}')


train_start = df_train["nth_timestamp"].min().date()
train_end   = df_train["nth_timestamp"].max().date()

val_start = df_val["nth_timestamp"].min().date()
val_end   = df_val["nth_timestamp"].max().date()

test_start = df_test["nth_timestamp"].min().date()
test_end   = df_test["nth_timestamp"].max().date()


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


color_map = {
    1: "darkorange",
    0: "steelblue"
}

length_bins = range(
    int((df["to_bow"] + df["to_stern"]).min()),
    int((df["to_bow"] + df["to_stern"]).max()) + 2
)

width_bins = range(
    int((df["to_port"] + df["to_starboard"]).min()),
    int((df["to_port"] + df["to_starboard"]).max()) + 2
)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("cargo/tanker vs other — length vs width", fontsize=16)

for ax, data, title in zip(
    axes,
    [geom_train, geom_val, geom_test],
    ["train", "validation", "test"]
):
    ax.scatter(
        data["length"], data["width"],
        c=data["shiptype"].map(color_map),
        s=10, alpha=0.6
    )

    if title == "train":
        ax.set_title(f"train {train_start} - {train_end} — n° obs: {len(data)}")
    elif title == "validation":
        ax.set_title(f"validation {val_start} - {val_end} — n° obs: {len(data)}")
    else:
        ax.set_title(f"test {test_start} - {test_end} — n° obs: {len(data)}")

    ax.set_xlabel("length")
    ax.set_ylabel("width")
    ax.grid(True)

axes[0].legend(handles=[
    plt.Line2D([0], [0], color="darkorange", lw=6, label="cargo/tanker"),
    plt.Line2D([0], [0], color="steelblue", lw=6, label="other")
])

output_dir = "ship type prediction/visual_plot_for_binary_classification(cargo_or_tanker_vs_other)_30th_timestamp"
os.makedirs(output_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "FIG1_scatter_binary.png"), dpi=300)
plt.close()


fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle("length and width distributions across dataset splits", fontsize=16)

datasets = [geom_train, geom_val, geom_test]
titles = ["train", "validation", "test"]

for i, (data, title) in enumerate(zip(datasets, titles)):
    if title == "train":
        axes2[0, i].set_title(f"train {train_start} - {train_end} length")
        axes2[1, i].set_title(f"train {train_start} - {train_end} width")
    elif title == "validation":
        axes2[0, i].set_title(f"validation {val_start} - {val_end} length")
        axes2[1, i].set_title(f"validation {val_start} - {val_end} width")
    else:
        axes2[0, i].set_title(f"test {test_start} - {test_end} length")
        axes2[1, i].set_title(f"test {test_start} - {test_end} width")

    axes2[0, i].hist(data["length"], bins=length_bins, color="darkorange", alpha=0.7)
    axes2[1, i].hist(data["width"], bins=width_bins, color="steelblue", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "FIG2_histograms_length_width.png"), dpi=300)
plt.close()


cargo_or_tanker_train = geom_train[geom_train["shiptype"] == 1]
cargo_or_tanker_val   = geom_val[geom_val["shiptype"] == 1]
cargo_or_tanker_test  = geom_test[geom_test["shiptype"] == 1]

other_train = geom_train[geom_train["shiptype"] == 0]
other_val   = geom_val[geom_val["shiptype"] == 0]
other_test  = geom_test[geom_test["shiptype"] == 0]

fig3, axes3 = plt.subplots(3, 4, figsize=(26, 18))
fig3.suptitle("cargo/tanker vs other — length & width distributions", fontsize=20)

splits = [
    (cargo_or_tanker_train, other_train, "train"),
    (cargo_or_tanker_val,   other_val,   "validation"),
    (cargo_or_tanker_test,  other_test,  "test")
]

for row, (c_or_t, oth, name) in enumerate(splits):

    if name == "train":
        tstart, tend = train_start, train_end
    elif name == "validation":
        tstart, tend = val_start, val_end
    else:
        tstart, tend = test_start, test_end

    axes3[row, 0].hist(c_or_t["length"], bins=40, color="darkorange", alpha=0.7)
    axes3[row, 0].set_title(f"{name} {tstart} - {tend} cargo/tanker length")

    axes3[row, 1].hist(oth["length"], bins=40, color="steelblue", alpha=0.7)
    axes3[row, 1].set_title(f"{name} {tstart} - {tend} other length")

    axes3[row, 2].hist(c_or_t["width"], bins=40, color="darkorange", alpha=0.7)
    axes3[row, 2].set_title(f"{name} {tstart} - {tend} cargo/tanker width")

    axes3[row, 3].hist(oth["width"], bins=40, color="steelblue", alpha=0.7)
    axes3[row, 3].set_title(f"{name} {tstart} - {tend} other width")

handles3 = [
    plt.Line2D([0], [0], color="darkorange", lw=6, label="cargo/tanker"),
    plt.Line2D([0], [0], color="steelblue", lw=6, label="other")
]

fig3.legend(handles=handles3, loc="lower center", ncol=2, fontsize=16, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig(os.path.join(output_dir, "FIG3_binary_length_width.png"), dpi=300)
plt.close()


plt.figure(figsize=(10, 8))
plt.title(
    f"cargo/tanker — length vs width\n"
    f"train {train_start} - {train_end}, "
    f"validation {val_start} - {val_end}, "
    f"test {test_start} - {test_end}"
)

plt.scatter(
    cargo_or_tanker_train["length"], cargo_or_tanker_train["width"],
    s=8, alpha=0.5, color="darkorange", label="train"
)

plt.scatter(
    cargo_or_tanker_val["length"], cargo_or_tanker_val["width"],
    s=8, alpha=0.5, color="green", label="validation"
)

plt.scatter(
    cargo_or_tanker_test["length"], cargo_or_tanker_test["width"],
    s=12, alpha=0.8, color="steelblue", label="test"
)

plt.xlabel("length")
plt.ylabel("width")
plt.grid(True)
plt.legend(loc="upper left")

plt.savefig(os.path.join(output_dir, "FIG4_cargo_or_tanker_only.png"), dpi=300)
plt.close()


plt.figure(figsize=(10, 8))
plt.title(
    f"other ships — length vs width\n"
    f"train {train_start} - {train_end}, "
    f"validation {val_start} - {val_end}, "
    f"test {test_start} - {test_end}"
)

plt.scatter(
    other_train["length"], other_train["width"],
    s=8, alpha=0.5, color="steelblue", label="train"
)

plt.scatter(
    other_val["length"], other_val["width"],
    s=8, alpha=0.5, color="green", label="validation"
)

plt.scatter(
    other_test["length"], other_test["width"],
    s=12, alpha=0.8, color="darkorange", label="test"
)

plt.xlabel("length")
plt.ylabel("width")
plt.grid(True)
plt.legend(loc="upper left")

plt.savefig(os.path.join(output_dir, "FIG5_other_only.png"), dpi=300)
plt.close()
