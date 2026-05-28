import numpy as np
import pandas as pd
import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def load_nth_position_features(n, val_timestamp, test_timestamp):
    con = duckdb.connect("maritime_shipping_ais.db")

    period_end = "2025-03-31"  # first 90 days

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

    -- Compute nth timestamp per vessel using nth_value()
    nth_ts AS (
        SELECT
            mmsi,
            nth_value(timestamp, {n}) OVER (
                PARTITION BY mmsi ORDER BY timestamp
            ) AS nth_timestamp
        FROM position
        WHERE timestamp < '{period_end}'
    ),

    -- Deduplicate (nth_value repeats per row)
    distinct_nth AS (
        SELECT DISTINCT mmsi, nth_timestamp
        FROM nth_ts
        WHERE nth_timestamp IS NOT NULL
    ),

    final AS (
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
    )

    SELECT *
    FROM final
    """).df()

    con.close()

    df = df.dropna()
    df["shiptype"] = (df["shiptype"] == "Cargo").astype(int)
    df["nth_timestamp"] = pd.to_datetime(df["nth_timestamp"])

    return df


def temporal_split(df, val_timestamp, test_timestamp):
    train = df[df["nth_timestamp"] < val_timestamp]
    val   = df[(df["nth_timestamp"] >= val_timestamp) &
               (df["nth_timestamp"] < test_timestamp)]
    test  = df[df["nth_timestamp"] >= test_timestamp]
    return train, val, test


def extract_features(df):
    X = df[["to_bow", "to_stern", "to_port", "to_starboard"]].values.astype(np.float32)
    y = df["shiptype"].values.astype(np.float32)
    return X, y


def scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
    )


def apply_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    return {
        "log_loss": log_loss(y, prob),
        "roc_auc": roc_auc_score(y, prob),
    }


def main():
    n = 500
    val_timestamp  = pd.Timestamp("2025-01-25")
    test_timestamp = pd.Timestamp("2025-02-14")

    df = load_nth_position_features(n, val_timestamp, test_timestamp)

    train_df, val_df, test_df = temporal_split(df, val_timestamp, test_timestamp)

    X_train, y_train = extract_features(train_df)
    X_val, y_val     = extract_features(val_df)
    X_test, y_test   = extract_features(test_df)

    X_train_s, X_val_s, X_test_s = scale(X_train, X_val, X_test)

    model = apply_logistic_regression(X_train_s, y_train)

    print("Train:", evaluate(model, X_train_s, y_train))
    print("Val:",   evaluate(model, X_val_s, y_val))
    print("Test:",  evaluate(model, X_test_s, y_test))


if __name__ == "__main__":
    main()
