import numpy as np
import pandas as pd
import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


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


def temporal_split(df, val_timestamp, test_timestamp):
    val_timestamp = pd.to_datetime(val_timestamp)
    test_timestamp = pd.to_datetime(test_timestamp)

    train = df[df["nth_timestamp"] < val_timestamp]
    val   = df[(df["nth_timestamp"] >= val_timestamp) &
               (df["nth_timestamp"] < test_timestamp)]
    test  = df[df["nth_timestamp"] >= test_timestamp]

    return train, val, test


def extract_features(df):
    X = df[["to_bow", "to_stern", "to_port", "to_starboard"]].astype(np.float32).values
    y = df["shiptype"].astype(np.float32).values
    return X, y


def scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


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
    period_start = "2025-04-07"
    period_end   = "2025-04-21"

    val_timestamp  = "2025-04-14"
    test_timestamp = "2025-04-18"

    df = load_nth_geometry_features(n, period_start, period_end)

    train_df, val_df, test_df = temporal_split(df, val_timestamp, test_timestamp)

    # NO DOWNSAMPLING ANYWHERE

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
