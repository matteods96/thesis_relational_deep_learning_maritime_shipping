import numpy
import pandas
import duckdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def load_nth_position_features(n, val_timestamp, test_timestamp):
    con = duckdb.connect("maritime_shipping_ais.db")

    df = con.execute(f"""
    WITH ordered AS (
        SELECT
            mmsi,
            shiptype,
            speed,
            course,
            heading,
            status,
            timestamp,
            ROW_NUMBER() OVER (PARTITION BY mmsi ORDER BY timestamp) AS rn
        FROM position
        JOIN vessel USING (mmsi)
    ),
    nth AS (
        SELECT *
        FROM ordered
        WHERE rn = {n}
    ),
    agg AS (
        SELECT
            mmsi,
            AVG(speed)   AS avg_speed,
            AVG(course)  AS avg_course,
            AVG(heading) AS avg_heading
        FROM ordered
        WHERE rn <= {n}
        GROUP BY mmsi
    )
    SELECT
        nth.mmsi,
        nth.shiptype,
        nth.timestamp AS nth_timestamp,
        agg.avg_speed,
        agg.avg_course,
        agg.avg_heading
    FROM nth
    JOIN agg USING (mmsi)
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
    X = df[["avg_speed", "avg_course", "avg_heading"]].values.astype(np.float32)
    y = df["shiptype"].values.astype(np.float32)
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
    n = 500  # same as your GNN task
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

