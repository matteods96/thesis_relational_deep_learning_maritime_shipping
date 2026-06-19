import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch_geometric.seed import seed_everything

from relbench.datasets import register_dataset, get_dataset
from relbench.tasks import get_task
from maritime_shipping_ais_relbench_dataset_new import MaritimeShippingAISDatasetNew

register_dataset("rel-custom-maritime_shipping_ais_new", MaritimeShippingAISDatasetNew)

seed_everything(42)

dataset = get_dataset("rel-custom-maritime_shipping_ais_new", download=False)
task = get_task("rel-custom-maritime_shipping_ais_new", "ship_type_np_task_new", download=False)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

entity_table = dataset.get_db().table_dict[task.entity_table]
entity_df = entity_table.df

mode_cols = [
    "mode_to_bow",
    "mode_to_stern",
    "mode_to_port",
    "mode_to_starboard",
]

ship_type_dict = {"Cargo/Tanker": 1, "Other": 0}

def merge_entity(table):
    df = table.df
    left_key = list(table.fkey_col_to_pkey_table.keys())[0]
    right_key = entity_table.pkey_col
    merged = pd.DataFrame()
    for col in df.columns:
        merged[col] = df[col]
    for col in mode_cols:
        merged[col] = entity_df[col].values
    return merged

train_df = merge_entity(train_table)
val_df = merge_entity(val_table)
test_df = merge_entity(test_table)

def prepare(df):
    y_raw = df["shiptype"]
    y = y_raw.map(ship_type_dict) if y_raw.dtype == object else y_raw.astype(int)
    X = df[mode_cols]
    return X, y

X_train, y_train = prepare(train_df)
X_val, y_val = prepare(val_df)
X_test, y_test = prepare(test_df)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

clf.fit(X_train, y_train)

pred_train = clf.predict(X_train)
pred_val = clf.predict(X_val)
pred_test = clf.predict(X_test)

print("Train:", task.evaluate(pred_train, train_table))
print("Val:", task.evaluate(pred_val, val_table))
print("Test:", task.evaluate(pred_test, test_table))
