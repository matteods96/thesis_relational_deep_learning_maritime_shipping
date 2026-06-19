import duckdb
import pandas as pd
import relbench
import shutil
import os

from relbench.base import Database, EntityTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    roc_auc,
)

from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names, register_task
from relbench.datasets import get_dataset, get_dataset_names, register_dataset
#from maritime_shipping_ais_relbench_dataset import MaritimeShippingAISDataset
from relbench.utils import get_relbench_cache_dir
from maritime_shipping_ais_relbench_dataset_new import MaritimeShippingAISDatasetNew
from relbench.utils import get_relbench_cache_dir


class ShipTypeNthPositionTaskNew(EntityTask):
    """
    Predict static ship type after the vessel's nth AIS position.
    """

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "mmsi"
    entity_table = "vessels"
    time_col = "nth_time"
    target_col = "shiptype"

    period_start = pd.Timestamp("2025-04-07")
    period_end   = pd.Timestamp("2025-04-21")

    timedelta = pd.Timedelta(days=1)
    num_eval_timestamps = 1

    metrics = [average_precision, accuracy, f1, roc_auc]
    num_labels = 2

    # REMOVE ONLY FEATURE WHICH COULD BE NOT RELEVANT FOR THE PREDICTION 
    remove_columns = [
        "shiptype",  # target
        "imo", "callsign", "shipname",
        "status", "turn", "speed", "lon", "lat", "course", "heading",
        "draught", "month", "day", "hour", "minute",
        "port_id", "port_name", "country_code", "country_name", "polygon",
        "median_to_bow","median_to_stern","median_to_port","median_to_starboard"
    ]

    time_independent_node_task = True

    n = 500
    interaction_table = "positions"
    interaction_table_time_col = "timestamp"


    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:

        con = duckdb.connect()
        con.register(self.interaction_table, db.table_dict[self.interaction_table].df)
        con.register(self.entity_table, db.table_dict[self.entity_table].df)

       
        query = f"""
            WITH distinct_interactions AS (
                SELECT DISTINCT
                    {self.entity_col},
                    {self.interaction_table_time_col}
                FROM {self.interaction_table}
            ),
            entity_interaction_ranks AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                    ROW_NUMBER() OVER (
                        PARTITION BY {self.entity_col}
                        ORDER BY {self.interaction_table_time_col}
                    ) AS interaction_number
                FROM distinct_interactions
            ),
            nth_interaction AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col} AS {self.time_col}
                FROM entity_interaction_ranks
                WHERE interaction_number = {self.n}
            )
            SELECT
                ni.{self.entity_col},
                ni.{self.time_col},
                et.{self.target_col}
            FROM nth_interaction ni
            JOIN {self.entity_table} et
              ON et.{self.entity_col} = ni.{self.entity_col}
            WHERE et.{self.target_col} IS NOT NULL
            ORDER BY ni.{self.time_col} ASC
        """

        df = con.execute(query).df()
        df = df.dropna()

        # Convert shiptype to binary
        ship_type_dict = {"Cargo/Tanker": 1, "Other": 0}
        df["shiptype"] = df["shiptype"].map(ship_type_dict)

        
        val_ts = self.dataset.val_timestamp
        test_ts = self.dataset.test_timestamp

        if len(timestamps) > 1 and timestamps[0] > timestamps[1]:
            split = "train"
        elif timestamps[0] == val_ts:
            split = "val"
        elif timestamps[0] == test_ts:
            split = "test"
        else:
            raise ValueError("Could not infer split")

        if split == "train":
            df = df[df[self.time_col] < val_ts]
        elif split == "val":
            df = df[(df[self.time_col] >= val_ts) & (df[self.time_col] < test_ts)]
        elif split == "test":
            df = df[df[self.time_col] >= test_ts]

        df = df.reset_index(drop=True)

        return Table(
            df=df,
            fkey_col_to_pkey_table={"mmsi": "vessels"},
            pkey_col="mmsi",
            time_col=self.time_col
        )

#main
if __name__ == "__main__":

    cache_dir = f"{get_relbench_cache_dir()}/rel-custom-maritime_shipping_ais_new/tasks/ship_type_np_task_new"
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    register_dataset("rel-custom-maritime_shipping_ais_new", MaritimeShippingAISDatasetNew)

    ais_dataset = get_dataset("rel-custom-maritime_shipping_ais_new", download=False)
    ais_db = ais_dataset.make_db()

    task = ShipTypeNthPositionTaskNew(
        ais_dataset,
        cache_dir="./cache/ship_type_np_task_new"
    )

    register_task("rel-custom-maritime_shipping_ais_new", "ship_type_np_task_new", ShipTypeNthPositionTaskNew)

    print("Training table:")
    print(task.get_table("train"))

    print("Validation table:")
    print(task.get_table("val"))

    print("Testing table:")
    print(task.get_table("test"))

    print("Task ready.")
