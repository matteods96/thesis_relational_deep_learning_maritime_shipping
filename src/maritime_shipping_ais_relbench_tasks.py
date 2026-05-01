import duckdb
import pandas as pd
import relbench
import shutil
import os

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)
import duckdb
import pandas as pd

from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names, register_task
from relbench.datasets import get_dataset, get_dataset_names, register_dataset
from maritime_shipping_ais_relbench_dataset import MaritimeShippingAISDataset
from relbench.utils import get_relbench_cache_dir



class ShipTypeNthPositionTask(EntityTask):
    """
    Predict the static property ship type of a vessel after its nth position,
    using only graph information available up to that point.

    This is a static node classification task:
    - The target is a static property of the vessel node.
    - Each vessel is included once, aligned to its nth recorded transaction in the "positions" table. 
    - Timedeltas and prediction windows are not used. Instead we predict on each vessel node at the timestamp of their nth position.
    """

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col='mmsi' # node identifier
    entity_table='vessels' # node table
    time_col = "nth_time"   # The selected time for modelling. Here when the vessel first appears in a specific position
    target_col='shiptype' # target column shiptype

    # dummy timedelta and num_eval_timestamps for BaseTask checks. Not used for windowing here
    timedelta = pd.Timedelta(days=5)
    num_eval_timestamps =1

    #Metrics and number of labels
    #metrics=[accuracy]
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_labels=2 #In the dataset there are 2 types of ship: Cargo and Tanker


    # Special attribute for static node property tasks. 
    # This list specifies which features to remove from the INPUT graph to the task.
    # This should be the target feature, or any features that directly map to the target feature.
    remove_feats=['shiptype']

    #Flag this is a special types of time-indipendent node property task
    time_independent_node_task=True
    remove_columns=remove_feats

    #Special attribute for time-indipendent node property task
    #This number indicates how many interactions the target should have at prediction time
    n=500
    interaction_table='positions'
    interaction_table_time_col='timestamp'


    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        """
        Construct a table containing the nth interaction timestamp and the corresponding target value for each entity.

        Steps performed:

        distinct_interactions:
           - Selects all distinct interaction timestamps for each entity from the interaction table.

        entity_interaction_ranks:
           - Assigns a sequential number (interaction_number) to each interaction per entity
             based on the interaction timestamp (earliest first). This allows identifying the nth interaction for each entity.

        nth_interaction:
           - Filters the interactions to keep only the nth interaction timestamp per entity.

        Final SELECT:
           - Joins the nth interaction with the entity table to get the target value.
           - Filters out entities with NULL target values.
           - Orders the resulting table by the interaction timestamp.

        """     
        con=duckdb.connect()
        con.register(self.interaction_table,db.table_dict[self.interaction_table].df)
        con.register(self.entity_table,db.table_dict[self.entity_table].df)



        query = f"""
            WITH distinct_interactions AS (
                SELECT DISTINCT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                FROM {self.interaction_table}
            ),
            entity_interaction_ranks AS (
                SELECT
                    {self.entity_col},
                    {self.interaction_table_time_col},
                    ROW_NUMBER() OVER (PARTITION BY {self.entity_col} ORDER BY {self.interaction_table_time_col}) AS interaction_number
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

        df=con.execute(query).df()
        #Convert shiptypes as binary label 
        ship_type_dict={'Tanker':1,'Cargo':0}
        df['shiptype']=df['shiptype'].map(ship_type_dict)


        # Infer split based on timestamps
        if len(timestamps) > 1 and timestamps[0] > timestamps[1]:
            split = "train"
        elif timestamps[0] == self.dataset.val_timestamp:
            split = "val"
        elif timestamps[0] == self.dataset.test_timestamp:
            split = "test"
        else:
            raise ValueError("Could not infer split from timestamps")

        val_ts = self.dataset.val_timestamp
        test_ts = self.dataset.test_timestamp

        # Then perform the split and return the associated table
        if split == "train":
            df = df[df[self.time_col] < val_ts] 
        elif split == "val":
            df = df[(df[self.time_col] >= val_ts) & (df[self.time_col] < test_ts)]
        elif split == "test":
            df = df[(df[self.time_col] >= test_ts)]

        df = df.reset_index(drop=True)
        
        return Table(
            df=df,
            fkey_col_to_pkey_table={'mmsi':'vessels'},
            pkey_col='mmsi',
            time_col=self.time_col
        )


if __name__ == "__main__":
    # --- FORCE CLEAR TASK CACHE ---
    cache_dir = f"{get_relbench_cache_dir()}/rel-custom-maritime_shipping_ais/tasks/ship_type_np_task"
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    register_dataset("rel-custom-maritime_shipping_ais", MaritimeShippingAISDataset)

    ais_dataset = get_dataset("rel-custom-maritime_shipping_ais", download=False)
    print("Dataset loaded:", ais_dataset)

    task = ShipTypeNthPositionTask(
        ais_dataset,
        cache_dir="./cache/ship_type_np_task"
    )
    print("Task created:", task)

    register_task("rel-custom-maritime_shipping_ais", "ship_type_np_task", ShipTypeNthPositionTask)
    print('Task registered')

    print("Training table:")
    train_table = task.get_table("train")
    print(train_table)

    print("Task uploaded successfully")










