# MODEL PARAMETER SEARCH USING VALIDATION LOSS AFTER CONVERGENCE

import os
import math
import json
import copy
import pickle
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from typing import Any, Dict, List, Optional
from torch import Tensor

from torch.nn import (
    BCEWithLogitsLoss,
    L1Loss,
    MSELoss,
    CrossEntropyLoss
)

from relbench.datasets import register_dataset, get_dataset
from relbench.tasks import register_task
from relbench.base import AutoCompleteTask

from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from torch_geometric.seed import seed_everything
import torch.optim.lr_scheduler as lr_scheduler

from utils import RelBenchModel, get_loaders, GloveTextEmbedding, make_col_stats

from maritime_shipping_ais_relbench_dataset import MaritimeShippingAISDataset
from maritime_shipping_ais_relbench_tasks import ShipTypeNthPositionTask


# 
# DATASET + TASK
# 

def register_dataset_and_task():
    register_dataset("rel-custom-maritime_shipping_ais", MaritimeShippingAISDataset)
    dataset = get_dataset("rel-custom-maritime_shipping_ais", download=False)

    task = ShipTypeNthPositionTask(
        dataset,
        cache_dir="./cache/ship_type_np_task"
    )
    register_task("rel-custom-maritime_shipping_ais", "ship_type_np_task", ShipTypeNthPositionTask)
    return dataset, task


def load_train_val_test_tables(task):
    return (
        task._get_table("train"),
        task._get_table("val"),
        task._get_table("test")
    )


def get_remove_columns(task):
    if getattr(task, "time_independent_node_task", False):
        return task.remove_columns
    elif isinstance(task, AutoCompleteTask):
        cols = task.remove_columns
        cols.append(task.target_col)
        return cols
    return []


def get_task_configuration(task):
    tasktype = task.task_type.value

    if tasktype == "regression":
        return 1, L1Loss(), "mae", False

    if tasktype == "binary_classification":
        return 1, BCEWithLogitsLoss(), "roc_auc", True

    if tasktype == "multiclass_classification":
        return task.num_labels, CrossEntropyLoss(), "accuracy", True


# 
# DATABASE + GRAPH
# 

def prepare_db_full(dataset):
    db_full = dataset.get_db(upto_test_timestamp=False)
    col_to_stype_dict = get_stype_proposal(db_full)

    for table_name, table in db_full.table_dict.items():
        df = table.df.copy()
        for col, s in col_to_stype_dict[table_name].items():
            if s == stype.text_embedded:
                df[col] = df[col].fillna("Unknown").astype(str)
        table.df = df

    for name, table in db_full.table_dict.items():
        if table.time_col is not None:
            table.df[table.time_col] = table.df[table.time_col].astype("datetime64[s]")

    return db_full, col_to_stype_dict


def build_graph_from_db(dataset, task, db_full, stype_dict_to_use, text_embedder_cfg):
    root_dir = "./data"

    data_full, _ = make_pkey_fkey_graph(
        db_full,
        col_to_stype_dict=stype_dict_to_use,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=os.path.join(root_dir, f"{dataset}_{task}_full_cache")
    )

    data_train, train_col_stats_dict = make_pkey_fkey_graph(
        db_full.upto(dataset.val_timestamp - pd.Timedelta("1ns")),
        col_to_stype_dict=stype_dict_to_use,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=os.path.join(root_dir, f"{dataset}_{task}_train_cache")
    )

    return data_full, data_train, train_col_stats_dict


def normalize_split_times(train_table, val_table, test_table):
    for tbl_name, tbl in {"train": train_table, "val": val_table, "test": test_table}.items():
        if tbl.time_col is not None:
            col = tbl.time_col
            tbl.df[col] = tbl.df[col].astype("datetime64[s]").copy()


# 
# MODEL PARAMETER SEARCH (CONVERGENCE-BASED)
# 

def evaluate_model_params(
    num_neighbors,
    channels,
    num_layers,
    train_params,
    data_train,
    data_full,
    train_table,
    val_table,
    task,
    train_col_stats_dict,
    out_channels,
    loss_fn,
    device
):
    # Loaders
    loader_train = get_loaders(
        data=data_train,
        task=task,
        tables={"train": train_table},
        num_neighbors=num_neighbors,
        batch_size=train_params["batch_size"],
        temporal_strategy="last",
        loader_type="neighbor",
        num_workers=0
    )["train"]

    loader_val = get_loaders(
        data=data_full,
        task=task,
        tables={"val": val_table},
        num_neighbors=num_neighbors,
        batch_size=train_params["batch_size"],
        temporal_strategy="last",
        loader_type="neighbor",
        num_workers=0
    )["val"]

    # Model
    model = RelBenchModel(
        model_type="graphsage",
        loader_type="neighbor",
        data=data_train,
        col_stats_dict=train_col_stats_dict,
        num_layers=num_layers,
        channels=channels,
        out_channels=out_channels,
        aggr="mean",
        norm="batch_norm",
        hgt_heads=16,
        temporal_encoding=False
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_params["learning_rate"],
        weight_decay=1e-2
    )
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=train_params["step_size"],
        gamma=train_params["gamma"]
    )

    # === Convergence tracking ===
    patience = 10
    epochs_without_improvement = 0
    best_val_loss = float("inf")
    best_state = None

    # === Training loop ===
    for epoch in range(1, train_params["n_epochs"] + 1):
        model.train()
        for batch in loader_train:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch, task.entity_table).view(-1)
            target = batch[task.entity_table].y.float()

            loss = loss_fn(pred.float(), target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # === Validation loss ===
        model.eval()
        val_loss_accum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in loader_val:
                batch = batch.to(device)
                p = model(batch, task.entity_table).view(-1)
                target = batch[task.entity_table].y.float()
                vloss = loss_fn(p.float(), target)
                val_loss_accum += vloss.item() * p.size(0)
                val_count += p.size(0)

        val_loss = val_loss_accum / val_count

        # === Convergence check ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Load best converged model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


# 
# MAIN
# 

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, task = register_dataset_and_task()
    train_table, val_table, test_table = load_train_val_test_tables(task)

    remove_columns = get_remove_columns(task)
    out_channels, loss_fn, tune_metric, higher_is_better = get_task_configuration(task)

    db_full, col_to_stype_dict = prepare_db_full(dataset)

    if len(remove_columns) > 0:
        col_to_stype_dict = {
            t: {c: s for c, s in cols.items() if c not in remove_columns}
            for t, cols in col_to_stype_dict.items()
        }

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256
    )

    data_full, data_train, train_col_stats_dict = build_graph_from_db(
        dataset, task, db_full, col_to_stype_dict, text_embedder_cfg
    )

    normalize_split_times(train_table, val_table, test_table)

    # Load best training hyperparameters
    with open("best_hyperparameters_1.txt", "r") as f:
        train_params = json.load(f)

    # Model parameter search space
    search_num_neighbors = [[32, 16], [64, 32], [128, 32]]
    search_channels = [64, 128, 256]
    search_num_layers = [2, 4, 6]

    best_score = float("inf")
    best_model_params = None

    # === Grid search ===
    for neigh in search_num_neighbors:
        for ch in search_channels:
            for nl in search_num_layers:
                print(f"Testing model config: neighbors={neigh}, channels={ch}, num_layers={nl}")

                score = evaluate_model_params(
                    neigh, ch, nl,
                    train_params,
                    data_train, data_full,
                    train_table, val_table,
                    task, train_col_stats_dict,
                    out_channels, loss_fn,
                    device
                )

                print("Validation loss after convergence:", score)

                if score < best_score:
                    best_score = score
                    best_model_params = {
                        "num_neighbors": neigh,
                        "channels": ch,
                        "num_layers": nl
                    }

    print("\nBest model parameters found:")
    print(best_model_params)
    print("Best validation loss after convergence:", best_score)

    with open("best_model_parameters_1.txt", "w") as f:
        json.dump(best_model_params, f)

    print("Saved best model parameters to best_model_parameters_1.txt")


if __name__ == "__main__":
    main()
