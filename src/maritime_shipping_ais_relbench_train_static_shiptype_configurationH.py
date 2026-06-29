#Importing libraries needed
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from relbench.tasks import get_task, get_task_names, register_task
from relbench.datasets import get_dataset, get_dataset_names, register_dataset
import os
import math
from tqdm import tqdm
import torch
from typing import List, Optional, Dict
from torch import Tensor
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
from torch_geometric.seed import seed_everything
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader
from relbench.modeling.utils import get_stype_proposal
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch_frame import stype 

import pickle


# for model
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd

from utils import RelBenchModel, get_loaders, GloveTextEmbedding, make_col_stats

from relbench.base import AutoCompleteTask

from maritime_shipping_ais_relbench_dataset import MaritimeShippingAISDataset

from maritime_shipping_ais_relbench_tasks import ShipTypeNthPositionTask

#Loading model and training parameters



TRAINING_CONFIG={'epochs':200,
                 'learning_rate':1e-4,
                 'step_size':1000,
                 'gamma':0.5,
                 'batch_size':128,
	         'weight_decay': 1e-2}

MODEL_CONFIG = {
    "temporal_strategy": "last",
    "num_neighbors": [64,64],
    "num_layers": 4,
    "channels": 4,
    "aggr": "mean",
    "temporal_encoding": False,
    "hgt_heads": 16,
    "loader_type": "neighbor",
    "model_type": "graphsage"}

#adv_compute_train_stats = True 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Implementing a function to display the training hyperparameters
def print_training_hyperparams(configuration):
    print("Training Hyperparameters:")
    print()
    print(f"Epochs: {configuration['epochs']}")
    print(f"Learning Rate: {configuration['learning_rate']}")
    print(f"Step Size: {configuration['step_size']}")
    print(f"Gamma: {configuration['gamma']}")
    print(f"Batch Size: {configuration['batch_size']}")
    print(f"Weight Decay: {configuration['weight_decay']}")
    print()
#Implementing a function to display the model GNN hyperparameters
def print_model_params(configuration):
    print("Model Parameters:")
    print(f"Num Neighbours: {configuration['num_neighbors']}")
    print(f"Channels: {configuration['channels']}")
    print(f"Layers: {configuration['num_layers']}")
    print()






#Registering dataset and task in Relbench

#Implementing a function that return a dataset and the task from relbench
def register_dataset_and_task():
    #Registering the maritime shipping dataset into Relbench
    register_dataset("rel-custom-maritime_shipping_ais", MaritimeShippingAISDataset)
    dataset = get_dataset("rel-custom-maritime_shipping_ais", download=False)
    #Defining a task for our prediction
    task = ShipTypeNthPositionTask(
            dataset,
            cache_dir="./cache/ship_type_np_task"
        )
    print("Task created:", task)
    #Registering the task for a given dataset in Relbench
    register_task("rel-custom-maritime_shipping_ais", "ship_type_np_task", ShipTypeNthPositionTask)
    print('Task registered')
    return dataset,task

#Implementing a function that given a task from Relbench it returns 
# the training, validation and testing tables


def load_train_val_test_tables(task):
    train_table = task._get_table("train")
    val_table = task._get_table("val")
    test_table = task._get_table("test")
    return train_table,val_table,test_table



#Implementing a function that return for a given task the columns to be removed
#as this can't be considered an input of the model
def get_remove_columns(task):
    # Check wether the task requires removing columns from the input features.
    # If so, add the column names to remove_columns
    if getattr(task, 'time_independent_node_task', False):
        remove_columns = task.remove_columns
    elif isinstance(task, AutoCompleteTask):
        remove_columns = task.remove_columns
        remove_columns.append(task.target_col) # For autocompletetask manually add the target col to remove_columns
    else:
        remove_columns = []
    if len(remove_columns) > 0:
        print(f'\nNote: This is a special node property task that requires removing columns {remove_columns} from the input data to avoid leakage.')
    return remove_columns
#Immplementing a function that given a task it set the out_channels, the relevant 
#loss,performance metric, and the criteria to evalutate the performance metric
def get_task_configuration(task):
    tasktype = task.task_type.value

    if tasktype == 'regression':
        loss_fn = L1Loss() #alternatively use loss_fn = MSELoss()    
        tune_metric = "mae"
        higher_is_better = False
        out_channels = 1    

    if tasktype == 'binary_classification':
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True

    if tasktype == 'multiclass_classification':
        out_channels = task.num_labels
        loss_fn = CrossEntropyLoss()
        tune_metric = "accuracy" 
        higher_is_better = True
    return out_channels,loss_fn, tune_metric, higher_is_better


#Implementing a fucntion that given a dataset in Relbench returns its corresponding database
def prepare_db_full(dataset):
    db_full = dataset.get_db(upto_test_timestamp=False) # Setting upto_test_time = False to get FULL database
    # Make dictionary with column types for full dataset
    col_to_stype_dict = get_stype_proposal(db_full) 

    print("=== Column type proposal ===")
    for table, cols in col_to_stype_dict.items():
        print(table)
        for col, stype in cols.items():
            print(f"  {col}: {stype}")


    print("Cleaning text-embedded columns...")
    for table_name, table in db_full.table_dict.items():
        df = table.df.copy()
        for col, s in col_to_stype_dict[table_name].items():
            if s == stype.text_embedded:   # <-- correct enum comparison
                print(f"Cleaning {table_name}.{col}")
                df[col] = df[col].fillna("Unknown").astype(str)
        table.df = df
    print("Finished cleaning text columns.")


    for name, table in db_full.table_dict.items():
        if table.time_col is not None:
            table.df[table.time_col] = table.df[table.time_col].astype("datetime64[s]")
            print(f"Table: {name}, column: {table.time_col}, dtype: {table.df[table.time_col].dtype}")
    return db_full, col_to_stype_dict

def build_graph_from_db(dataset,task,db_full,stype_dict_to_use,text_embedder_cfg):
    root_dir = "./data"

        # Instantiate the FULL graph to be used for training/validation/testing. The SAMPLER or LOADER will be responsible for not sampling validation or test data during training.
    data_full, full_stats = make_pkey_fkey_graph(
        db_full,
        col_to_stype_dict=stype_dict_to_use,  # speficied column types 
        text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
        cache_dir=os.path.join(            # Careful about caching! Make sure we are using the correct version of the graph for modelling 
            root_dir, f"{dataset}_{task}_full_cache" # store materialized graph for convenience
        ),  
    )
    print('Building or Loading Train Graph and Train Col Stats')
    data_train, train_col_stats_dict = make_pkey_fkey_graph(
        db_full.upto(dataset.val_timestamp - pd.Timedelta("1ns")),  # only use data up to val timestamp for training statistics
        col_to_stype_dict=stype_dict_to_use, 
        text_embedder_cfg=text_embedder_cfg, 
        cache_dir=os.path.join(root_dir, f"{dataset}_{task}_train_cache")            
        )
    
    
    """if adv_compute_train_stats and dataset == 'rel-custom-maritime_shipping_ais': 
        # Special inductive train stats for static tables
        # If using this overwrite the simple train_col_stats_dict obtained above
        # Get the col_stats_dict for the training data (used for normalisation)
        train_col_stats_dict = make_col_stats(
            db=db_full,
            timestamp=dataset.val_timestamp,  #up_to_timestamp
            static_tables=["vessels", "ports"],
            key_map={"vessels": "mmsi", "ports": "port_code"},
            col_to_stype_dict=stype_dict_to_use,
            text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
            cache_dir=os.path.join(root_dir, f"{dataset}_{task}_train_cache")
        )"""
    print('Computing load train column stats done')
    print('train_cols_stats dict',train_col_stats_dict)
    return data_full,data_train,train_col_stats_dict




def normalize_split_times(train_table,val_table,test_table):
    # Final timestamp normalization before loader creation
    print("Normalizing timestamps inside train/val/test tables...")

    for tbl_name, tbl in {"train": train_table, "val": val_table, "test": test_table}.items():
        if tbl.time_col is not None:
            col = tbl.time_col
            tbl.df[col] = tbl.df[col].astype("datetime64[s]").copy()
            print(f"{tbl_name}.{col} dtype -> {tbl.df[col].dtype}")


#Getting data loaders
def create_loaders(data_train,data_full,task):
    train_table,val_table,test_table=load_train_val_test_tables(task)
    
    loader_train = get_loaders(
    data=data_train,
    task=task,
    tables= {"train": train_table},
    num_neighbors=MODEL_CONFIG['num_neighbors'],
    batch_size=TRAINING_CONFIG['batch_size'],
    temporal_strategy=MODEL_CONFIG['temporal_strategy'],
    loader_type=MODEL_CONFIG['loader_type'],  
    num_workers=0)

    tables_inference={"val": val_table, "test": test_table}

    loader_inference = get_loaders(
    data=data_full,
    task=task,
    tables=tables_inference,
    num_neighbors=MODEL_CONFIG['num_neighbors'],
    batch_size=TRAINING_CONFIG['batch_size'],
    temporal_strategy=MODEL_CONFIG['temporal_strategy'],
    loader_type=MODEL_CONFIG['loader_type'],  
    num_workers=0)
    return loader_train,loader_inference


def build_model(task,data_train,train_col_stats_dict):

    model = RelBenchModel(
    model_type=MODEL_CONFIG['model_type'],
    loader_type=MODEL_CONFIG['loader_type'],    
    data=data_train,
    col_stats_dict=train_col_stats_dict,
    num_layers=MODEL_CONFIG['num_layers'],
    channels=MODEL_CONFIG['channels'],
    out_channels=get_task_configuration(task)[0],
    aggr=MODEL_CONFIG['aggr'],
    norm="batch_norm",
    hgt_heads=MODEL_CONFIG['hgt_heads'],
    temporal_encoding=MODEL_CONFIG['temporal_encoding'],
    ).to(device)
    return model 

def train(loader,model,task) -> float:
    
    model.train()
    tasktype=task.task_type.value
    loss_fn=get_task_configuration(task)[1]

    loss_accum = count_accum = 0
    for batch in tqdm(loader):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        if tasktype=='multiclass_classification':
            target = batch[task.entity_table].y.long()


        if tasktype=='regression' or tasktype=='binary_classification':
            target = batch[task.entity_table].y.float()  

        pred = pred.float()        
        loss = loss_fn(pred, target)   
        
        loss.backward()
        optimizer.step()
        #calculating the loss for a batch and multiply by its the batch size
        loss_accum += loss.detach().item() * pred.size(0) 
        count_accum += pred.size(0)#total number of samples across all batches
    train_loss=loss_accum/count_accum
    return train_loss
    
@torch.no_grad()
def test(loader,model,task) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()

def run_train_and_evaluate(model,task,loader_train,loader_inference,loss_fn,optimizer,scheduler,tune_metric,higher_is_better,num_epochs):
    tasktype=task.task_type.value
    best_val_metric = -math.inf if higher_is_better else math.inf
    state_dict=None

    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    train_metrics_list=[]
    val_metrics_list = []
    test_metrics_list = []
    train_table = task._get_table("train") 
    val_table = task._get_table("val")
    test_table = task._get_table("test")

    for epoch in range(1, num_epochs + 1):
        train_loss = train(loader_train['train'],model,task) # defining train loss value as accuracy for non-regression task (for printing)
     
        

        train_pred = test(loader_train["train"], model, task)
        train_metrics = task.evaluate(train_pred, train_table)
        train_metrics_list.append(train_metrics)
        scheduler.step()      # step scheduler
        #For validation:
       
        val_pred = test(loader_inference["val"],model,task)
        val_metrics = task.evaluate(val_pred, val_table)
        #For testing
        
        test_pred = test(loader_inference["test"],model,task)
        test_metrics = task.evaluate(test_pred,test_table)  

        val_loss_accum = 0
        val_count_accum = 0
        for batch in loader_inference["val"]:
            batch = batch.to(device)
            pred = model(batch, task.entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if tasktype == "multiclass_classification":
                target = batch[task.entity_table].y.long()
            else:
                target = batch[task.entity_table].y.float()

            loss = loss_fn(pred.float(), target)
            val_loss_accum += loss.detach().item() * pred.size(0)
            val_count_accum += pred.size(0)

        val_loss = val_loss_accum / val_count_accum

        # === TEST LOSS (batch-weighted) ===
        test_loss_accum = 0
        test_count_accum = 0
        for batch in loader_inference["test"]:
            batch = batch.to(device)
            pred = model(batch, task.entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if tasktype == "multiclass_classification":
                target = batch[task.entity_table].y.long()
            else:
                target = batch[task.entity_table].y.float()

            loss = loss_fn(pred.float(), target)
            test_loss_accum += loss.detach().item() * pred.size(0)
            test_count_accum += pred.size(0)

        test_loss = test_loss_accum / test_count_accum
        

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Val Metric ({tune_metric}): {val_metrics[tune_metric]:.4f}"
        )
        #print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}, Test metrics: {test_metrics}")

        # Save metrics per epoch
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        val_metrics_list.append(val_metrics)
        test_metrics_list.append(test_metrics)

        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(state_dict)
    val_pred = test(loader_inference["val"],model,task)
    val_metrics = task.evaluate(val_pred, val_table)
    print(f"Best Val metrics: {val_metrics}")

    test_pred = test(loader_inference["test"],model,task)
    #test_metrics = task.evaluate(test_pred)
    test_metrics = task.evaluate(test_pred,test_table) # Manually set test table. 
    print(f"Best test metrics: {test_metrics}")

    return  (train_loss_list,val_loss_list,test_loss_list,train_metrics_list,val_metrics_list,test_metrics_list,test_pred,test_table,val_metrics,test_metrics)
    


def save_results_to_csv(output_dir,tasktype,test_table,target_col_name,test_pred,train_loss_list,val_loss_list,test_loss_list,train_metrics_list,val_metrics_list,test_metrics_list):
    os.makedirs(output_dir, exist_ok=True)
    # test targets
    targets_path = os.path.join(output_dir, "test_targets.csv")
    test_table.df[target_col_name].to_csv(targets_path, index=False)

    # test predictions
    preds_path = os.path.join(output_dir, "test_predictions.csv")
    # If test_pred is a numpy array, convert to DataFrame

    if tasktype=='regression':
        preds_df = pd.DataFrame(test_pred, columns=["predictions"])
        preds_df.to_csv(preds_path, index=False)

    if tasktype=='multiclass_classification' :
    # Convert test predictions to numpy
        if isinstance(test_pred, torch.Tensor):
            test_pred = test_pred.cpu().numpy()
        pred_class = np.argmax(test_pred, axis=1)
        # Save predictions
        preds_df = pd.DataFrame(pred_class, columns=["predictions"])
        preds_df.to_csv(preds_path, index=False)

    if tasktype == 'binary_classification':
        # logits > 0 → class 1
        pred_class = (test_pred > 0).astype(int)
        preds_df = pd.DataFrame(pred_class, columns=["predictions"])
        preds_df.to_csv(preds_path, index=False)


    # save training curves
    # Save training loss, validation loss and test loss
    pd.DataFrame({"train_loss": train_loss_list}).to_csv(
        os.path.join(output_dir, "train_loss.csv"), index_label="epoch"
    )

    pd.DataFrame({"val_loss": val_loss_list}).to_csv(
        os.path.join(output_dir, "val_loss.csv"), index_label="epoch"
    )

    pd.DataFrame({"test_loss": test_loss_list}).to_csv(
        os.path.join(output_dir, "test_loss.csv"), index_label="epoch"
    )
    # train metrics
    pd.DataFrame(train_metrics_list).to_csv(
    os.path.join(output_dir, "train_metrics.csv"), index_label="epoch"
)


    # val metrics
    pd.DataFrame(val_metrics_list).to_csv(
        os.path.join(output_dir, "val_metrics.csv"), index_label="epoch"
    )

    # test metrics
    pd.DataFrame(test_metrics_list).to_csv(
        os.path.join(output_dir, "test_metrics.csv"), index_label="epoch"
    )

    print(f"Saved test targets to {targets_path}")
    print(f"Saved test predictions to {preds_path}")
        


def plot_learning_curves_from_csv(output_dir):
    train_loss = pd.read_csv(os.path.join(output_dir, "train_loss.csv"))["train_loss"]
    val_loss = pd.read_csv(os.path.join(output_dir, "val_loss.csv"))["val_loss"]
    test_loss = pd.read_csv(os.path.join(output_dir, "test_loss.csv"))["test_loss"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10,6))
    plt.plot(epochs,train_loss, label='Training loss')
    plt.plot(epochs,val_loss, label=' Validation loss')
    plt.plot(epochs,test_loss, label='Testing loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss BCE Logit Loss")
    plt.title("Learning curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'Learning curves.png'))
    plt.close()
    print("Saved learning curves")

def plot_roc_auc_curves_from_csv(output_dir):
    # Load metrics CSVs
    train_metrics = pd.read_csv(os.path.join(output_dir, "train_metrics.csv"))
    val_metrics = pd.read_csv(os.path.join(output_dir, "val_metrics.csv"))
    test_metrics = pd.read_csv(os.path.join(output_dir, "test_metrics.csv"))

    # Extract ROC-AUC values
    train_roc = train_metrics["roc_auc"]
    val_roc = val_metrics["roc_auc"]
    test_roc = test_metrics["roc_auc"]

    epochs = range(1, len(train_roc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_roc, label="Training ROC-AUC", color="blue")
    plt.plot(epochs, val_roc, label="Validation ROC-AUC", color="green")
    plt.plot(epochs, test_roc, label="Test ROC-AUC", color="red")

    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC Curves Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "roc_auc_curves.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved ROC-AUC curves to {save_path}")




def main():
    seed_everything(42)

    # dataset and task
    dataset, task = register_dataset_and_task()
    train_table, val_table, test_table = load_train_val_test_tables(task)
    print('Train table',train_table)
    print('Validation table',val_table)
    print('Testing table',test_table)

    tasktype = task.task_type.value
    target_col_name = task.target_col
    task_name = "ship_type_np_task"

    # remove leakage columns
    remove_columns = get_remove_columns(task)
    print('Remove columns')
    print(remove_columns)

    # database and stypes


    
    db_full, col_to_stype_dict = prepare_db_full(dataset)

    if len(remove_columns) > 0:
        col_to_stype_dict = {
            table_name: {
                col: col_type
                for col, col_type in cols.items()
                if not  (col in remove_columns)
            }
            for table_name, cols in col_to_stype_dict.items()#removing columns to be excluded by the model
        }
    print('Cols type')
    print(col_to_stype_dict)
    print('End cols type')

    # text embedder
    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256,
    )
    print('Col to stype:',col_to_stype_dict)
    # graphs
    data_full, data_train, train_col_stats_dict = build_graph_from_db(
        dataset, task, db_full, col_to_stype_dict, text_embedder_cfg
    )#havr all feature
    # vessels
    print("vessels keys:")
    print(data_full['vessels'].keys())

    print("vessels tf:")
    print(data_full['vessels'].tf)


    # vessels_details
    print("vessels_details keys:")
    print(data_full['vessels_details'].keys())

    print("vessels_details tf:")
    print(data_full['vessels_details'].tf)


    # positions
    print("positions keys:")
    print(data_full['positions'].keys())

    print("positions tf:")
    print(data_full['positions'].tf)


    # voyages
    print("voyages keys:")
    print(data_full['voyages'].keys())

    print("voyages tf:")
    print(data_full['voyages'].tf)


    # ports
    print("ports keys:")
    print(data_full['ports'].keys())

    print("ports tf:")
    print(data_full['ports'].tf)


    print('Data train')
    print(data_train)

    

    #remove feature after building graph


    # normalize timestamps
    normalize_split_times(train_table, val_table, test_table)

    # loaders
    loader_train, loader_inference = create_loaders(data_train, data_full, task)

    # task configuration
    out_channels, loss_fn, tune_metric, higher_is_better = get_task_configuration(task)

    

    # model
    model = build_model(task, data_train, train_col_stats_dict)

    # optimizer and scheduler
    global optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=TRAINING_CONFIG["step_size"],
        gamma=TRAINING_CONFIG["gamma"],
    )

    # training and evaluation
    (
        train_loss_list,
        val_loss_list,
        test_loss_list,
        train_metrics_list,
        val_metrics_list,
        test_metrics_list,
        test_pred,
        test_table,
        val_metrics,
        test_metrics,
    ) = run_train_and_evaluate(
        model=model,
        task=task,
        loader_train=loader_train,
        loader_inference=loader_inference,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        tune_metric=tune_metric,
        higher_is_better=higher_is_better,
        num_epochs=TRAINING_CONFIG["epochs"],
    )
    # save results
    output_dir = "ship type prediction/heterogenuos graph_sage_configurationH_db"
    save_results_to_csv(
        output_dir=output_dir,
        tasktype=tasktype,
        test_table=test_table,
        target_col_name=target_col_name,
        test_pred=test_pred,
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        test_loss_list=test_loss_list,
        train_metrics_list=train_metrics_list,
        val_metrics_list=val_metrics_list,
        test_metrics_list=test_metrics_list,
    )

    # plot curves
    plot_learning_curves_from_csv(output_dir)
    plot_roc_auc_curves_from_csv(output_dir)
     
    print('Prediction of the heterogenuos graph sage  based on entire db under configuration H ')
    # Display parameters
    print_training_hyperparams(TRAINING_CONFIG)
    print_model_params(MODEL_CONFIG)

    
if __name__=="__main__":
    main()