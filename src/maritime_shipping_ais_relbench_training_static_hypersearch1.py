import numpy as np
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from relbench.tasks import get_task, get_task_names, register_task
from relbench.datasets import get_dataset, get_dataset_names, register_dataset
import os
import math
import numpy as np
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
import json
import pickle
import json


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



#Registering the Danish AIS dataset into Relbench
register_dataset("rel-custom-maritime_shipping_ais", MaritimeShippingAISDataset)
ais_dataset = get_dataset("rel-custom-maritime_shipping_ais", download=False)
#Defining a task for our prediction
task = ShipTypeNthPositionTask(
        ais_dataset,
        cache_dir="./cache/ship_type_np_task"
    )
print("Task created:", task)
#Registering the task for a given dataset in Relbench
register_task("rel-custom-maritime_shipping_ais", "ship_type_np_task", ShipTypeNthPositionTask)
print('Task registered')




# Model params
temporal_strategy = 'last'                 # Sample strategy. Either uniform or last, which will focus on neighbours close in time to seed node
num_neighbours = [128 for i in range(2)]  # The number of neighbours to sample per depth level
num_layers =2# Number of layers in GNN model
channels = 128                             # Number of channels in GNN model
aggr = "mean"                              # Aggregator function in GNN model
temporal_encoding = False                  # Whether to use temporal encoding in the model (only for GraphSAGE + neighbor loader currently)

w_decay = 1e-2
# Hgt params
hgt_heads = 16

# Batch Sampler 
#loader_type = 'neighbor' #'neighbor'  # 'neighbor' or 'hgt'. OBS hgt loader is NOT time aware
#model_type = 'graphsage'  # or 'graphsage' / 'hgt

loader_type = 'neighbor' #'neighbor'  # 'neighbor' or 'hgt'. OBS hgt loader is NOT time aware
model_type = 'graphsage'  # or 'graphsage' / 'hgt

# Task param
data_name='rel-custom-maritime_shipping_ais'
task_name='ship_type_np_task'


# Advanced inductive temporal method for computing train statistics that also tracks non-time-stamped-nodes related to time-stamped-nodes. Currently only available for rel-hm dataset.
# if False then train stats are computed using all time-stamped nodes present in the training data, and ALL non-time-stamped nodes. 
adv_compute_train_stats = True  

# Print more info. Advised for debugging
verbose = True 


dataset = get_dataset(data_name, download=False) 
task = get_task(data_name, task_name, download=False)

# Using _get_table for uncached
train_table = task._get_table("train") 
val_table = task._get_table("val")
test_table = task._get_table("test")

target_col_table_name = task.entity_table
target_col_name = task.target_col
tasktype = task.task_type.value

print('Training table')
print(train_table)
print('Validation table')
print(val_table)
print('Testing table')
print(test_table)
print('target_col_table_name: ',target_col_table_name)
print('target_col_name: ',target_col_name)
print('tasktype :',tasktype )

# Print what we are working on
print(f'\nWorking on task {task_name} ({tasktype}) for dataset {data_name}...')


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


if tasktype == 'regression':
    loss_fn = L1Loss()
    #loss_fn = MSELoss()    
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

# Some book keeping
seed_everything(42)


train_table = task._get_table("train") 
val_table = task._get_table("val")
test_table = task._get_table("test")


print('Training table')
print(train_table)
print('Validation table')
print(val_table)
print('Testing table')
print(test_table)


if verbose == True:
    print('\n\nTraining table view:')
    print(train_table.df)
    print('\nVal table shape:')
    print(val_table.df.shape)
    print('\nTest table shape:')
    print(test_table.df)
if verbose == True and tasktype == 'multiclass_classification' or tasktype == 'binary_classification':
    print('training target distribution (head 10)')
    print(train_table.df[target_col_name].value_counts().head(10))
    print('val target distribution (head 10)')
    print(val_table.df[target_col_name].value_counts().head(10))
    print('test target distribution (head 10)')
    print(test_table.df[target_col_name].value_counts().head(10))
    print('unique labels train')
    print(train_table.df[target_col_name].unique())
    print('unique labels val')
    print(val_table.df[target_col_name].unique())
    print('unique labels test')
    print(test_table.df[target_col_name].unique())
if verbose == True and tasktype == 'regression':
    print("Train targets: min, max, mean, std:", train_table.df[target_col_name].min(),
    train_table.df[target_col_name].max(), train_table.df[target_col_name].mean(),
    train_table.df[target_col_name].std())
    print("Value counts (top 5):")
    print(train_table.df[target_col_name].value_counts().head(5))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if verbose == True:
    print('device',device)  
root_dir = "./data"

text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device=device), batch_size=256
)



# Build Graph from Dataset ====================================
print('Building or Loading Full Graph')
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







print('Here above result text embedded')



# If task is task requires removing columns from input data then remove specified input features from the input data to avoid leakage
# AutoCompleteTask should do this automatically but we double check here that it is actually done.
if len(remove_columns) > 0:
    # Make a dictionary of input features where some columns are removed. 
    # Later use this dictionary when instantiating the graph
    col_to_stype_dict_clean = {
        table_name: {
            col: col_type
            for col, col_type in cols.items()
            if not (table_name == target_col_table_name and col in remove_columns)
        }
        for table_name, cols in col_to_stype_dict.items()
    }

    stype_dict_to_use = col_to_stype_dict_clean # The modified data with features removed
else: # If not a time-independent special task then use full data (note: Autocomplete task will automatically remove the required features from the data)
    stype_dict_to_use = col_to_stype_dict # The original full data


print(stype_dict_to_use)
# Instantiate the FULL graph to be used for training/validation/testing. The SAMPLER or LOADER will be responsible for not sampling validation or test data during training.
data_full, full_stats = make_pkey_fkey_graph(
    db_full,
    col_to_stype_dict=stype_dict_to_use,  # speficied column types 
    text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
    cache_dir=os.path.join(            # Careful about caching! Make sure we are using the correct version of the graph for modelling 
        root_dir, f"{data_name}_{task_name}_full_cache" # store materialized graph for convenience
    ),  
)
print('Building or Loading Train Graph and Train Col Stats')
data_train, train_col_stats_dict = make_pkey_fkey_graph(
    db_full.upto(dataset.val_timestamp - pd.Timedelta("1ns")),  # only use data up to val timestamp for training statistics
    col_to_stype_dict=stype_dict_to_use, 
    text_embedder_cfg=text_embedder_cfg, 
    cache_dir=os.path.join(root_dir, f"{data_name}_{task_name}_train_cache")            
    )


if verbose == True:
    # Optionally manually inspect the entity table with the target value in to verify it includes the input features we expect
    print(f'Manual inspection of input graph (for table [{target_col_table_name}]):')
    print('Remove columns required for this task:', remove_columns)
    print('IF REMOVE COLUMNS IS REQUIRED FOR THIS TASK, THEN DOUBLE CHECK THAT THESE COLUMNS ARE NOT IN THE DATA BELOW!')
    print(data_full[target_col_table_name].tf)  # shows the TensorFrame of the table containing the target value.

print('Done')

# ==================================== Compute Train Col Stats ====================================
if verbose:
    print('Computing or Loading Train Column Stats')

if adv_compute_train_stats and data_name == 'rel-custom-maritime_shipping_ais': # Special inductive train stats implemented for H&M dataset currently. If using other dataset make sure to change static_tables and key_map
    # If using this overwrite the simple train_col_stats_dict obtained above
    # Get the col_stats_dict for the training data (used for normalisation)
    train_col_stats_dict = make_col_stats(
        db=db_full,
        timestamp=dataset.val_timestamp,  #up_to_timestamp
        static_tables=["vessels", "ports"],
        key_map={"vessels": "mmsi", "ports": "port_code"},
        col_to_stype_dict=stype_dict_to_use,
        text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
        cache_dir=os.path.join(root_dir, f"{data_name}_{task_name}_train_cache")
    )
    print('Computing load train column stats done')
    print('train_cols_stats dict',train_col_stats_dict)
# ================================================================================================
# ==================================== Get Data Loaders and model ====================================
# Final timestamp normalization before loader creation
print("Normalizing timestamps inside train/val/test tables...")

for tbl_name, tbl in {"train": train_table, "val": val_table, "test": test_table}.items():
    if tbl.time_col is not None:
        col = tbl.time_col
        tbl.df[col] = tbl.df[col].astype("datetime64[s]").copy()
        print(f"{tbl_name}.{col} dtype -> {tbl.df[col].dtype}")

def evaluate_hyperparams(n_epochs,learning_rate,step_size, gamma, batch_size):
    loader_dict_train = get_loaders(
    data=data_train,
    task=task,
    tables= {"train": train_table},
    num_neighbors=num_neighbours,
    batch_size=batch_size,
    temporal_strategy=temporal_strategy,
    loader_type=loader_type,  
    num_workers=0,)

    loader_dict_val=get_loaders(
    data=data_full,
    task=task,
    tables= {"val": val_table},
    num_neighbors=num_neighbours,
    batch_size=batch_size,
    temporal_strategy=temporal_strategy,
    loader_type=loader_type,  
    num_workers=0,)

    model = RelBenchModel(
    model_type=model_type,
    loader_type=loader_type,    
    data=data_train,
    col_stats_dict=train_col_stats_dict,
    num_layers=num_layers,
    channels=channels,
    out_channels=out_channels,
    aggr=aggr,
    norm="batch_norm",
    hgt_heads=hgt_heads,
    temporal_encoding=temporal_encoding,
    ).to(device)

    optimizer =  torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay) 

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val = -math.inf if higher_is_better else math.inf


    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in loader_dict_train['train']:
            batch=batch.to(device)
            optimizer.zero_grad()

            pred=model(batch,task.entity_table)
            pred=pred.view(-1) if pred.size(1)==1 else pred


            if tasktype=='multiclass_classification':
                target=batch[task.entity_table].y.long()
            else:
                target=batch[task.entity_table].y.float()

            loss=loss_fn(pred.float(),target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        #Validation
        model.eval()
        preds=[]
        for batch in loader_dict_val['val']:
            batch=batch.to(device)
            p=model(batch,task.entity_table)
            p=p.view(-1) if p.size(1)==1 else p
            preds.append(p.detach().cpu())
        preds=torch.cat(preds,dim=0).numpy()

        val_metrics=task.evaluate(preds, val_table)
        metric=val_metrics[tune_metric]

        if(higher_is_better and metric>best_val) or (not higher_is_better and metric<best_val):
            best_val=metric

    return best_val

n_epochs_list = [100, 150, 200]
learning_rates = [5e-4, 5e-5,5e-6]
step_sizes = [500, 1000]
gammas = [0.5, 0.9]
batch_sizes = [64, 128, 256]


best_score=-math.inf if higher_is_better else math.inf
best_params=None

for n in n_epochs_list:
    for lr in learning_rates:
        for ss in step_sizes:
            for gs in gammas:
                for bs in batch_sizes:
                    print(f"Testing configuration: epochs={n},lr={lr},step={ss},gamma={gs},batch_size={bs}")
                    score=evaluate_hyperparams(n,lr,ss,gs,bs)
                    print('Validation score',score)
                    if (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
                        best_score=score
                        best_params={
                            'n_epochs':n,
                            'learning_rate':lr,
                            'step_size':ss,
                            'gamma':gs,
                            'batch_size':bs}
print('Best hyperparameters values found')
print(best_params)
print('Best validation score',best_score)
print()
print("Writing best training parameters in txt file best_hyperparameters_1.txt")
with open('best_hyperparameters_1.txt','w') as txt_file:
    json.dump(best_params,txt_file)# encoded dict into JSON
print("Done writing dict into .txt file")







