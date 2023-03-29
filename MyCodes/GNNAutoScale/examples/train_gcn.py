import argparse
from math import ceil
import time
import sys
import os
import pickle

from MyTimer import Timer
from torch_autoscale.models import GCN
from torch_autoscale import compute_micro_f1

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from typing import List, Tuple

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import DGLBuiltinDataset
from dgl.data import CoraGraphDataset
from dgl.data import CiteseerGraphDataset
from dgl.data import RedditDataset
from dgl.data import PubmedGraphDataset
from dgl.data import CoauthorCSDataset
from dgl.data import WikiCSDataset
from dgl.data import FlickrDataset
import pdb

from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler

# profiling tools
import torch.profiler
from torch.profiler import record_function

from torch.utils.tensorboard import SummaryWriter

# Running Configurations
METIS_PARTITION = True
SHULFFLE_DATA = True
# Running time profiling
PROFILING = False
# Accuracy convergence logging in Tensorboard
ACCLOG = False
PROF_PATH = '/home/lihz/Codes/dgl/MyCodes/Profiling/GNNAutoScale/tensorboard/Reddit'
PART_PATH = '/home/lihz/Codes/dgl/MyCodes/GNNAutoScale/PartitionedGraph'

# Hyperparameters
DATASET = 'Reddit'
SELF_LOOP = True
# GCN Norm is set to True ('both' in GraphConv) by default
NUM_LAYERS = 2
HIDDEN_CHANNELS = 256
DROPOUT = 0.5
NUM_PARTS = 200
BATCH_SIZE = 50
LR = 0.01
REG_WEIGHT_DECAY = 0.0
NONREG_WEIGHT_DECAY = 0.0
GRAD_NORM = None
EPOCHS = 5

"""
for small datasets, the following settings should be default:
DROP_INPUT = True
BATCH_NORM = False
RESIDUAL = False    
LINEAR = False
POOL_SIZE = None
"""
DROP_INPUT = False
BATCH_NORM = False
RESIDUAL = False
LINEAR = False
POOL_SIZE = 2

# for small datasets, buffer_size should be set to None
BUFFER_SIZE = None


torch.manual_seed(12345)

class GlobalIterater(object):
    def __init__(self):
        self.iter = 0

    def __call__(self):
        self.iter += 1
        return self.iter - 1

if ACCLOG:
    writer = SummaryWriter(PROF_PATH)
    global_iter = GlobalIterater()

if PROFILING:
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(PROF_PATH),
        record_shapes=True,
        with_stack=True,
        with_modules=True
    )
    prof.schedule = torch.profiler.schedule(
            skip_first=0,
            wait=0, 
            warmup=1,
            active=2, 
            # repeat=2
            )

def load_dataset(dataset_name: str) -> DGLBuiltinDataset:
    if SELF_LOOP:
        transform = (AddSelfLoop())
    else:
        transform = None
    if dataset_name == 'Cora':
        data = CoraGraphDataset(transform=transform)
    elif dataset_name == 'Citeseer':
        data = CiteseerGraphDataset(transform=transform)
    elif dataset_name == 'Pubmed':
        data = PubmedGraphDataset(transform=transform)
    elif dataset_name == 'CoauthorCS':
        data = CoauthorCSDataset(transform=transform)
    elif dataset_name == 'WikiCS':
        data = WikiCSDataset(transform=transform)
    elif dataset_name == 'Reddit':
        data = RedditDataset(transform=transform)
    elif dataset_name == 'Flickr':
        data = FlickrDataset(transform=transform)

    return data

def retrieve_dataset_info(data: DGLBuiltinDataset):
    g = data[0]
    num_classes = data.num_classes
    in_size = g.ndata['feat'].shape[1]
    out_size = num_classes
    train_mask = g.ndata['train_mask']
    # train_ids = train_mask.nonzero().squeeze()
    val_mask = g.ndata["val_mask"]
    labels = g.ndata["label"]
    return g, in_size, out_size, train_mask, val_mask, labels


def graph_partition(g, num_parts, partition_method = 'metis') -> List[torch.Tensor]:
    if partition_method == 'metis':
        store_path = os.path.join(PART_PATH, '{}-{}-{}.pickle'.format(DATASET, num_parts, partition_method))
        # Metis Partition
        os.makedirs(os.path.dirname(PART_PATH), exist_ok=True)
        if os.path.exists(store_path):
            with open(store_path, "rb") as file:
                part_dict = pickle.load(file)
        else:
            part_dict = dgl.metis_partition(
                g, num_parts, reshuffle=True, 
                # NOTE: Compared to the original pyg-gas, we balance the training nodes in each partition
                # balance_ntypes=g.ndata['train_mask']
            )
            with open(store_path, "wb") as file:
                pickle.dump(part_dict, file)
        # When set reshuffle=True, the node IDs in batches are new IDs
        batches = [part_dict[i].ndata[dgl.NID] for i in range(num_parts)]
        # Get the corresponding original IDs
        original_ids = [part_dict[i].ndata['orig_id'] for i in range(num_parts)]
        new_order = torch.cat(original_ids, dim=0)
        g = dgl.reorder_graph(g, node_permute_algo='custom', permute_config={'nodes_perm': new_order})
    elif partition_method == 'sequential':
        # Sequantial Partition
        num_nodes = g.num_nodes()
        num_nodes_per_part = 68
        id_seq = torch.arange(num_nodes)
        batches = torch.split(id_seq, num_nodes_per_part)
    else:
        raise NotImplementedError

    return batches

class SubgraphConstructor(object):
    def __init__(self, g, num_layers):
        self.g = g
        self.num_layers = num_layers
    
    def construct_subgraph(self, batches: Tuple[torch.Tensor]):
        IB_nodes = torch.cat(batches, dim=0)
        subgraph = dgl.in_subgraph(self.g, IB_nodes)
        block = dgl.to_block(subgraph, dst_nodes=IB_nodes)
        block.num_nodes_in_each_batch = torch.tensor([batch.size(dim=0) for batch in batches])
        return [block for _ in range(self.num_layers)]

def parse_args_from_block(block, training=True):
    if training is False:
        return None

    else:
        feat = block.srcdata['feat']
        # the number of IB-nodes is the number of output nodes
        num_output_nodes = block.num_dst_nodes()
        # output_nodes are in-batch-nodes
        output_nodes = block.dstdata[dgl.NID].to('cpu')
        IB_nodes = output_nodes
        # only those nodes in input-nodes yet not in input-nodes are out-batch-nodes
        input_nodes = block.srcdata[dgl.NID].to('cpu')
        OB_nodes = input_nodes.index_select(0, torch.arange(output_nodes.size(0), input_nodes.size(0)))
        n_ids = torch.cat((IB_nodes, OB_nodes), dim=0)
        count = block.num_nodes_in_each_batch
        index = 0
        lead_node_id = torch.tensor([0])
        for i in count[:-1]:
            index += i
            lead_node_id =  torch.cat((lead_node_id, torch.flatten(n_ids[index]).to('cpu')), dim=0)
        offset = lead_node_id

        return feat, num_output_nodes, n_ids, offset, count

def train(model: GCN, loader: TorchDataLoader, optimizer, device):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    
    for it, blocks in enumerate(loader):
        block = blocks[0]
        num_input_nodes = block.num_src_nodes()
        num_output_nodes = block.num_dst_nodes()
        num_edges = block.num_edges()
        print("block_num_input_nodes: {}, block_num_output_nodes: {}, block_num_edges: {}".format(num_input_nodes, num_output_nodes, num_edges))
        block = block.to(device)
        args = parse_args_from_block(block, training=True)
        out = model(block, *args)
        train_mask = block.dstdata['train_mask']
        loss = criterion(out[train_mask], block.dstdata['label'][train_mask])

        if ACCLOG:
            writer.add_scalar('train_loss', loss, global_iter())
        loss.backward()
        if GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
        optimizer.step()

        # prepare_data_tic = time.time()
        # block = blocks[0]
        # block = block.to(device)
        # args = parse_args_from_block(block, training=True)
        # prepare_data_toc = time.time()

        # forward_tic = time.time()
        # out = model(block, *args)
        # forward_toc = time.time()

        # backward_tic = time.time()
        # train_mask = block.dstdata['train_mask']
        # loss = criterion(out[train_mask], block.dstdata['label'][train_mask])

        # if ACCLOG:
        #     writer.add_scalar('train_loss', loss, global_iter())
        # loss.backward()
        # if GRAD_NORM is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
        # backward_toc = time.time()
        
        # weight_update_tic = time.time()
        # optimizer.step()
        # weight_update_toc = time.time()

        # print('Time(s): || Prepare Data: {:4f}, Forward: {:4f}, Backward: {:4f}, Weight Update: {:4f}'
        #       .format(
        #             prepare_data_toc - prepare_data_tic,
        #             forward_toc - forward_tic,
        #             backward_toc - backward_tic,
        #             weight_update_toc - weight_update_tic
        #         )
        #     )

@torch.no_grad()
def test(model: GCN, g: dgl.DGLGraph, device):
    model.eval()

    block = dgl.to_block(g).to(device)
    feat = block.srcdata['feat']
    labels = block.srcdata['label'].cpu()
    train_mask = block.srcdata['train_mask'].cpu()
    val_mask = block.srcdata['val_mask'].cpu()
    test_mask = block.srcdata['test_mask'].cpu()
    
    # Full-batch inference since the graph is small
    out = model(block, feat).cpu()
    train_acc = compute_micro_f1(out, labels, train_mask)
    val_acc = compute_micro_f1(out, labels, val_mask)
    test_acc = compute_micro_f1(out, labels, test_mask)

    return train_acc, val_acc, test_acc

@torch.no_grad()
def layerwise_minibatch_test(model: GCN, g: dgl.DGLGraph, out_size: int, device):
    model.eval()

    block = dgl.to_block(g)
    feat = block.srcdata['feat']
    labels = block.srcdata['label'].cpu()
    train_mask = block.srcdata['train_mask'].cpu()
    val_mask = block.srcdata['val_mask'].cpu()
    test_mask = block.srcdata['test_mask'].cpu()
    
    num_total_nodes = g.num_nodes()
    num_partitions_per_it = BATCH_SIZE
    num_partitions = NUM_PARTS
    node_batch_size = num_partitions_per_it * (num_total_nodes // num_partitions)
    out, _ = model.layerwise_inference(g, feat, node_batch_size, out_size, device)

    train_acc = compute_micro_f1(out, labels, train_mask)
    val_acc = compute_micro_f1(out, labels, val_mask)
    test_acc = compute_micro_f1(out, labels, test_mask)

    return train_acc, val_acc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    data = load_dataset(DATASET)
    g, in_size, out_size, train_mask, val_mask, labels = retrieve_dataset_info(data)

    batches = graph_partition(g, num_parts=NUM_PARTS, partition_method='metis')

    constructor = SubgraphConstructor(g, NUM_LAYERS)
    num_partitions_per_it = BATCH_SIZE
    subgraph_loader = TorchDataLoader(
        dataset = batches,
        batch_size=num_partitions_per_it,
        collate_fn = constructor.construct_subgraph,
        shuffle=SHULFFLE_DATA
    )

    best_val_acc = test_acc = 0
    model = GCN(
        num_nodes=g.num_nodes(),
        in_channels=in_size,
        hidden_channels=16,
        out_channels=out_size,
        num_layers=NUM_LAYERS,
        drop_input=DROP_INPUT,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
        residual=RESIDUAL,
        linear=LINEAR,
        pool_size=POOL_SIZE,  # Number of pinned CPU buffers
        buffer_size=BUFFER_SIZE,  # Size of pinned CPU buffers (max #out-of-batch nodes)
    ).to(device)

    if PROFILING:
        prof.start()

    # tic = time.time()
    # test(model, g, device) # Fill the history.
    # toc = time.time()
    # print("Fill History Time(s): {:.4f}".format(toc - tic))

    optimizer = torch.optim.Adam([
        dict(params=model.reg_modules.parameters(), weight_decay=REG_WEIGHT_DECAY),
        dict(params=model.nonreg_modules.parameters(), weight_decay=NONREG_WEIGHT_DECAY)
    ], lr=LR)
    for epoch in range(0, EPOCHS):
        train_timer = Timer()
        train_timer.start()
        with record_function("2: Train"):
            print("Training epoch {}".format(epoch))
            train(model, subgraph_loader, optimizer, device)
        train_timer.end()
        test_timer = Timer()
        test_timer.start()
        with record_function("3: Test"):
            train_acc, val_acc, tmp_test_acc = layerwise_minibatch_test(model, g, out_size, device)
            # train_acc, val_acc, tmp_test_acc = test(model, g, device)
        test_timer.end()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if ACCLOG:
            writer.add_scalar('Train Accuracy', train_acc, epoch)
            writer.add_scalar('Val Accuracy', val_acc, epoch)
            writer.add_scalar('Test Accuracy', tmp_test_acc, epoch)
        
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')
        print(f'Train Time(s): {train_timer.duration():.4f}, Test Time(s): {test_timer.duration():.4f}')
        if PROFILING:
            prof.step()

    if PROFILING:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total"))


if __name__ == '__main__':
    main()