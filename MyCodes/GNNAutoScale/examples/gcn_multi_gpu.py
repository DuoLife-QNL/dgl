import argparse
from math import ceil
import time
import sys
import torch

from torch_autoscale.models import GCN
from torch_autoscale import compute_micro_f1

from torch.utils.data import DataLoader

from typing import Tuple

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CoraGraphDataset
import pdb

from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler

# profiling tools
import torch.profiler
from torch.profiler import record_function

from torch.utils.tensorboard import SummaryWriter

METIS_PARTITION = False
DEBUG = False
SHULFFLE_DATA = False

writer = SummaryWriter('/home/lihz/Codes/dgl/MyCodes/Profiling/GNNAutoScale/tensorboard/dgl-multi-gpu')
torch.manual_seed(12345)
class GlobalIterater(object):
    def __init__(self):
        self.iter = 0

    def __call__(self):
        self.iter += 1
        return self.iter - 1

global_iter = GlobalIterater()


prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/lihz/Codes/dgl/MyCodes/Profiling/GNNAutoScale/tensorboard/dgl'),
    record_shapes=True,
    with_stack=True,
    with_modules=True
)
prof.schedule = torch.profiler.schedule(
        skip_first=2,
        wait=2, 
        warmup=2,
        active=2, 
        repeat=2
        )


class SubgraphConstructor(object):
    def __init__(self, g, num_layers):
        self.g = g
        self.num_layers = num_layers
    
    def construct_subgraph(self, batches: Tuple[torch.Tensor]):
        IB_nodes = torch.cat(batches, dim=0)
        subgraph = dgl.in_subgraph(self.g, IB_nodes)
        block = dgl.to_block(subgraph)
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

criterion = torch.nn.CrossEntropyLoss()
def train(model: GCN, loader, optimizer):
    model.train()

    for it, blocks in enumerate(loader):
        block = blocks[0].to(model.device)
        out = model(block, *parse_args_from_block(block, training=True))
        train_mask = block.dstdata['train_mask']
        loss = criterion(out[train_mask], block.dstdata['label'][train_mask])
        writer.add_scalar('train_loss', loss, global_iter())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

@torch.no_grad()
def test(model: GCN, g: dgl.DGLGraph):
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



def main():
    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    data = CoraGraphDataset(transform=transform)
    g = data[0]
    in_size = g.ndata['feat'].shape[1]
    out_size = data.num_classes
    train_mask = g.ndata['train_mask']
    train_ids = train_mask.nonzero().squeeze()
    val_mask = g.ndata["val_mask"]
    labels = g.ndata["label"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--profiling', type=bool, default=False)
    args = parser.parse_args()
    PROFILING  = args.profiling
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'


    # 由metis或者顺序划分得到多个节点区间后，一次训练选择多少个节点区间
    num_partitions_per_it = 1

    num_layers = 3

    if METIS_PARTITION:
        # Metis Partition
        num_parts = 500
        part_dict = dgl.metis_partition(g, num_parts, reshuffle=True, 
        # balance_ntypes=g.ndata['train_mask']
        )
        batches = [part_dict[i].ndata[dgl.NID] for i in range(num_parts)]
        original_ids = [part_dict[i].ndata['orig_id'] for i in range(num_parts)]
        new_order = torch.cat(original_ids, dim=0)
        subgraph_list = [_ for _ in part_dict.values()]
        g = dgl.reorder_graph(g, node_permute_algo='custom', permute_config={'nodes_perm': new_order})


    else:
        # Sequantial Partition
        num_nodes = g.num_nodes()
        # 每个partition内部的节点数量
        num_nodes_per_part = 2
        id_seq = torch.arange(num_nodes)
        # batches是一个batch list，每一个batch相当于一个partition
        batches = torch.split(id_seq, num_nodes_per_part)

    num_parts = len(batches)
    print("num_parts: {}".format(num_parts))


    constructor = SubgraphConstructor(g, num_layers)
    subgraph_loader = DataLoader(
    dataset = batches,
    batch_size=num_partitions_per_it,
    collate_fn = constructor.construct_subgraph,
    shuffle=SHULFFLE_DATA
    )

    best_val_acc = test_acc = 0
    # Make use of the pre-defined GCN+GAS model:
    model = GCN(
        num_nodes=g.num_nodes(),
        in_channels=in_size,
        hidden_channels=16,
        out_channels=out_size,
        num_layers=num_layers,
        dropout=0.5,
        drop_input=True,
        pool_size=1,  # Number of pinned CPU buffers
        buffer_size=2000,  # Size of pinned CPU buffers (max #out-of-batch nodes)
    ).to(device)

    if PROFILING:
        prof.start()

    tic = time.time()
    test(model, g) # Fill the history.
    toc = time.time()
    print("Fill History Time(s): {:.4f}".format(toc - tic))

    optimizer = torch.optim.Adam([
        dict(params=model.reg_modules.parameters(), weight_decay=5e-4),
        dict(params=model.nonreg_modules.parameters(), weight_decay=0)
    ], lr=0.01)
    for epoch in range(0, 200):
        with record_function("2: Train"):
            train(model, subgraph_loader, optimizer)
        with record_function("3: Test"):
            train_acc, val_acc, tmp_test_acc = test(model, g)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        writer.add_scalar('Train Accuracy', train_acc, epoch)
        writer.add_scalar('Val Accuracy', val_acc, epoch)
        writer.add_scalar('Test Accuracy', tmp_test_acc, epoch)
        
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')
        if PROFILING:
            prof.step()

    if PROFILING:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total"))


if __name__ == '__main__':
    main()