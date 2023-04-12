import argparse
from math import ceil
import time
import sys
import os
import pickle
import logging

from MyTimer import Timer
from torch_autoscale.models import GCN, FMGCN
from torch_autoscale import compute_micro_f1
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import gc
from torch.utils.data import DataLoader as TorchDataLoader

from typing import List, Tuple, Optional

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


torch.manual_seed(12345)
log = logging.getLogger(__name__)


class GlobalIterater(object):
    def __init__(self):
        self.iter = 0

    def __call__(self):
        self.iter += 1
        return self.iter - 1

def init_acclog_prof(acc_log: bool = False, profiling: bool = False, prof_path: Optional[str] = None):
    writer = None
    global_iter = None
    prof = None
    if acc_log:
        writer = SummaryWriter(prof_path)
        global_iter = GlobalIterater()

    if profiling:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=1,
                wait=1, 
                warmup=1,
                active=2, 
                repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_path),
            with_stack=True,
            with_modules=True,
            profile_memory=True, 
            record_shapes=True
        )
    return writer, global_iter, prof
        

def load_dataset(dataset_name: str, self_loop: bool = False) -> DGLBuiltinDataset:
    if self_loop:
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


def graph_partition(g, num_parts, partition_method = 'metis', part_path: Optional[str] = None, dataset_name: Optional[str] = None) -> Tuple[dgl.DGLGraph, List[torch.Tensor]]:
    if partition_method == 'metis':
        store_path = os.path.join(part_path, '{}-{}-{}.pickle'.format(dataset_name, num_parts, partition_method))
        # Metis Partition
        os.makedirs(os.path.dirname(part_path), exist_ok=True)
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

    return g, batches

class SubgraphConstructor(object):
    def __init__(self, g, num_layers):
        self.g = g
        self.num_layers = num_layers
        self.full_neighbor_sampler = MultiLayerFullNeighborSampler(1)
    
    def construct_subgraph_gas(self, batches: Tuple[torch.Tensor]):
        IB_nodes = torch.cat(batches, dim=0)
        subgraph = dgl.in_subgraph(self.g, IB_nodes)
        block = dgl.to_block(subgraph, dst_nodes=IB_nodes)
        block.num_nodes_in_each_batch = torch.tensor([batch.size(dim=0) for batch in batches])
        return [block for _ in range(self.num_layers)]

    def construct_subgraphs_fm(self, batches: Tuple[torch.Tensor]):
        IB_nodes = torch.cat(batches, dim=0)
        num_IB_nodes = IB_nodes.size(0)
        _, _, blocks = self.full_neighbor_sampler.sample(self.g, IB_nodes)
        block_all2IB = blocks[0]
        OB_nodes = block_all2IB.srcdata[dgl.NID][num_IB_nodes:]
        block_all2IB.num_nodes_in_each_batch = torch.tensor([batch.size(dim=0) for batch in batches])
        # extract a block such that the OB-nodes are the dst nodes and the IB-nodes are the source nodes from graph self.g
        subgraph_IB2all = dgl.out_subgraph(self.g, IB_nodes)
        subgraph_IB2OB = dgl.in_subgraph(subgraph_IB2all, OB_nodes)
        block_IB2OB = dgl.to_block(subgraph_IB2OB, dst_nodes=OB_nodes, include_dst_in_src=False)
        return block_all2IB, block_IB2OB
        
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

def gas_train(model: GCN, loader: TorchDataLoader, optimizer, device, acc_log: bool = False, writer: Optional[SummaryWriter] = None, global_iter: Optional[GlobalIterater] = None, grad_norm: Optional[float] = None):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    
    for it, blocks in enumerate(loader):
        block = blocks[0]
        optimizer.zero_grad()
        block = block.to(device)
        args = parse_args_from_block(block, training=True)
        out = model(block, *args)
        train_mask = block.dstdata['train_mask']
        loss = criterion(out[train_mask], block.dstdata['label'][train_mask])
        # print("Iteration {} | Loss {:.4f}".format(it, loss.item()))
        log.info("Iteration {} | Loss {:.4f}".format(it, loss.item()))

        if acc_log:
            writer.add_scalar('train_loss', loss, global_iter())
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass

def fm_train(model: FMGCN, loader: TorchDataLoader, optimizer, device, acc_log: bool = False, writer: Optional[SummaryWriter] = None, global_iter: Optional[GlobalIterater] = None, grad_norm: Optional[float] = None):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    
    for it, (block_all2IB, block_IB2OB) in enumerate(loader):
        optimizer.zero_grad()
        # num_input_nodes = block.num_src_nodes()
        # num_output_nodes = block.num_dst_nodes()
        # num_edges = block.num_edges()
        # print("block_num_input_nodes: {}, block_num_output_nodes: {}, block_num_edges: {}".format(num_input_nodes, num_output_nodes, num_edges))
        block_all2IB = block_all2IB.to(device)
        block_IB2OB = block_IB2OB.to(device)
        feat, num_output_nodes, n_ids, offset, count = parse_args_from_block(block_all2IB, training=True)
        out = model(block_all2IB, feat, block_IB2OB, num_output_nodes, n_ids, offset, count)
        train_mask = block_all2IB.dstdata['train_mask']
        loss = criterion(out[train_mask], block_all2IB.dstdata['label'][train_mask])
        print("Iteration {} | Loss {:.4f}".format(it, loss.item()))
        log.info("Iteration {} | Loss {:.4f}".format(it, loss.item()))

        if acc_log:
            writer.add_scalar('train_loss', loss, global_iter())
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()


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
def layerwise_minibatch_test(model: GCN, g: dgl.DGLGraph, out_size: int, device, inference_batch_size: int, history_refresh: bool = False):
    model.eval()

    dst_nodes = torch.arange(0, g.number_of_nodes())
    block = dgl.to_block(g, dst_nodes=dst_nodes)
    feat = block.srcdata['feat']
    labels = block.srcdata['label'].cpu()
    train_mask = block.srcdata['train_mask'].cpu()
    val_mask = block.srcdata['val_mask'].cpu()
    test_mask = block.srcdata['test_mask'].cpu()
    
    
    out, _ = model.layerwise_inference(g, feat, inference_batch_size, out_size, device, history_refresh=history_refresh)

    train_acc = compute_micro_f1(out, labels, train_mask)
    val_acc = compute_micro_f1(out, labels, val_mask)
    test_acc = compute_micro_f1(out, labels, test_mask)

    return train_acc, val_acc, test_acc


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig):
    """
    Parsing Arguments
    """
    dataset_settings = conf._dataset
    run_env = conf.run_env
    profiling_settings = run_env.prof
    device_settings = run_env.device_settings
    TRAINING_METHOD: str = conf.training_method.name
    # Running Configurations
    METIS_PARTITION = dataset_settings.metis_partition
    SHUFFLE_DATA = dataset_settings.shuffle_data
    HISTORY_REFRESH = dataset_settings.history_refresh
    # Running time profiling
    PROFILING = profiling_settings.profiling
    # Accuracy convergence logging in Tensorboard
    ACCLOG = profiling_settings.acclog
    PROF_PATH = profiling_settings.prof_path
    PART_PATH = dataset_settings.part_path

    writer, global_iter, prof = init_acclog_prof(ACCLOG, PROFILING, PROF_PATH)

    # Hyperparameters
    DATASET = dataset_settings.name
    SELF_LOOP = dataset_settings.self_loop
    # GCN Norm is set to True ('both' in GraphConv) by default
    NUM_LAYERS = dataset_settings.num_layers
    HIDDEN_CHANNELS = dataset_settings.hidden_channels
    DROPOUT = dataset_settings.dropout
    NUM_PARTS = dataset_settings.num_parts
    BATCH_SIZE = dataset_settings.batch_size
    INFERENCE_BATCH_SIZE = dataset_settings.inference_batch_size
    LR = dataset_settings.lr
    REG_WEIGHT_DECAY = dataset_settings.reg_weight_decay
    NONREG_WEIGHT_DECAY = dataset_settings.nonreg_weight_decay
    GRAD_NORM = dataset_settings.grad_norm
    EPOCHS = dataset_settings.epochs
    TEST_EVERY = dataset_settings.test_every

    DROP_INPUT = dataset_settings.drop_input
    BATCH_NORM = dataset_settings.batch_norm
    RESIDUAL = dataset_settings.residual
    LINEAR = dataset_settings.linear
    POOL_SIZE = dataset_settings.pool_size
    BUFFER_SIZE = dataset_settings.buffer_size

    device = f'cuda:{device_settings.device}' if torch.cuda.is_available() else 'cpu'

    data = load_dataset(DATASET, self_loop=SELF_LOOP)
    g, in_size, out_size, train_mask, val_mask, labels = retrieve_dataset_info(data)

    if INFERENCE_BATCH_SIZE is None:
        num_total_nodes = g.num_nodes()
        num_partitions_per_it = BATCH_SIZE
        num_partitions = NUM_PARTS
        INFERENCE_BATCH_SIZE = num_partitions_per_it * (num_total_nodes // num_partitions)
    
    partition_method = 'metis' if METIS_PARTITION else 'sequential'
    g, batches = graph_partition(g, num_parts=NUM_PARTS, partition_method=partition_method, part_path=PART_PATH, dataset_name=DATASET)

    constructor = SubgraphConstructor(g, NUM_LAYERS)
    if TRAINING_METHOD == 'gas':
        subgraph_loader = TorchDataLoader(
            dataset = batches,
            batch_size=BATCH_SIZE,
            collate_fn = constructor.construct_subgraph_gas,
            shuffle=SHUFFLE_DATA
        )
    elif TRAINING_METHOD == 'graphfm':
        subgraph_loader = TorchDataLoader(
            dataset = batches,
            batch_size=BATCH_SIZE,
            collate_fn = constructor.construct_subgraphs_fm,
            shuffle=SHUFFLE_DATA
        )

    best_val_acc = test_acc = 0

    if TRAINING_METHOD == 'gas':
        model = GCN(
            num_nodes=g.num_nodes(),
            in_channels=in_size,
            hidden_channels=HIDDEN_CHANNELS,
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
    elif TRAINING_METHOD == 'graphfm':
        model = FMGCN(
            num_nodes=g.num_nodes(),
            in_channels=in_size,
            hidden_channels=HIDDEN_CHANNELS,
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

    tic = time.time()
    # test(model, g, device) # Fill the history.
    layerwise_minibatch_test(model, g, out_size, device, INFERENCE_BATCH_SIZE, history_refresh=True)
    toc = time.time()
    print("Fill History Time(s): {:.4f}".format(toc - tic))

    optimizer = torch.optim.Adam([
        dict(params=model.reg_modules.parameters(), weight_decay=REG_WEIGHT_DECAY),
        dict(params=model.nonreg_modules.parameters(), weight_decay=NONREG_WEIGHT_DECAY)
    ], lr=LR)
    for epoch in range(0, EPOCHS):
        train_timer = Timer()
        train_timer.start()
        with record_function("2: Train"):
            print("Training epoch {}".format(epoch))
            if TRAINING_METHOD == 'gas':
                gas_train(model, subgraph_loader, optimizer, device, acc_log=ACCLOG, writer=writer, global_iter=global_iter, grad_norm=GRAD_NORM)
            elif TRAINING_METHOD == 'graphfm':
                fm_train(model, subgraph_loader, optimizer, device, acc_log=ACCLOG, writer=writer, global_iter=global_iter, grad_norm=GRAD_NORM)
        train_timer.end()
        log.info("Train time (s): {:4f}".format(train_timer.duration()))
        if epoch % TEST_EVERY == 0:
            test_timer = Timer()
            test_timer.start()
            with record_function("3: Test"):
                train_acc, val_acc, tmp_test_acc = layerwise_minibatch_test(model, g, out_size, device, INFERENCE_BATCH_SIZE, history_refresh=HISTORY_REFRESH)
                # train_acc, val_acc, tmp_test_acc = test(model, g, device)
            test_timer.end()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            if ACCLOG:
                writer.add_scalar('Train Accuracy', train_acc, epoch)
                writer.add_scalar('Val Accuracy', val_acc, epoch)
                writer.add_scalar('Test Accuracy', tmp_test_acc, epoch)
            
            log.info("Test time (s): {:4f}".format(test_timer.duration()))
            log.info(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')
        # prevent from dgl cuda OOM
        torch.cuda.empty_cache()
        if PROFILING:
            prof.step()

    if PROFILING:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total"))


if __name__ == '__main__':
    main()