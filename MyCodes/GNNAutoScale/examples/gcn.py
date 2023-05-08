import argparse
from math import ceil
import time
import sys
import os
import pickle
import logging
from torch_autoscale.Metric import Metric

from MyTimer import Timer
from load_reddit_dataset import load_reddit
from torch_autoscale.models import GCN, FMGCN
from torch_autoscale import compute_micro_f1
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
import gc
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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
import torch.multiprocessing as mp

# profiling tools
import torch.profiler
from torch.profiler import record_function

from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(12345)


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
                skip_first=0,
                wait=0, 
                warmup=0,
                active=2, 
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_path),
            # with_stack=True,
            with_modules=True,
            profile_memory=True, 
            record_shapes=True
        )
    return writer, global_iter, prof

class VirtualDatasetClass(List):
    def __init__(self, data, n_classes):
        self.data = data
        self.length = len(data)
        self.num_classes = n_classes
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return self.length
        

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
    elif dataset_name == 'Reddit2':
        graph, n_classes = load_reddit()
        graph = dgl.add_self_loop(graph)
        data = VirtualDatasetClass([graph], n_classes)
    elif dataset_name == 'Flickr':
        data = FlickrDataset(transform=transform)

    return data

def retrieve_dataset_info(data: DGLBuiltinDataset) -> Tuple[dgl.DGLGraph, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, g, num_layers, device, IB2OB_construction: str = 'naive', metric: Optional[Metric] = None):
        self.g = g
        self.num_layers = num_layers
        self.full_neighbor_sampler = MultiLayerFullNeighborSampler(1)
        if metric is None:
            self.metric = Metric()
        self.metric = metric
        self.device = device
        self.IB2OB_construction = IB2OB_construction
    
    def construct_subgraph_gas(self, batches: Tuple[torch.Tensor]):
        IB_nodes = torch.cat(batches, dim=0)
        self.metric.start('construct subgraph: all2IB')
        # with Metric(timer_name = "construct subgraph: all2IB"):
        subgraph = dgl.in_subgraph(self.g, IB_nodes)
        self.metric.stop('construct subgraph: all2IB')

        self.metric.start('H2D: subgraph all2IB')
        subgraph = subgraph.to(self.device)
        self.metric.stop('H2D: subgraph all2IB')
        # self.metric.start('to_block: subgraph all2IB')
        block = dgl.to_block(subgraph, dst_nodes=IB_nodes)
        # self.metric.stop('to_block: subgraph all2IB')
        block.num_nodes_in_each_batch = torch.tensor([batch.size(dim=0) for batch in batches])
        return [block for _ in range(self.num_layers)]

    def construct_subgraphs_fm(self, batches: Tuple[torch.Tensor]):
        IB_nodes = torch.cat(batches, dim=0)
        num_IB_nodes = IB_nodes.size(0)
        
        """
        construct block all2IB
        """
        self.metric.start('construct subgraph: all2IB')
        subgraph_all2IB = dgl.in_subgraph(self.g, IB_nodes)
        self.metric.stop('construct subgraph: all2IB')

        self.metric.start('H2D: subgraph_all2IB')  
        subgraph_all2IB = subgraph_all2IB.to(self.device)
        self.metric.stop('H2D: subgraph_all2IB')
        
        # self.metric.start('to_block: subgraph all2IB')
        block_all2IB = dgl.to_block(subgraph_all2IB, dst_nodes=IB_nodes)
        # self.metric.stop('to_block: subgraph all2IB')

        """
        Construct block IB2OB
        """
        block_all2IB.num_nodes_in_each_batch = torch.tensor([batch.size(dim=0) for batch in batches])
        OB_nodes = block_all2IB.srcdata[dgl.NID][num_IB_nodes:]

        self.metric.start('construct subgraph: IB2OB')
        if self.IB2OB_construction == 'naive':
            # extract a block such that the OB-nodes are the dst nodes and the IB-nodes are the source nodes from graph self.g
            subgraph_IB2all = dgl.out_subgraph(self.g, IB_nodes)
            self.metric.start('H2D: subgraph_IB2all')
            subgraph_IB2all = subgraph_IB2all.to(self.device)
            self.metric.stop('H2D: subgraph_IB2all')
            subgraph_IB2OB = dgl.in_subgraph(subgraph_IB2all, OB_nodes)
        elif self.IB2OB_construction == 'opt':
            subgraph_OB2IB = dgl.out_subgraph(subgraph_all2IB, OB_nodes)
            subgraph_IB2OB = dgl.reverse(subgraph_OB2IB)
        else:
            raise NotImplementedError
        self.metric.stop('construct subgraph: IB2OB')
        

        self.metric.start('to_block: subgraph IB2OB')
        block_IB2OB = dgl.to_block(subgraph_IB2OB, dst_nodes=OB_nodes, include_dst_in_src=False)
        self.metric.stop('to_block: subgraph IB2OB')
        return block_all2IB, block_IB2OB
    
    # def construct_subgraphs_fm_opt(self, batches: Tuple[torch.Tensor]):
    #     IB_nodes = torch.cat(batches, dim=0)
    #     num_IB_nodes = IB_nodes.size(0)

    #     self.metric.start('construct subgraph: all2IB')  
    #     subgraph_all2IB = dgl.in_subgraph(self.g, IB_nodes)
    #     self.metric.stop('construct subgraph: all2IB')
        
    #     self.metric.start('H2D: subgraph_all2IB')  
    #     subgraph_all2IB = subgraph_all2IB.to(self.device)
    #     self.metric.stop('H2D: subgraph_all2IB')

    #     self.metric.start('to_block: all2IB')
    #     block_all2IB = dgl.to_block(subgraph_all2IB, dst_nodes=IB_nodes)
    #     self.metric.stop('to_block: all2IB')

    #     block_all2IB.num_nodes_in_each_batch = torch.tensor([batch.size(dim=0) for batch in batches])
    #     OB_nodes = block_all2IB.srcdata[dgl.NID][num_IB_nodes:]

    #     self.metric.start('construct subgraph: IB2OB')
    #     subgraph_OB2IB = dgl.out_subgraph(subgraph_all2IB, OB_nodes)
    #     subgraph_IB2OB = dgl.reverse(subgraph_OB2IB)
    #     self.metric.stop('construct subgraph: IB2OB')

    #     self.metric.start('to_block: subgraph IB2OB')
    #     block_IB2OB = dgl.to_block(subgraph_IB2OB, dst_nodes=OB_nodes, include_dst_in_src=False)
    #     self.metric.stop('to_block: subgraph IB2OB')

    #     return block_all2IB, block_IB2OB
        
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

def gas_train(model: GCN, loader: TorchDataLoader, optimizer, device, acc_log: bool = False, writer: Optional[SummaryWriter] = None, global_iter: Optional[GlobalIterater] = None, grad_norm: Optional[float] = None, metric: Optional[Metric] = None):
    if metric is None:
        metric = Metric()
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    
    for it, blocks in enumerate(loader):
        block = blocks[0]
        optimizer.zero_grad()
        # metric.start('H2D: block_all2IB')
        block = block.to(device)
        # metric.stop('H2D: block_all2IB')
        metric.start('parse_arg')
        args = parse_args_from_block(block, training=True)
        metric.stop('parse_arg')
        metric.start('forward')
        out = model(block, *args)
        metric.stop('forward')
        train_mask = block.dstdata['train_mask']
        loss = criterion(out[train_mask], block.dstdata['label'][train_mask])
        # print("Iteration {} | Loss {:.4f}".format(it, loss.item()))
        # log.info("Iteration {} | Loss {:.4f}".format(it, loss.item()))

        if acc_log:
            writer.add_scalar('train_loss', loss, global_iter())
        metric.start('backward')
        loss.backward()
        metric.stop('backward')
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass

def fm_train(model: FMGCN, loader: TorchDataLoader, optimizer, device, acc_log: bool = False, writer: Optional[SummaryWriter] = None, global_iter: Optional[GlobalIterater] = None, grad_norm: Optional[float] = None, metric: Optional[Metric] = None):
    if metric is None:
        metric = Metric()
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    
    for it, (block_all2IB, block_IB2OB) in enumerate(loader):
        optimizer.zero_grad()
        # num_input_nodes = block.num_src_nodes()
        # num_output_nodes = block.num_dst_nodes()
        # num_edges = block.num_edges()
        # print("block_num_input_nodes: {}, block_num_output_nodes: {}, block_num_edges: {}".format(num_input_nodes, num_output_nodes, num_edges))
        # metric.start('H2D: block_all2IB & block_IB2OB')
        block_all2IB = block_all2IB.to(device)
        block_IB2OB = block_IB2OB.to(device)
        # metric.stop('H2D: block_all2IB & block_IB2OB')

        metric.start('parse_arg')
        feat, num_output_nodes, n_ids, offset, count = parse_args_from_block(block_all2IB, training=True)
        metric.stop('parse_arg')

        metric.start('forward')
        out = model(block_all2IB, feat, block_IB2OB, num_output_nodes, n_ids, offset, count)
        metric.stop('forward')
        train_mask = block_all2IB.dstdata['train_mask']
        loss = criterion(out[train_mask], block_all2IB.dstdata['label'][train_mask])
        # log.info("Iteration {} | Loss {:.4f}".format(it, loss.item()))

        if acc_log:
            writer.add_scalar('train_loss', loss, global_iter())
        metric.start('backward')
        loss.backward()
        metric.stop('backward')
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


def run(rank, world_size, devices: List[int], dataset_info, conf: DictConfig, prof_utils: Tuple):
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
    PART_PATH = dataset_settings.part_path

    writer, global_iter, prof = prof_utils
    log = logging.getLogger(__name__)
    metric = Metric(logger=log)

    # Hyperparameters
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

    if world_size > 1:
        # Init distributed env
        dist_init_method = "tcp://{master_ip}:{master_port}".format(
            master_ip="127.0.0.1", master_port="12345"
        )
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            init_method=dist_init_method
        )

    dev_id = devices[rank]
    torch.cuda.set_device(dev_id)
    device = f'cuda:{dev_id}'

    pid = os.getpid()
    print('pid: {}, device id: {}'.format(pid, dev_id))

    g, batches, in_size, out_size, train_mask, val_mask, labels = dataset_info
    if INFERENCE_BATCH_SIZE is None:
        num_total_nodes = g.num_nodes()
        num_partitions_per_it = BATCH_SIZE
        num_partitions = NUM_PARTS
        INFERENCE_BATCH_SIZE = num_partitions_per_it * (num_total_nodes // num_partitions)
    
    if TRAINING_METHOD == 'graphfm':
        IB2OB_CONSTRUCT = conf.training_method.ib2ob_construct
        constructor = SubgraphConstructor(g, NUM_LAYERS, device, IB2OB_construction= IB2OB_CONSTRUCT, metric=metric)
    else:
        constructor = SubgraphConstructor(g, NUM_LAYERS, device, metric=metric)
    
    loader_args = {
        'dataset': batches,
        'batch_size': BATCH_SIZE
    }
    if world_size > 1:
        dist_sampler = DistributedSampler(dataset=batches, shuffle=SHUFFLE_DATA)
        loader_args['sampler'] = dist_sampler
    else:
        loader_args['shuffle'] = SHUFFLE_DATA
    if TRAINING_METHOD == 'gas':
        loader_args['collate_fn'] = constructor.construct_subgraph_gas
    elif TRAINING_METHOD == 'graphfm':
        loader_args['collate_fn'] = constructor.construct_subgraphs_fm
    
    subgraph_loader = TorchDataLoader(**loader_args)

    best_val_acc = test_acc = 0
    
    model_args = {
        'num_nodes': g.num_nodes(),
        'in_channels': in_size,
        'hidden_channels': HIDDEN_CHANNELS,
        'out_channels': out_size,
        'num_layers': NUM_LAYERS,
        'drop_input': DROP_INPUT,
        'dropout': DROPOUT,
        'batch_norm': BATCH_NORM,
        'residual': RESIDUAL,
        'linear': LINEAR,
        'pool_size': POOL_SIZE,  # Number of pinned CPU buffers
        'buffer_size': BUFFER_SIZE,  # Size of pinned CPU buffers (max #out-of-batch nodes)
        'metric': metric
    }

    if TRAINING_METHOD == 'gas':
        model = GCN(**model_args).to(device)
    elif TRAINING_METHOD == 'graphfm':
        model = FMGCN(**model_args).to(device)

    if PROFILING:
        prof.start()

    tic = time.time()
    # Fill the history.
    layerwise_minibatch_test(model, g, out_size, device, INFERENCE_BATCH_SIZE, history_refresh=True)
    toc = time.time()
    log.info("Fill History Time(s): {:.4f}".format(toc - tic))
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[dev_id], 
            output_device=dev_id, 
            find_unused_parameters=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        optimizer = torch.optim.Adam([
            dict(params=model.reg_modules.parameters(), weight_decay=REG_WEIGHT_DECAY),
            dict(params=model.nonreg_modules.parameters(), weight_decay=NONREG_WEIGHT_DECAY)
        ], lr=LR)

    """
    Training Loop
    """
    # torch.cuda.empty_cache()
    for epoch in range(0, EPOCHS):
        train_timer = Timer()
        train_timer.start()
        metric.start('epoch')
        with record_function("2: Train"):
            log.info("Training epoch {}".format(epoch))
            if TRAINING_METHOD == 'gas':
                gas_train(model, subgraph_loader, optimizer, device, acc_log=ACCLOG, writer=writer, global_iter=global_iter, grad_norm=GRAD_NORM, metric=metric)
            elif TRAINING_METHOD == 'graphfm':
                fm_train(model, subgraph_loader, optimizer, device, acc_log=ACCLOG, writer=writer, global_iter=global_iter, grad_norm=GRAD_NORM, metric=metric)
        metric.stop('epoch')
        train_timer.end()
        if rank == 0:
            log.info("Train time (s): {:4f}".format(train_timer.duration()))
            print("Train time (s): {:4f}".format(train_timer.duration()))
        if epoch % TEST_EVERY == 0 and rank == 0:
            test_timer = Timer()
            test_timer.start()
            with record_function("3: Test"):
                train_acc, val_acc, tmp_test_acc = layerwise_minibatch_test(
                    model if world_size == 1 else model.module,
                    g, out_size, device, INFERENCE_BATCH_SIZE, history_refresh=HISTORY_REFRESH
                    )
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
            print("Test time (s): {:4f}".format(test_timer.duration()))
            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')
        # prevent from dgl cuda OOM
        # torch.cuda.empty_cache()
        if PROFILING:
            prof.step()
        if world_size > 1:
            dist.barrier()

    metric.print_metrics()
    if PROFILING:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total"))

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig):
    dataset_settings = conf._dataset
    run_env = conf.run_env
    profiling_settings = run_env.prof
    device_settings = run_env.device_settings

    METIS_PARTITION = dataset_settings.metis_partition
    PART_PATH = dataset_settings.part_path

    GPU = device_settings.device
    DATASET = dataset_settings.name
    SELF_LOOP = dataset_settings.self_loop
    NUM_PARTS = dataset_settings.num_parts
    PROFILING = profiling_settings.profiling
    ACCLOG = profiling_settings.acclog
    PROF_PATH = profiling_settings.prof_path
    PART_PATH = dataset_settings.part_path

    # check if GPU is type int 


    if isinstance(GPU, int):
        devices = [GPU]
    else:
        devices = list(map(int, GPU.split(',')))
    n_gpus = len(devices)

    data = load_dataset(DATASET, self_loop=SELF_LOOP)
    g, in_size, out_size, train_mask, val_mask, labels = retrieve_dataset_info(data)
    partition_method = 'metis' if METIS_PARTITION else 'sequential'
    g, batches = graph_partition(g, num_parts=NUM_PARTS, partition_method=partition_method, part_path=PART_PATH, dataset_name=DATASET)
    dataset_info = (g, batches, in_size, out_size, train_mask, val_mask, labels)

    writer, global_iter, prof = init_acclog_prof(ACCLOG, PROFILING, PROF_PATH)
    profiling_utils = (writer, global_iter, prof)

    g.create_formats_()
    if n_gpus == 1:
        run(0, n_gpus, devices, dataset_info, conf, profiling_utils)
    else:
        mp.spawn(run, args=(n_gpus, devices, dataset_info, conf, profiling_utils), nprocs=n_gpus)


if __name__ == '__main__':
    main()