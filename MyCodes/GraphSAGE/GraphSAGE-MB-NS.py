from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from torchmetrics.functional.classification import multiclass_f1_score
import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset, CiteseerGraphDataset, CoraGraphDataset, RedditDataset, PubmedGraphDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse

import GPUtil

from torch.profiler import profile, record_function, ProfilerActivity

from math import ceil

PROFILING = False

PROF_PATH = '/home/lihz/Codes/dgl/MyCodes/Profiling/GraphSAGE-MB-NS/SAGEConv/Default'
# Initiate profiler
on_trace_ready = None
if PROFILING:
    on_trace_ready = torch.profiler.tensorboard_trace_handler(PROF_PATH)
prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=on_trace_ready,
    record_shapes=True,
    with_stack=True,
    with_modules=True
)
prof.schedule = torch.profiler.schedule(
        skip_first=1,
        wait=1, 
        warmup=1,
        active=2, 
        repeat=2
)

torch.manual_seed(5828)
class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'gcn'))  # type: ignore
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'gcn'))  # type: ignore
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # with record_function("SAGEConv"):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    # f1_score = multiclass_f1_score(preds, labels, num_classes=num_classes, average='micro')
    targets = torch.cat(ys)
    logits = torch.cat(y_hats)
    num_classes = logits.shape[1]
    _, preds = torch.max(logits, dim=1)
    f1_score = multiclass_f1_score(preds, targets, num_classes=num_classes, average='micro')
    return f1_score.item()

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
        return MF.accuracy(pred, label)

def train(args, device, g, dataset, model):
    # create sampler & dataloader
    # With DGL Dataset
    n_nodes = g.number_of_nodes()
    seq = torch.arange(0, n_nodes)
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    train_idx = torch.masked_select(seq, train_mask)
    n_train_nodes = train_idx.size(0)
    val_idx = torch.masked_select(seq, val_mask)
    test_idx = torch.masked_select(seq, test_mask)
    # With OGB dataset
    # train_idx = dataset.train_idx.to(device)
    # val_idx = dataset.val_idx.to(device)
    # sampler = NeighborSampler([10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
    #                           prefetch_node_feats=['feat'],
    #                           prefetch_labels=['label'])
    batch_size = args.batch_size
    steps_per_epoch = ceil(n_train_nodes / batch_size)
    sampler = NeighborSampler([25, 10],  # fanout for [layer-0, layer-1]
                              prefetch_node_feats=['feat'],
                              prefetch_labels=['label'])
    # use_uva = (args.mode == 'mixed')
    use_uva = True
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1000, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)
    test_dataloader = DataLoader(g, test_idx, sampler, device=device,
                                    batch_size=1000, shuffle=True,
                                    drop_last=False, num_workers=0,
                                    use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    if PROFILING:
        prof.start()
    max_mem = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            # get the input size and output size of each block
            for block in blocks:
                block_input_node_number = block.number_of_src_nodes()
                block_output_node_number = block.number_of_dst_nodes()
                block_nedges = block.number_of_edges()
                # print("number of epoch: {}, number of iteration: {}, block_input_node_number: {}, block_output_node_number: {}, block_nedges: {}".format(epoch, it, block_input_node_number, block_output_node_number, block_nedges))
            with record_function("Forward Computation"):
                y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if PROFILING:
                prof.step()
        torch.cuda.empty_cache()
        GPUs = GPUtil.getGPUs()
        if GPUs[0].memoryUsed > max_mem:
            max_mem = GPUs[0].memoryUsed
        micro_f1 = evaluate(model, g, val_dataloader)
        test_micro_f1 = evaluate(model, g, test_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Val_Micro_F1 {:.4f} | Test_Micro_F1 {:.4f} "
              .format(epoch, total_loss / (it+1), micro_f1, test_micro_f1))
    if PROFILING:
        prof.stop()
    print("Max memory used: {}".format(max_mem))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Force executing NN Computation on GPU
    # parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
    #                     help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
    #                          "'puregpu' for pure-GPU training."
    # )
    parser.add_argument("--num-hidden", type=int, default=16, help="Size of hidden layer.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--num-epochs", type=int, default=200)
    args = parser.parse_args()
    # if not torch.cuda.is_available():
    #     args.mode = 'cpu'
    # print(f'Training in {args.mode} mode.')

    # device = torch.device("cuda")
    # dev_id = int(args.gpu)
    device = 'cuda:0'
    # torch.cuda.set_device(dev_id)
    # load and preprocess dataset
    print('Loading data')
    # dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    # dataset = CiteseerGraphDataset()
    dataset = CoraGraphDataset()
    # dataset  = PubmedGraphDataset()
    # dataset = RedditDataset(transform=AddSelfLoop())
    g = dataset[0]
    # g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    g = g.to('cpu')
    # device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, args.num_hidden, out_size).to(device)

    # model training
    print('Training...')
    train(args, device, g, dataset, model)
