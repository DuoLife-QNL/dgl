import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset, CiteseerGraphDataset, CoraGraphDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse

from torch.profiler import profile, record_function, ProfilerActivity

from math import ceil

# Initiate profiler
prof = profile(
    activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    record_shapes=True
)

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))  # type: ignore
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))  # type: ignore
        # self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
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
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))

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
    train_idx = torch.masked_select(seq, train_mask)
    n_train_nodes = train_idx.size(0)
    val_idx = torch.masked_select(seq, val_mask)
    # With OGB dataset
    # train_idx = dataset.train_idx.to(device)
    # val_idx = dataset.val_idx.to(device)
    # sampler = NeighborSampler([10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
    #                           prefetch_node_feats=['feat'],
    #                           prefetch_labels=['label'])
    batch_size = args.batch_size
    steps_per_epoch = ceil(n_train_nodes / batch_size)
    # prof.schedule = torch.profiler.schedule(
    #     wait=steps_per_epoch, warmup=steps_per_epoch,
    #     active=10*steps_per_epoch, repeat=1)
    sampler = NeighborSampler([25, 10],  # fanout for [layer-0, layer-1]
                              prefetch_node_feats=['feat'],
                              prefetch_labels=['label'])
    # use_uva = (args.mode == 'mixed')
    use_uva = True
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva, profiling=True)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1000, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    prof.start()
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            with record_function("Forward Computation"):
                y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # prof.step()
        acc = evaluate(model, g, val_dataloader)

        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))
    prof.stop()
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Force executing NN Computation on GPU
    # parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
    #                     help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
    #                          "'puregpu' for pure-GPU training."
    # )
    parser.add_argument("--num-hidden", type=int, default=16, help="Size of hidden layer.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--num-epochs", type=int, default=200)
    args = parser.parse_args()
    # if not torch.cuda.is_available():
    #     args.mode = 'cpu'
    # print(f'Training in {args.mode} mode.')

    device = torch.device("cuda")
    dev_id = int(args.gpu)
    torch.cuda.set_device(dev_id)
    # load and preprocess dataset
    print('Loading data')
    # dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    # dataset = CiteseerGraphDataset(transform=AddSelfLoop())
    dataset = CoraGraphDataset(transform=AddSelfLoop())
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

    # test the model
    # print('Testing...')
    # acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    # print("Test Accuracy {:.4f}".format(acc.item()))
