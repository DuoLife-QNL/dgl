import argparse
import profile

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, FlickrDataset, RedditDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler

from torchmetrics.functional.classification import multiclass_f1_score

PROFILING = False
DATASET = 'reddit'
SELF_LOOP = True
NUM_LAYERS = 2
HIDDEN_CHANNELS = 256
DROPOUT = 0.5
BATCH_SIZE = 10
LR = 0.01
WEIGHT_DECAY = 0.0

DROP_INPUT = True
EPOCHS = 400


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(NUM_LAYERS - 1):
            self.layers.append(
                dglnn.GraphConv(in_size, hid_size, activation=F.relu)
            )
            in_size = hid_size
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, blocks, features):
        h = features
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            if i == 0 and DROP_INPUT:
                h = self.dropout(h)
            else:
                h = self.dropout(h)
            h = layer(block, h)
            h = F.relu(h)
        return h


def evaluate(model, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    targets = torch.cat(ys)
    logits = torch.cat(y_hats)
    num_classes = logits.shape[1]
    _, preds = torch.max(logits, dim=1)
    f1_score = multiclass_f1_score(preds, targets, num_classes=num_classes, average='micro')
    return f1_score.item()



def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    # get training node indices from train mask
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    # get validation node indices from val mask
    val_mask = masks[1]
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
    # get test node indices from test mask
    test_mask = masks[2]
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batch_size = BATCH_SIZE
    sampler = MultiLayerFullNeighborSampler(NUM_LAYERS, prefetch_node_feats=['feat'], prefetch_labels=['label'])
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=0)
    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=batch_size, shuffle=True,
                                drop_last=False, num_workers=0)
    test_dataloader = DataLoader(g, test_idx, sampler, device=device,
                                 batch_size=batch_size, shuffle=True,
                                 drop_last=False, num_workers=0)

    # training loop
    prof = None
    if PROFILING:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/gcn'),
            record_shapes=True,
            with_stack=True,
            # profile_memory=True
        )
        prof.start()
    best_val_acc = 0
    cor_test_acc = 0
    for epoch in range(EPOCHS):
        
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = evaluate(model, val_dataloader)
        test_acc = evaluate(model, test_dataloader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            cor_test_acc = test_acc
        print("Epoch {:05d} | Loss {:.4f} | Val_Acc {:.4f} | Test_Acc {:.4f}"
              .format(epoch, total_loss / (it+1), val_acc, test_acc))
        if PROFILING:
            prof.step()
    print("Best Val Acc {:.4f} | Cor Test Acc {:.4f}".format(best_val_acc, cor_test_acc))
    if PROFILING:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="citeseer",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    transform = None
    if SELF_LOOP:
        transform = (
            AddSelfLoop()
        )  # by default, it will first remove self-loops to prevent duplication
    if DATASET == "cora":
        data = CoraGraphDataset(transform=transform)
    elif DATASET == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif DATASET == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    elif DATASET == 'flickr':
        data = FlickrDataset(transform=transform)
    elif DATASET == 'reddit':
        data = RedditDataset(transform=transform)
    g = data[0]
    print(type(g))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)
    g = g.to(device)
    # g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5).to(device)
    norm[torch.isinf(norm)] = 0
    g.ndata["norm"] = norm.unsqueeze(1)

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, HIDDEN_CHANNELS, out_size).to(device)
    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    # print("Testing...")
    # acc = evaluate(g, features, labels, masks[2], model)
    # print("Test accuracy {:.4f}".format(acc))
    