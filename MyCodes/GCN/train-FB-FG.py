import argparse
import profile

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional.classification import multiclass_f1_score

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

PROFILING = False
DATASET = 'pubmed'
SELF_LOOP = False
NUM_LAYERS = 2
HIDDEN_CHANNELS = 16
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 0.0
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

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        num_classes = logits.shape[1]
        targets = labels[mask]
        _, preds = torch.max(logits, dim=1)
        f1_score = multiclass_f1_score(preds, targets, num_classes=num_classes, average='micro')
        return f1_score.item()

def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
    for epoch in range(EPOCHS):
        model.train()
        if PROFILING:
            with torch.profiler.record_function("Forward Computation"):
                logits = model(g, features)
        else:
            logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        if PROFILING:
            prof.step()
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
    g = data[0]
    print(type(g))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(2)
    g = g.int().to(device)
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
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
    