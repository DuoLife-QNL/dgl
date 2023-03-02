import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_f1_score

import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from torch.profiler import profile, record_function, ProfilerActivity

prof = profile(
    activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    record_shapes=True,
    with_stack=True
)

# set pytorch seed to 1234
torch.manual_seed(1234)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/home/lihz/Codes/dgl/MyCodes/Profiling/GraphSAGE/tensorboard/FB_FG')
class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def evaluate_micro_F1(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        num_classes = logits.shape[1]
        _, preds = torch.max(logits, dim=1)
        f1_score = multiclass_f1_score(preds, labels, num_classes=num_classes, average='micro')
        return f1_score.item()


def train(args, g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask, val_mask = masks
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    prof.schedule = torch.profiler.schedule(
        skip_first=10,
        wait=2, warmup=2,
        active=1, repeat=2)
    # prof.start()
    # training loop
    for epoch in range(args.num_epochs):
        model.train()
        with record_function("CUSTOM: Forward Computation"):
            logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        micro_f1 = evaluate_micro_F1(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | micro_f1 {:.4f} ".format(
                epoch, loss.item(), micro_f1
            )
        )
        writer.add_scalar('Test Accuracy', micro_f1, epoch)
        # prof.step()
    # prof.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed')",
    )
    parser.add_argument("--num-hidden", type=int, default=16, help="Size of hidden layer.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num-epochs", type=int, default=200)
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphSage module")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_id = int(args.gpu)
    torch.cuda.set_device(dev_id)
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"]

    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = SAGE(in_size, args.num_hidden, out_size).to(device)

    # model training
    print("Training...")
    train(args, g, features, labels, masks, model)

    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total"))
    # prof.export_chrome_trace("/home/lihz/Codes/dgl/MyCodes/Profiling/GraphSAGE/GraphSAGE-FB-FG-trace.json")

    # test the model
    # print("Testing...")
    # acc = evaluate(g, features, labels, g.ndata["test_mask"], model)
    # print("Test accuracy {:.4f}".format(acc))
