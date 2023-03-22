from dgl.data import DGLBuiltinDataset

from dgl.data import CoraGraphDataset
from dgl.data import CiteseerGraphDataset
from dgl.data import PubmedGraphDataset
from dgl.data import FlickrDataset
from dgl.data import RedditDataset

from ogb.nodeproppred import DglNodePropPredDataset

def retrieve_dataset_info(data: DGLBuiltinDataset):
    g = data[0]
    # get total number of nodes of graph g
    num_nodes = g.number_of_nodes()
    train_mask = g.ndata['train_mask']
    # train_ids = train_mask.nonzero().squeeze()
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    # get number of training, validation, and test nodes
    num_train = train_mask.int().sum().item()
    num_val = val_mask.int().sum().item()
    num_test = test_mask.int().sum().item()
    return num_nodes, num_train, num_val, num_test

def retrieve_dataset_info_ogbn(datasetname: str):
    dataset = DglNodePropPredDataset(name = datasetname)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    num_nodes = graph.number_of_nodes()
    num_train = train_idx.shape[0]
    num_val = valid_idx.shape[0]
    num_test = test_idx.shape[0]
    return num_nodes, num_train, num_val, num_test

def print_dataset_info(datasetname: str, num_nodes: int, num_train: int, num_val: int, num_test: int):
    print("======= {} =======".format(datasetname))
    print("Total number of nodes: ", num_nodes)
    print("Total number of training nodes: ", num_train)
    print("Total number of validation nodes: ", num_val)
    print("Total number of test nodes: ", num_test)
    # calculate the percentage of training nodes
    print("Percentage of training nodes: {:.2f}%".format(num_train / num_nodes * 100))


def main():

    # The DGL Builtin Dataset
    for dataset in (
        CoraGraphDataset(), 
        CiteseerGraphDataset(), 
        PubmedGraphDataset(), 
        FlickrDataset(), 
        RedditDataset()
    ):
        num_nodes, num_train, num_val, num_test = retrieve_dataset_info(dataset)
        print_dataset_info(dataset.name, num_nodes, num_train, num_val, num_test)

    # The OGBN datasets
    for dataset_name in (
        "ogbn-products",
        'ogbn-arxiv',
        # 'ogbn-papers100M',
        # 'ogbn-mag'
    ):
        num_nodes, num_train, num_val, num_test = retrieve_dataset_info_ogbn(dataset_name)
        print_dataset_info(dataset_name, num_nodes, num_train, num_val, num_test)


if   __name__   ==   '__main__' : 
    main ()
