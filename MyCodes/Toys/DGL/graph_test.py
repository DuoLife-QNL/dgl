from typing import Tuple

import torch

from dgl.data import DGLBuiltinDataset
from dgl.data import CoraGraphDataset
from dgl.data import CiteseerGraphDataset
from dgl.data import RedditDataset
from dgl.data import PubmedGraphDataset
from dgl.data import CoauthorCSDataset
from dgl.data import WikiCSDataset
from dgl.data import FlickrDataset

from dgl import AddSelfLoop
from dgl import DGLGraph



def print_info(dataset: DGLBuiltinDataset):
    g = dataset[0]
    # print the name of the grah, number of nodes, and number of edges in graph
    print('Name of the graph: {}'.format(dataset.name))
    print('Number of nodes: {}'.format(g.number_of_nodes()))
    print('Numbor of edges: {}'.format(g.number_of_edges()))

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

def check_bi_direct(g: DGLGraph):    
    # for each edge in graph, check if there exists a reversed edge
    src_nodes, dst_nodes = g.edges()
    src_nodes: torch.Tensor
    dst_nodes: torch.Tensor

    src_nodes = src_nodes.tolist()
    dst_nodes = dst_nodes.tolist()

    for src, dst in zip(src_nodes, dst_nodes):
        if not g.has_edges_between(dst, src):
            return False
    return True

def main():
    # print the dataset infomation of Cora, Citeseer, Pubmed, Reddit
    for dataset_name in ['Cora', 'Citeseer', 'Pubmed', 'Reddit']:
        dataset = load_dataset(dataset_name)
        print_info(dataset)
        g = dataset[0]
        print('Is the graph bi-directional: {}'.format(check_bi_direct(g)))

if __name__ == '__main__':
    main()