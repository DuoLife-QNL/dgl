from typing import Optional, Tuple

import time

import torch
from torch import Tensor
from torch.utils.data import RandomSampler

import dgl
from dgl import AddSelfLoop, DGLGraph, in_subgraph
from dgl.data import RedditDataset
from dgl.heterograph import DGLHeteroGraph
from dgl.sampling import sample_neighbors

import pandas as pd
import matplotlib.pyplot as plt



def sample_nodes(g: DGLGraph, num_dst_nodes: int, sample_all_neighbors: bool = True, num_neighbors: Optional[int] = None) -> Tuple[DGLHeteroGraph, Tensor]:
    r"""
    Sample all neighbors from a seed node if num_src_nodes is not given.
    """
    # sample num_dst_nodes seed nodes first
    n_nodes = g.number_of_nodes()
    # sampling_weights = torch.ones(n_nodes)
    # # sample seed nodes without replacement
    # dst_nodes = torch.multinomial(sampling_weights, num_dst_nodes, replacement=False)

    nodes = torch.arange(n_nodes)
    dst_nodes = RandomSampler(nodes, replacement=False, num_samples=num_dst_nodes)
    dst_nodes = torch.tensor(list(dst_nodes))

    subgraph = None
    if sample_all_neighbors:
        subgraph = in_subgraph(g, dst_nodes)
    else:
        assert(num_neighbors is not None)
        subgraph = sample_neighbors(g, dst_nodes, num_neighbors)

    return subgraph, dst_nodes

def test_to_block_time(g: DGLHeteroGraph, dst_nodes: Tensor, prefix: str):
    tic = time.time()
    block = dgl.to_block(g, dst_nodes)
    toc = time.time()
    num_src_nodes = block.number_of_src_nodes()
    num_dst_nodes = block.number_of_dst_nodes()
    nedges = block.number_of_edges()
    print("============================================")
    print("Given info: {}".format(prefix))
    print("Block info: num_src_nodes: {}, num_dst_nodes: {}, num_edges: {}".format(num_src_nodes, num_dst_nodes, nedges))
    duration = toc - tic
    print("to_block time : {:4f}".format(duration))
    print("============================================")
    del block
    return nedges, duration

def one_test(prefix: str, g: DGLGraph, num_dst_nodes: int, sample_all_neighbors: bool = True, num_neighbors: Optional[int] = None, pass_dst_nodes = True):
    # construct the subgraph with dst nodes only
    subgraph, dst_nodes = sample_nodes(g, num_dst_nodes, sample_all_neighbors, num_neighbors)
    # run the test
    if not pass_dst_nodes:
        dst_nodes = None
    prefix = prefix + ":: " "num_dst_nodes: {}, pass_dst_nodes: {}, sample_all_neighbors: {}, num_neighbors: {}".format(num_dst_nodes, pass_dst_nodes, sample_all_neighbors, num_neighbors)
    block_nedges, duration = test_to_block_time(subgraph, dst_nodes, prefix)
    g_nedges = subgraph.number_of_edges()
    return g_nedges, block_nedges, duration

def basic_test(g: DGLGraph, num_dst_nodes: int, repeat: int = 1):
    # test mini-batch sampler
    for i in range(0, repeat):
        one_test("NS", g, num_dst_nodes, False, 10, True)
    # one_test("NS", g, num_dst_nodes, False, 10, False)

    # test dgl-gas
    # one_test("dgl-gas", g, num_dst_nodes, True, pass_dst_nodes=True)
    # one_test("dgl-gas", g, num_dst_nodes, True, pass_dst_nodes=False)

def gas_batch_test(g:DGLGraph, num_dst_nodes_low: int, num_dst_nodes_high: int, step: int):
    g_nedges_list = []
    block_nedges_list = []
    durations = []
    for num_dst_nodes in range(num_dst_nodes_low, num_dst_nodes_high, step):
        g_nedges, block_nedges, duration = one_test("dgl-gas", g, num_dst_nodes, True, pass_dst_nodes=True)
        g_nedges_list.append(g_nedges)
        block_nedges_list.append(block_nedges)
        durations.append(duration)
    # output g_nedges_list, block_nedges_list, durations to csv file
    df = pd.DataFrame({'g_nedges': g_nedges_list, 'block_nedges': block_nedges_list, 'durations': durations})
    df.to_csv("gas_batch_test.csv")
    # draw two plots, the first is the g_nedges vs durations, the second is the block_nedges vs durations
    # draw the first plot
    x = g_nedges_list
    y = durations
    plt.plot(x, y)
    plt.xlabel('g_nedges')
    plt.ylabel('durations')
    plt.title('g_nedges vs durations')
    plt.savefig('gas_batch_test_g_nedges_vs_durations.png')
    plt.close()
    # draw the second plot
    x = block_nedges_list
    y = durations
    plt.plot(x, y)
    plt.xlabel('block_nedges')
    plt.ylabel('durations')
    plt.title('block_nedges vs durations')
    plt.savefig('gas_batch_test_block_nedges_vs_durations.png')
    plt.close()


def main():
    # load reddit dataset
    dataset = RedditDataset(transform=AddSelfLoop())
    g = dataset[0]
    
    basic_test(g, 58000, repeat=100)

    # gas_batch_test(g, 1000, 230000, 1000)


if __name__ == '__main__':
    main()