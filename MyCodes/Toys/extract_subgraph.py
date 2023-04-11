import dgl
import torch

g = dgl.graph((torch.tensor([0, 0, 0, 3, 1, 1]), torch.tensor([1, 0, 3, 2, 3, 2])))
sg_2_dst = dgl.in_subgraph(g, [2, 3])
print(sg_2_dst.edges())
sg_from_src = dgl.out_subgraph(sg_2_dst, [0, 1])
print(sg_from_src.edges())
# ret = dgl.to_block(sg_2_dst, src_nodes=[0, 1])
# ret = dgl.to_block(g, src_nodes=[0, 1], dst_nodes=[2, 3])
# print(ret.srcdata[dgl.NID])
# print(ret.dstdata[dgl.NID])
# print(g.edges())
# print(ret.edges())
