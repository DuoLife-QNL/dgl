from typing import Optional

import time

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
import dgl
import dgl.nn as dglnn
from torch_autoscale.models import ScalableGNN, FMGNN
from torch_autoscale import FMHistory
from dgl.heterograph import DGLBlock


class FMGCN(FMGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, linear: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual
        self.linear = linear

        self.lins = ModuleList()
        if linear:
            self.lins.append(Linear(in_channels, hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0 and not linear:
                in_dim = in_channels
            if i == num_layers - 1 and not linear:
                out_dim = out_channels
            conv = dglnn.GraphConv(in_dim, out_dim)
            # conv = dglnn.SAGEConv(in_dim, out_dim, aggregator_type='gcn')
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)

    @property
    def reg_modules(self):
        if self.linear:
            return ModuleList(list(self.convs) + list(self.bns))
        else:
            return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins if self.linear else self.convs[-1:]

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    @torch.no_grad()
    def extract_feat_IB2OB(self, feat_all: Tensor, block_2IB: DGLBlock, block_IB2OB: DGLBlock) -> Tensor:
        # the feat tensor of all IB nodes
        feat_allIB = feat_all[:block_2IB.num_dst_nodes()]
        # the IDs of the IB nodes which should be aggregated to the OB nodes
        IBs_used_for_OB = block_IB2OB.srcdata[dgl.NID]
        # the IDs of all IB nodes
        IBs_all = block_2IB.dstdata[dgl.NID]
        selected_indices = []

        # initiate a bucket with the size of number of nodes, each element is None
        bucket = [None] * self.num_nodes

        for i, nid in enumerate(IBs_all):
            bucket[nid.item()] = i
        for nid in IBs_used_for_OB:
            selected_indices.append(bucket[nid.item()])
        
        # i = j = 0
        # while j < IBs_used_for_OB.size(0):
        #     if IBs_used_for_OB[j].item() == IBs_all[i].item():
        #         selected_indices.append(i)
        #         i += 1
        #         j += 1
        #     else:
        #         assert(IBs_used_for_OB[j].item() > IBs_all[i].item())
        #         i += 1
        # selected_indices = torch.tensor(selected_indices, dtype=int, device=feat_all.device)
        # select the feat of nodes in IB used for aggregation to OB
        feat_IB_used_for_OB = feat_allIB[selected_indices]

        return feat_IB_used_for_OB

    def forward(self, block_2IB: DGLBlock, feat: Tensor, block_IB2OB: DGLBlock, *args) -> Tensor:
        """
        args:
            block_2IB: IB + OB nodes as the src nodes, IB as the dst nodes
            block_IB2OB: IB as the src nodes, OB as the dst nodes
        """
        if self.drop_input:
            feat = F.dropout(feat, p=self.dropout, training=self.training)

        if self.linear:
            feat = self.lins[0](feat).relu_()
            feat = F.dropout(feat, p=self.dropout, training=self.training)

        for num_layer, conv, bn, hist in zip(range(self.num_layers), self.convs[:-1], self.bns, self.histories):
            hist: FMHistory
            """
            Prepare the feature of those IB nodes who are used for the aggregation to OB nodes.
            """
            feat_IB2OB =  self.extract_feat_IB2OB(feat, block_2IB, block_IB2OB).detach()

            h = conv(block_2IB, feat)
            if self.batch_norm:
                h = bn(h)
            if self.residual and h.size(-1) == feat.size(-1):
                h += feat[:h.size(0)]
            feat = h.relu_()

            """
            Compute and update the OB nodes embeddings for next layer, using the embeddings of IB nodes computed in current layer. This is the core operation of GraphFM.
            """
            feat_IB2OB = conv(block_IB2OB, feat_IB2OB)
            if self.batch_norm:
                bn.eval()
                feat_IB2OB = bn(feat_IB2OB)
                bn.train()
            feat_IB2OB = feat_IB2OB.relu_()
            hist_device = hist.emb.device
            hist.FM_and_hist_update(feat_IB2OB, block_IB2OB.dstdata[dgl.NID].to(hist_device))

            # tic = time.time()
            feat = self.push_and_pull(hist, feat, *args)
            # toc = time.time() 
            # print("pull and push time for layer {}: {:4f}".format(num_layer, toc - tic))
            feat = F.dropout(feat, p=self.dropout, training=self.training)
        
        h = self.convs[-1](block_2IB, feat)

        if not self.linear:
            return h

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual and h.size(-1) == feat.size(-1):
            h += feat[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[1](h)

    @torch.no_grad()
    def forward_layer(self, layer_number: int, block: DGLBlock, feat: Tensor):
        if layer_number == 0:
            if self.drop_input:
                feat = F.dropout(feat, p=self.dropout, training=self.training)
            if self.linear:
                feat = self.lins[0](feat).relu_()
                feat = F.dropout(feat, p=self.dropout, training=self.training)
        else:
            feat = F.dropout(feat, p=self.dropout, training=self.training)
        
        h = self.convs[layer_number](block, feat)

        if layer_number < self.num_layers - 1 or self.linear:
            if self.batch_norm:
                h = self.bns[layer_number](h)
            if self.residual and h.size(-1) == feat.size(-1):
                h += feat[:h.size(0)]
            h = h.relu_()

        if self.linear:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lins[1](h)

        return h
