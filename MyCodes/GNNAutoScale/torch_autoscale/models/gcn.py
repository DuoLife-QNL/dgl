from typing import Optional

import time

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
import dgl
import dgl.nn as dglnn
from torch_autoscale.models import ScalableGNN, GASGNN
from dgl.heterograph import DGLBlock


class GCN(GASGNN):
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


    def forward(self, block_2IB: DGLBlock, feat: Tensor, *args) -> Tensor:
        if self.drop_input:
            feat = F.dropout(feat, p=self.dropout, training=self.training)

        if self.linear:
            feat = self.lins[0](feat).relu_()
            feat = F.dropout(feat, p=self.dropout, training=self.training)

        for num_layer, conv, bn, hist in zip(range(self.num_layers), self.convs[:-1], self.bns, self.histories):
            h = conv(block_2IB, feat)
            if self.batch_norm:
                h = bn(h)
            if self.residual and h.size(-1) == feat.size(-1):
                h += feat[:h.size(0)]
            feat = h.relu_()
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
