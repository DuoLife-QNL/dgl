from typing import Optional, Tuple

import torch
from torch import Tensor

from dgl import DGLGraph

from torch_autoscale import Metric

class GPUCache(object):
    def __init__(self, embedding_dim: int, g: DGLGraph, cache_size: int, device) -> None:
        """
        Find top cache_size nodes with highest degrees in graph g, and cache their embeddings on GPU.
        """
        self.cache_size = cache_size
        self.device = device
        self.embedding_dim = embedding_dim
        self.g = g
        self.cache = torch.empty((cache_size, embedding_dim), device=self.device)
        # Sort nodes by degree in descending order, and get the node id of the top cache_size nodes
        self.cached_nodes_ids_cpu = torch.argsort(self.g.in_degrees(), descending=True)[:self.cache_size]
        self.cached_nodes_ids = self.cached_nodes_ids_cpu.to(device)
        # Create a lookup table, for a given node id, get the index of the node in the cache
        self.cached_nodes_ids_lookup = torch.zeros(
            self.g.number_of_nodes(), dtype=torch.int64, device=self.device
        ).scatter_(0, self.cached_nodes_ids, torch.arange(self.cache_size, device=self.device))
        # Create a mask for cached nodes
        self.cached_node_mask = torch.zeros(
                self.g.number_of_nodes(), dtype=torch.bool, device=self.device
            ).scatter_(0, self.cached_nodes_ids, True).to(device)
        
    def get_in_cache_mask(self, n_ids: Tensor) -> Tensor:
        return self.cached_node_mask[n_ids]
    
    def split_by_device(self, n_ids: Tensor, x: Tensor) -> list:
        """
        Split the input tensor n_ids and x into two tensors, one for cached nodes and the other for uncached nodes.
        """
        push2GPU_mask = self.get_in_cache_mask(n_ids)
        push2CPU_mask = ~push2GPU_mask
        n_ids_push2GPU = n_ids[push2GPU_mask]
        n_ids_push2CPU = n_ids[push2CPU_mask]
        x_push2GPU = x[push2GPU_mask]
        x_push2CPU = x[push2CPU_mask]
        return [(n_ids_push2GPU, x_push2GPU), (n_ids_push2CPU, x_push2CPU)]
    
    def push_by_n_ids(self, n_ids: Tensor, x: Tensor):
        """
        Push embeddings to GPU cache for nodes with id in n_ids.
        Return the nodes ids and x needed to push to CPU cache
        """
        push_info = self.split_by_device(n_ids, x)
        n_ids_push2GPU, x_push2GPU = push_info[0]
        n_ids_push2CPU, x_push2CPU = push_info[1]
        self.cache[self.cached_nodes_ids_lookup[n_ids_push2GPU]] = x_push2GPU
        return n_ids_push2CPU, x_push2CPU
    
    def pull_by_n_ids(self, n_ids: Tensor):
        """
        Pull embeddings from GPU cache for nodes with id in n_ids.
        """
        in_cache_mask = self.get_in_cache_mask(n_ids)
        n_ids_in_cache = n_ids[in_cache_mask]
        x_in_cache = self.cache[self.cached_nodes_ids_lookup[n_ids_in_cache]]
        return in_cache_mask, x_in_cache
        
    
    def init_GPU_cache(self, x: Tensor):
        """
        Initialize GPU cache. X is the embedding of all nodes.
        """
        x_GPU_cache = x[self.cached_node_ids_cpu]
        self.cache = x_GPU_cache.to(self.device)

         

class History(torch.nn.Module):
    r"""A historical embedding storage module."""
    def __init__(
            self, num_embeddings: int, embedding_dim: int, hitory_cache_device=None, mv2share_memory=False, 
            set_gpu_cache=False, g: Optional[DGLGraph] = None, cache_size: Optional[int] = None,
            gpu_device = None
        ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
            
        self.gpu_cache: Optional[GPUCache] = None
        if set_gpu_cache:
            self.gpu_cache = GPUCache(embedding_dim, g, cache_size, gpu_device)

        pin_memory = hitory_cache_device is None or str(hitory_cache_device) == 'cpu'
        if mv2share_memory:
            self.emb = torch.empty(num_embeddings, embedding_dim, device=hitory_cache_device,
                               pin_memory=pin_memory).share_memory_()
        else:
            self.emb = torch.empty(num_embeddings, embedding_dim, device=hitory_cache_device,
                               pin_memory=pin_memory)

        self._device = torch.device('cpu')

        self._push_count = torch.zeros(1, dtype=torch.int32).share_memory_()

        self.reset_parameters()

    def add_metric(self, metric: Metric):
        self.metric = metric

    def reset_parameters(self):
        self.emb.fill_(0)

    def _apply(self, fn):
        # Set the `_device` of the module without transfering `self.emb`.
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def pull_without_gpu_cache(self, n_id: Optional[Tensor] = None) -> Tensor:
        out = self.emb
        if n_id is not None:
            # Move the n_id to device of embeddings for correctly run under pytorch DDP
            n_id = n_id.to(self.emb.device)
            # Under pytorch DDP, all input tensors (including n_id) is moved 
            # to the GPU of the current process by DDP, and this assertion will fail.
            assert n_id.device == self.emb.device
            self.metric.start("Pull_2.1: index_select")
            out = out.index_select(0, n_id)
            self.metric.stop("Pull_2.1: index_select")
        self.metric.start("Pull_2.2: transfer to GPU")
        ret = out.to(device=self._device)
        self.metric.stop("Pull_2.2: transfer to GPU")
        return ret

    @torch.no_grad()
    def pull_with_gpu_cache(self, n_id: Optional[Tensor] = None) -> Tensor:
        self.metric.start('Pull_1: pull_from_gpu')
        in_cache_mask, x_in_cache = self.gpu_cache.pull_by_n_ids(n_id)
        self.metric.stop('Pull_1: pull_from_gpu')
        self.metric.start('Pull_2: pull_from_cpu')
        x_not_in_cache = self.pull_without_gpu_cache(n_id[~in_cache_mask])
        self.metric.stop('Pull_2: pull_from_cpu')
        self.metric.set('pull_from_gpu_num_nodes', in_cache_mask.sum().item())
        self.metric.set('pull_from_cpu_num_nodes', (~in_cache_mask).sum().item())
        """
        Create the final x. x is a tensor with the same size as n_id, each element is the embedding of a node. 
        For positions in x such that the corresponding in_cache_mask is True, which means that node comes from the GPU cache, 
        put the corresponding element from x_in_cache. Otherwise, put the corresponding element from x_not_in_cache.
        """
        x = torch.empty((n_id.size(0), self.embedding_dim), device=x_in_cache.device)
        x[in_cache_mask] = x_in_cache
        x[~in_cache_mask] = x_not_in_cache
        return x


    @torch.no_grad()
    def pull(self, n_id: Optional[Tensor] = None) -> Tensor:
        if self.gpu_cache is None:
            return self.pull_without_gpu_cache(n_id)
        else:
            return self.pull_with_gpu_cache(n_id)


    @torch.no_grad()
    def push_in_chunks(self, x, n_ids: Optional[Tensor] = None,
             offset: Optional[Tensor] = None, count: Optional[Tensor] = None):

        if n_ids is None and x.size(0) != self.num_embeddings:
            raise ValueError

        elif n_ids is None and x.size(0) == self.num_embeddings:
            self.emb.copy_(x)

        elif offset is None or count is None:
            assert n_ids.device == self.emb.device
            self.emb[n_ids] = x.to(self.emb.device)

        else:  # Push in chunks:
            src_o = 0
            self.metric.start('Push_1: transfer to CPU')
            x = x.to(self.emb.device)
            self.metric.stop('Push_1: transfer to CPU')

            self.metric.start('Push_2: insert to history')
            for dst_o, c, in zip(offset.tolist(), count.tolist()):
                self.emb[dst_o:dst_o + c] = x[src_o:src_o + c]
                src_o += c
            self.metric.stop('Push_2: insert to history')

        self._push_count[0] = self._push_count[0].item() + 1


    def push_with_gpu_cache(self, x, n_ids: Optional[Tensor] = None,
             offset: Optional[Tensor] = None, count: Optional[Tensor] = None):
        if n_ids is None and x.size(0) != self.num_embeddings:
            raise ValueError

        elif n_ids is None and x.size(0) == self.num_embeddings:
            # Initiate CPU history cache
            self.emb.copy_(x)
            # Initiate GPU history cache
            n_ids = torch.arange(self.num_embeddings, device=self.emb.device)
            # The x of full graph is now set on CPU. We cannot move complete x to GPU to invoke gpu_cache.push_by_n_ids()
            self.gpu_cache.init_GPU_cache(x)
            

        elif offset is None or count is None:
            assert n_ids.device == self.emb.device
            self.emb[n_ids] = x.to(self.emb.device)

        else:  # Push node by node:
            # push to GPU first
            push_num_nodes = x.size(0)
            self.metric.set('push_total_nodes', push_num_nodes)
            self.metric.start('Push_1: push to GPU cache')
            n_ids, x = self.gpu_cache.push_by_n_ids(n_ids, x)
            push_2CPU_num_nodes = x.size(0)
            self.metric.set('push_2CPU_num_nodes', push_2CPU_num_nodes)
            self.metric.set('push_2GPU_num_nodes', push_num_nodes - push_2CPU_num_nodes)
            self.metric.stop('Push_1: push to GPU cache')
            src_o = 0
            self.metric.start('Push_2: transfer to CPU')
            x = x.to(self.emb.device)
            self.metric.stop('Push_2: transfer to CPU')

            self.metric.start('Push_3: insert to history')
            self.emb[n_ids] = x
            # for i, n_id in enumerate(n_ids.tolist()):
            #     self.emb[n_id] = x[i]
            self.metric.stop('Push_3: insert to history')

        self._push_count[0] = self._push_count[0].item() + 1
        
    @torch.no_grad()
    def push(self, x, n_ids: Optional[Tensor] = None,
             offset: Optional[Tensor] = None, count: Optional[Tensor] = None):
        if self.gpu_cache is not None:
            self.push_with_gpu_cache(x, n_ids, offset, count)
        else:
            self.push_in_chunks(x, n_ids, offset, count)
        

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_embeddings}, '
                f'{self.embedding_dim}, emb_device={self.emb.device}, '
                f'device={self._device})')
    
    def get_push_count(self):
        return self._push_count.item()
    
    def reset_push_count(self):
        self._push_count = torch.zeros(1, dtype=torch.int32).share_memory_()

class FMHistory(History):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, gamma: float = 0.0):
        super().__init__(num_embeddings, embedding_dim, device)
        self.gamma = gamma

    def FM_and_hist_update(self, x, hist_n_id):
        hist_outbatch_emb = self.pull(hist_n_id)
        h_bar_out_batch =  self.gamma * x + (1.0 - self.gamma) * hist_outbatch_emb
        self.push(h_bar_out_batch, n_ids=hist_n_id)