from typing import Optional, Callable, Dict

import warnings
import tqdm

import torch
from torch import Tensor

from torch_autoscale import History, AsyncIOPool, FMHistory
from torch.utils.data import DataLoader as TorchDataLoader

import dgl
from dgl.heterograph import DGLBlock


class ScalableGNN(torch.nn.Module):
    r"""An abstract class for implementing scalable GNNs via historical
    embeddings.
    This class will take care of initializing :obj:`num_layers - 1` historical
    embeddings, and provides a convenient interface to push recent node
    embeddings to the history, and to pull previous embeddings from the
    history.
    In case historical embeddings are stored on the CPU, they will reside
    inside pinned memory, which allows for asynchronous memory transfers of
    historical embeddings.
    For this, this class maintains a :class:`AsyncIOPool` object that
    implements the underlying mechanisms of asynchronous memory transfers as
    described in our paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        hidden_channels (int): The number of hidden channels of the model.
            As a current restriction, all intermediate node embeddings need to
            utilize the same number of features.
        num_layers (int): The number of layers of the model.
        pool_size (int, optional): The number of pinned CPU buffers for pulling
            histories and transfering them to GPU.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
        buffer_size (int, optional): The size of pinned CPU buffers, i.e. the
            maximum number of out-of-mini-batch nodes pulled at once.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
    """
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers - 1 if pool_size is None else pool_size
        self.buffer_size = buffer_size

        self.histories = torch.nn.ModuleList([
            History(num_nodes, hidden_channels, device)
            for _ in range(num_layers - 1)
        ])

        self.pool: Optional[AsyncIOPool] = None
        self._async = False
        self.__out: Optional[Tensor] = None

    @property
    def emb_device(self):
        return self.histories[0].emb.device

    @property
    def device(self):
        return self.histories[0]._device

    # 在调用Module.to()函数的时候，会调用Module._apply()函数，此处借用这个机会来定义是否使用pool做异步传输
    def _apply(self, fn: Callable) -> None:
        super()._apply(fn)
        # We only initialize the AsyncIOPool in case histories are on CPU:
        if (str(self.emb_device) == 'cpu' and str(self.device)[:4] == 'cuda'
                and self.pool_size is not None
                and self.buffer_size is not None):
            self.pool = AsyncIOPool(self.pool_size, self.buffer_size,
                                    self.histories[0].embedding_dim)
            self.pool.to(self.device)
        return self

    def reset_parameters(self):
        for history in self.histories:
            history.reset_parameters()

    def __call__(self, **kwargs):
        raise NotImplementedError

    def push_and_pull(self, history: History, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None) -> Tensor:
        r"""Pushes and pulls information from :obj:`x` to :obj:`history` and
        vice versa."""

        if n_id is None and x.size(0) != self.num_nodes:
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            history.push(x)
            return x

        assert n_id is not None

        if batch_size is None:
            history.push(x, n_id)
            return x

        if not self._async:
            history.push(x[:batch_size], n_id[:batch_size], offset, count)
            h = history.pull(n_id[batch_size:])
            return torch.cat([x[:batch_size], h], dim=0)

        else:
            out = self.pool.synchronize_pull()[:n_id.numel() - batch_size]
            self.pool.async_push(x[:batch_size], offset, count, history.emb)
            out = torch.cat([x[:batch_size], out], dim=0)
            self.pool.free_pull()
            return out
        
    def history_pull(self, history: History, n_id: Tensor) -> Tensor:
        """ 
        Pulls information from history
        Note that hitory is of one layer
        """

        return history.pull(n_id)

    @property
    def _out(self):
        if self.__out is None:
            self.__out = torch.empty(self.num_nodes, self.out_channels,
                                     pin_memory=True)
        return self.__out


    @torch.no_grad()
    def layerwise_inference(self, g: dgl.DGLGraph, feat: Tensor, node_batch_size: int, 
                            out_channels: int, device, history_refresh: bool = False):
        """
        Perform the full-graph inference layer by layer.
        
        Arg: 
            node_batch_size: The number of nodes in a batch, different from the number of partitions 
                used in training.
        """
        nodes = torch.arange(g.number_of_nodes())
        ys = []
        for l in range(self.num_layers):
            y = torch.zeros(
                g.number_of_nodes(),
                self.hidden_channels if l != self.num_layers - 1 else out_channels,
            )

            for start in range(0, len(nodes), node_batch_size):
                end = start + node_batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(
                    dgl.in_subgraph(g, batch_nodes), batch_nodes
                )
                block = block.int().to(device)
                induced_nodes = block.srcdata[dgl.NID]
                feat = feat.to(device)

                h = feat[induced_nodes]
                h = self.forward_layer(l, block, h)

                y[start:end] = h.cpu()

            ys.append(y)
            feat = y
            # We do not push the final output into history
            if history_refresh and l < self.num_layers - 1:
                self.histories[l].push(feat)
        return y, ys

    @torch.no_grad()
    def forward_layer(self, layer_number: int, block: DGLBlock, h: Tensor):
        raise NotImplementedError
    

class GASGNN(ScalableGNN):
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int,
                pool_size: Optional[int] = None,
                buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                        buffer_size, device)
    def __call__(
        self,
        block_2IB: DGLBlock,
        x,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        # loader: EvalSubgraphLoader = None,
        **kwargs,
    ) -> Tensor:
        r"""Enhances the call of forward propagation by immediately start
        pulling historical embeddings for all layers asynchronously.
        After forward propogation is completed, the push of node embeddings to
        the histories will be synchronized.

        For example, given a mini-batch with node indices
        :obj:`n_id = [0, 1, 5, 6, 7, 3, 4]`, where the first 5 nodes
        represent the mini-batched nodes, and nodes :obj:`3` and :obj:`4`
        denote out-of-mini-batched nodes (i.e. the 1-hop neighbors of the
        mini-batch that are not included in the current mini-batch), then
        other input arguments should be given as:

        .. code-block:: python

            batch_size = 5
            offset = [0, 5]
            count = [2, 3]

        Args:
            x (Tensor, optional): Node feature matrix. (default: :obj:`None`)
            adj_t (SparseTensor, optional) The sparse adjacency matrix.
                (default: :obj:`None`)
            batch_size (int, optional): The in-mini-batch size of nodes.
                (default: :obj:`None`)
            n_id (Tensor, optional): The global indices of mini-batched and
                out-of-mini-batched nodes. (default: :obj:`None`)
            offset (Tensor, optional): The offset of mini-batched nodes inside
                a utilize a contiguous memory layout. (default: :obj:`None`)
            count (Tensor, optional): The number of mini-batched nodes inside a
                contiguous memory layout. (default: :obj:`None`)
            loader (EvalSubgraphLoader, optional): A subgraph loader used for
                evaluating the given GNN in a layer-wise fashsion.
        """

        # if loader is not None:
        #     return self.mini_inference(loader)

        # We only perform asynchronous history transfer in case the following
        # conditions are met:
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)

        if (batch_size is not None and not self._async
                and str(self.emb_device) == 'cpu'
                and str(self.device)[:4] == 'cuda'):
            warnings.warn('Asynchronous I/O disabled, although history and '
                          'model sit on different devices.')

        if self._async:
            for hist in self.histories:
                self.pool.async_pull(hist.emb, None, None, n_id[batch_size:])

        out = self.forward(block_2IB, x, batch_size, n_id, offset, count, **kwargs)

        if self._async:
            for hist in self.histories:
                self.pool.synchronize_push()

        self._async = False

        return out
            
class FMGNN(ScalableGNN):
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int, pool_size: Optional[int] = None, buffer_size: Optional[int] = None, device=None, gamma: float = 0.0):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size, buffer_size, device)
        self.histories = torch.nn.ModuleList([
            FMHistory(num_nodes, hidden_channels, device, gamma)
            for _ in range(num_layers - 1)
        ])

    def __call__(
        self,
        block_2IB: DGLBlock,
        x,
        block_2OB: Optional[DGLBlock] = None,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        # loader: EvalSubgraphLoader = None,
        **kwargs,
    ) -> Tensor:
        r"""Enhances the call of forward propagation by immediately start
        pulling historical embeddings for all layers asynchronously.
        After forward propogation is completed, the push of node embeddings to
        the histories will be synchronized.

        For example, given a mini-batch with node indices
        :obj:`n_id = [0, 1, 5, 6, 7, 3, 4]`, where the first 5 nodes
        represent the mini-batched nodes, and nodes :obj:`3` and :obj:`4`
        denote out-of-mini-batched nodes (i.e. the 1-hop neighbors of the
        mini-batch that are not included in the current mini-batch), then
        other input arguments should be given as:

        .. code-block:: python

            batch_size = 5
            offset = [0, 5]
            count = [2, 3]

        Args:
            x (Tensor, optional): Node feature matrix. (default: :obj:`None`)
            adj_t (SparseTensor, optional) The sparse adjacency matrix.
                (default: :obj:`None`)
            batch_size (int, optional): The in-mini-batch size of nodes.
                (default: :obj:`None`)
            n_id (Tensor, optional): The global indices of mini-batched and
                out-of-mini-batched nodes. (default: :obj:`None`)
            offset (Tensor, optional): The offset of mini-batched nodes inside
                a utilize a contiguous memory layout. (default: :obj:`None`)
            count (Tensor, optional): The number of mini-batched nodes inside a
                contiguous memory layout. (default: :obj:`None`)
            loader (EvalSubgraphLoader, optional): A subgraph loader used for
                evaluating the given GNN in a layer-wise fashsion.
        """

        # if loader is not None:
        #     return self.mini_inference(loader)

        # We only perform asynchronous history transfer in case the following
        # conditions are met:
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)

        if (batch_size is not None and not self._async
                and str(self.emb_device) == 'cpu'
                and str(self.device)[:4] == 'cuda'):
            warnings.warn('Asynchronous I/O disabled, although history and '
                          'model sit on different devices.')

        if self._async:
            for hist in self.histories:
                self.pool.async_pull(hist.emb, None, None, n_id[batch_size:])

        out = self.forward(block_2IB, x, block_2OB, batch_size, n_id, offset, count, **kwargs)

        if self._async:
            for hist in self.histories:
                self.pool.synchronize_push()

        self._async = False

        return out