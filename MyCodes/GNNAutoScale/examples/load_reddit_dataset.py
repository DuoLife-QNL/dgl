import json
import numpy as np
import scipy
import torch as th
import dgl

def load_reddit():
    prefix = '/home/lihz/Codes/dgl/MyCodes/dataset/Reddit2/raw/'

    with open(prefix + 'role.json') as f:
        role = json.load(f)

    adj_full = scipy.sparse.load_npz(prefix + 'adj_full.npz')
    feats = np.load(prefix + 'feats.npy')
    n_node = feats.shape[0]

    ys = [-1] * n_node
    with open(prefix + 'class_map.json') as f:
        class_map = json.load(f)
        for key, item in class_map.items():
            ys[int(key)] = item
    label = th.tensor(ys)

    g = dgl.from_scipy(adj_full)
    node_data = g.ndata

    # label = list(class_map.values())
    # node_data['label'] = th.tensor(label)
    node_data['label'] = label
    n_classes = label.max().item() + 1

    node_data['train_mask'] = th.zeros(n_node, dtype=th.bool)
    node_data['val_mask'] = th.zeros(n_node, dtype=th.bool)
    node_data['test_mask'] = th.zeros(n_node, dtype=th.bool)
    node_data['train_mask'][role['tr']] = True
    node_data['val_mask'][role['va']] = True
    node_data['test_mask'][role['te']] = True

    assert th.all(th.logical_not(th.logical_and(node_data['train_mask'], node_data['val_mask'])))
    assert th.all(th.logical_not(th.logical_and(node_data['train_mask'], node_data['test_mask'])))
    assert th.all(th.logical_not(th.logical_and(node_data['val_mask'], node_data['test_mask'])))
    assert th.all(
        th.logical_or(th.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

    feats = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    node_data['feat'] = th.tensor(feats, dtype=th.float)

    return g, n_classes