import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
import torch
import random
from torch_geometric.utils import degree

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def load_acm( ):
    # The order of node types: 0 p 1 a 2 s
    path = "./data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_p = sp.load_npz(path + "p_feat.npz")

    train_val_test_idx = np.load('./data/{}/train_val_test_idx.npz'.format('acm'))
    test_idx = train_val_test_idx['test_idx']

    gauss1 = np.random.normal(0, 1, (7167,64))
    feat_a = torch.from_numpy(gauss1).float()
    gauss2 = np.random.normal(0, 1, (7167,64))
    feat_a2 = torch.from_numpy(gauss2).float()

    gauss3 = np.random.normal(0, 1, (60,64))
    feat_s = torch.from_numpy(gauss3).float()
    gauss4 = np.random.normal(0, 1, (60,64))
    feat_s2 = torch.from_numpy(gauss4).float()


    labels = th.FloatTensor(label)
    labels = labels.argmax(axis=1)
    feat_p = th.FloatTensor(preprocess_features(feat_p))

    return  [feat_p,feat_a,feat_s],[feat_p,feat_a2,feat_s2],  labels , test_idx

def load_dblp( ):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "./data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")

    train_val_test_idx = np.load('./data/{}/train_val_test_idx.npz'.format('dblp'))
    test_idx = train_val_test_idx['test_idx']


    labels = th.FloatTensor(label)
    labels = labels.argmax(axis=1)

    feat_p = th.FloatTensor(preprocess_features(feat_p))


    gauss01 = np.random.normal(0, 1, (4057,64))
    feat_a = torch.from_numpy(gauss01).float()
    gauss02 = np.random.normal(0, 1, (4057,64))
    feat_a2 = torch.from_numpy(gauss02).float()

    gauss1 = np.random.normal(0, 1, (20,64))
    feat_c = torch.from_numpy(gauss1).float()
    gauss2 = np.random.normal(0, 1, (7723,64))
    feat_t = torch.from_numpy(gauss2).float()

    gauss12 = np.random.normal(0, 1, (20,64))
    feat_c2 = torch.from_numpy(gauss12).float()
    gauss22 = np.random.normal(0, 1, (7723,64))
    feat_t2 = torch.from_numpy(gauss22).float()

    return [feat_a, feat_p, feat_c, feat_t], [feat_a2, feat_p, feat_c2, feat_t2], labels, test_idx

def load_yelp():
    dataset = 'yelp'
    labels = np.load('./data/{}/labels.npy'.format(dataset))
    train_val_test_idx = np.load('./data/{}/train_val_test_idx.npz'.format(dataset))
    test_idx = train_val_test_idx['test_idx']


    gauss1 = np.random.normal(0, 1, (1286,64))
    feat_1 = torch.from_numpy(gauss1).float()
    gauss2 = np.random.normal(0, 1, (4,64))
    feat_2 = torch.from_numpy(gauss2).float()
    gauss3 = np.random.normal(0, 1, (9,64))
    feat_3 = torch.from_numpy(gauss3).float()

    gauss12 = np.random.normal(0, 1, (1286,64))
    feat_12 = torch.from_numpy(gauss12).float()
    gauss22 = np.random.normal(0, 1, (4,64))
    feat_22 = torch.from_numpy(gauss22).float()
    gauss32 = np.random.normal(0, 1, (9,64))
    feat_32 = torch.from_numpy(gauss32).float()

    feats = []
    feats.append(th.FloatTensor(preprocess_features(sp.load_npz('./data/yelp/features_0.npz'))))

    feats.append(feat_1)
    feats.append(feat_2)
    feats.append(feat_3)

    feats2 = []
    feats2.append(th.FloatTensor(preprocess_features(sp.load_npz('./data/yelp/features_0.npz'))))
    feats2.append(feat_12)
    feats2.append(feat_22)
    feats2.append(feat_32)

    return feats,feats2, labels, test_idx


def load_imdb():
    dataset = 'imdb'

    labels = np.load('./data/{}/labels.npy'.format(dataset))
    train_val_test_idx = np.load('./data/{}/train_val_test_idx.npz'.format(dataset))
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    gauss1 = np.random.normal(0, 1, (2081,64))
    feat_1 = torch.from_numpy(gauss1).float()
    gauss2 = np.random.normal(0, 1, (5257,64))
    feat_2 = torch.from_numpy(gauss2).float()

    gauss12 = np.random.normal(0, 1, (2081,64))
    feat_12 = torch.from_numpy(gauss12).float()
    gauss22 = np.random.normal(0, 1, (5257,64))
    feat_22 = torch.from_numpy(gauss22).float()


    feats = []
    feats.append(th.FloatTensor(preprocess_features(sp.load_npz('./data/imdb/features_0.npz'))))
    feats.append(feat_1)
    feats.append(feat_2)


    feats2 = []
    feats2.append(th.FloatTensor(preprocess_features(sp.load_npz('./data/imdb/features_0.npz'))))
    feats2.append(feat_12)
    feats2.append(feat_22)

    return feats,feats2, labels, test_idx

def load_data(dataset):
    if dataset == "acm":
        data = load_acm()
    elif dataset == "dblp":
        data = load_dblp()
    elif dataset == "imdb":
        data = load_imdb()
    elif dataset == "yelp":
        data = load_yelp()
    return data