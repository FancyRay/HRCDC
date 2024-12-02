import dgl
from HRCDC import *
import torch
from data_loader import data_loader
from utils.data import  load_data
import argparse
import time
from utils.tools import evaluate_results_nc
from ruamel.yaml import YAML
import random
import os
import numpy as np

ap = argparse.ArgumentParser(description='HRCDC')
ap.add_argument('--dataset', type=str, default='acm')
ap.add_argument('--weight_decay', type=float, default=1e-4)
ap.add_argument('--device', type=int, default=7)
ap.add_argument('--hidden_dim', type=int, default=128, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num_heads', type=int, default=4, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--lr', type=float, default=5e-3)
ap.add_argument('--use_norm', type=bool, default=True)
ap.add_argument('--use_drop', type=bool, default=True)
ap.add_argument('--deco_drop', type=bool, default=False)
ap.add_argument('--maskrate', type=str, default='0.1,0.0005,0.5')
ap.add_argument('--recon_loss_lam', type=float, default=0.0005)
ap.add_argument('--cross_loss_lam', type=float, default=0.0005)
ap.add_argument('--within_loss_lam', type=float, default=0.0005)
ap.add_argument('--alpha', type=int, default=4)
ap.add_argument('--mask_key', type=str, default='0')
ap.add_argument('--target_key', type=str, default='0')
ap.add_argument('--mlphidden', type=int, default='128')


yaml_path = 'args.yaml'
with open(yaml_path) as args_file:
    args = ap.parse_args()
    args_key = args.dataset
    try:
        ap.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
    except KeyError:
        raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

args = ap.parse_args()

seed = 1029
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

maskrate = args.maskrate
recon_loss_lam = args.recon_loss_lam
cross_loss_lam = args.cross_loss_lam
within_loss_lam = args.within_loss_lam
alpha = args.alpha
mask_key = args.mask_key
target_key = args.target_key
dataset = args.dataset
checkpointfile = f"checkpoint/checkpoint_{dataset}.pt"
print(args)
device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

dl = data_loader(args.dataset)
edge_dict = {}

for i, meta_path in dl.links['meta'].items():
    edge_dict[(str(meta_path[0]), str(meta_path[0]) + '_' + str(meta_path[1]), str(meta_path[1]))] = (
    torch.tensor(dl.links['data'][i].tocoo().row - dl.nodes['shift'][meta_path[0]]),
    torch.tensor(dl.links['data'][i].tocoo().col - dl.nodes['shift'][meta_path[1]]))

node_count = {}
for i, count in dl.nodes['count'].items():
    node_count[str(i)] = count


G = dgl.heterograph(edge_dict, num_nodes_dict=node_count, device=device)
"""
for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = dataset.nodes['attr'][ntype]
    # print(G.nodes['attr'][ntype].shape)
"""

G.node_dict = {}
G.edge_dict = {}
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype]

features_list,features_list2,labels, test = load_data(args.dataset)

features_list = [features.to(device) for features in features_list]
features_list2 = [features.to(device) for features in features_list2]
test = torch.LongTensor(test).to(device)

in_dims = [features.shape[1] for features in features_list]

labels = torch.LongTensor(labels).to(device)

model = HRCDC(G, n_inps=in_dims, n_hid=args.hidden_dim,  n_layers=args.num_layers,
            n_heads=args.num_heads, use_norm=args.use_norm,use_drop=args.use_drop,deco_drop=args.deco_drop,mask_key=args.mask_key, mlp_hidden=args.mlphidden).to(device)

optimizer = torch.optim.AdamW(model.parameters(),
                              weight_decay=args.weight_decay,lr=args.lr)

train_step = 0

for epoch in range(args.epoch):
    model.train()
    emb_mask, loss_within, loss_cross,loss_recon = model(G, mask_key,target_key,features_list,features_list2,0,epoch,maskrate,alpha)

    train_loss = loss_within*within_loss_lam + loss_cross*cross_loss_lam +  loss_recon*recon_loss_lam
    optimizer.zero_grad()
    train_loss.backward()

    optimizer.step()
    train_step += 1

    print('Epoch {:05d} | loss_align1 {:.4f} | loss_align2 {:.4f}  | loss_recon {:.4f} '.format(
        epoch, loss_within ,loss_cross,loss_recon))
    print('Epoch {:05d} | train_loss {:.4f} | '.format(epoch, train_loss))

model.load_state_dict(torch.load(checkpointfile,map_location='cuda:0'))
model.eval()
test_logits = []
with torch.no_grad():
    emb , _ ,_,_= model(G,mask_key,target_key,features_list,features_list2,1,epoch,maskrate,alpha)
    svm_macro, svm_micro, nmi, ari = evaluate_results_nc(emb.cpu().numpy(),
                                                         labels.cpu().numpy(),
                                                         num_classes=labels.max().item() + 1)




