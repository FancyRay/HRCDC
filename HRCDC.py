import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import consistency_loss

class MLPlayers(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, use_batchnorm=False, use_dropout=False):
        super(MLPlayers, self).__init__()
        self.layers = nn.ModuleList()

        def add_layer(in_dim, out_dim, batch_norm=False, dropout=False):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.Tanh())
            if dropout:
                self.layers.append(nn.Dropout(0.2))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(out_dim))

        if hidden_dim == 0:
            add_layer(input_dim, output_dim, use_batchnorm, use_dropout)
        else:
            add_layer(input_dim, hidden_dim, use_batchnorm, use_dropout)
            add_layer(hidden_dim, output_dim, use_batchnorm, use_dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                # self.norms.append(nn.LayerNorm(out_dim))
                self.norms.append(nn.BatchNorm1d(out_dim, affine=False))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = edges.data['id'][0]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key = torch.bmm(edges.src['k'].transpose(1, 0), relation_att).transpose(1, 0)
        att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val = torch.bmm(edges.src['v'].transpose(1, 0), relation_msg).transpose(1, 0)
        return {'a': att, 'v': val}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, inp_key, out_key):
        node_dict, edge_dict = G.node_dict, G.edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]]
            q_linear = self.q_linears[node_dict[dsttype]]

            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)

            G.apply_edges(func=self.edge_attention, etype=etype)
        G.multi_update_all({etype: (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer='mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1 - alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.norms[n_id](self.drop(self.norms[n_id](trans_out)))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)

class HRCDC(nn.Module):
    def __init__(self, G, n_inps, n_hid, n_layers, n_heads, mask_key, mlp_hidden,use_norm,use_drop,deco_drop):
        super(HRCDC, self).__init__()
        self.gcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.n_inps = n_inps
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.mask_key = mask_key
        self.adapt_ws = nn.ModuleList()
        self.crosspros = nn.ModuleList()
        for t in range(len(G.node_dict)):
            self.adapt_ws.append(nn.Linear(n_inps[t], n_hid))
            self.norms.append(nn.BatchNorm1d(n_hid, affine=False))
            self.crosspros.append(MLPlayers(n_hid, n_hid,0,use_norm,False))

        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, len(G.node_dict), len(G.edge_dict), n_heads, use_norm=use_norm))

        self.feat_map_emb = MLPlayers(n_inps[int(mask_key)],n_hid,mlp_hidden, use_norm , use_drop)
        self.deco_layers = MLPlayers(n_hid,n_inps[int(mask_key)],mlp_hidden, use_norm , deco_drop)
        self.withinpro = MLPlayers(n_hid, n_hid,0,use_norm,False)

        self.reconloss = nn.MSELoss(reduction='sum')



    def forward(self, G, mask_key, target_key, features_list, features_list2,istest, epoch, maskrate, alpha):
        maskrate = self.get_mask_rate(input_mask_rate=maskrate, epoch=epoch, istest=istest)
        feat_maskp, feat_maskp2, masknodes, masknodes2 = self.mask_attr(features_list[int(mask_key)], maskrate)

        feat_mask1 = []
        feat_mask2 = []
        if int(mask_key)==0:
            feat_mask1.append(feat_maskp)
            feat_mask2.append(feat_maskp2)
            for i in range(len(features_list)-1):
                feat_mask1.append(features_list[i+1])
                feat_mask2.append(features_list2[i+1])

        if int(mask_key)==1:
            feat_mask1.append(features_list[0])
            feat_mask2.append(features_list[0])
            feat_mask1.append(feat_maskp)
            feat_mask2.append(feat_maskp2)
            for i in range(len(features_list)-2):
                feat_mask1.append(features_list[i+2])
                feat_mask2.append(features_list2[i+2])

        for ntype in G.ntypes:
            G.nodes[ntype].data['inp1'] = feat_mask1[int(ntype)]
            G.nodes[ntype].data['inp2'] = feat_mask2[int(ntype)]
        for ntype in G.ntypes:
            n_id = G.node_dict[ntype]
            G.nodes[ntype].data['h1'] = self.norms[n_id](torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['inp1'])) )
            G.nodes[ntype].data['h2'] = self.norms[n_id](torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['inp2'])) )
        for i in range(self.n_layers):
            self.gcs[i](G, 'h1', 'h1')
            self.gcs[i](G, 'h2', 'h2')

        emb_mask1 = G.nodes[mask_key].data['h1']
        emb_mask2 = G.nodes[mask_key].data['h2']

        emb_semantic = self.feat_map_emb(features_list[int(mask_key)])
        emb_within1 = self.withinpro(emb_mask1)
        emb_within2 = self.withinpro(emb_mask2)
        emb_semantic_w = whitening(emb_semantic, eps=0.01)
        emb_within1_w = whitening(emb_within1, eps=0.01)
        emb_within2_w = whitening(emb_within2, eps=0.01)

        losswithin = consistency_loss(emb_semantic_w, emb_within1_w,alpha=alpha) +  consistency_loss(emb_semantic_w, emb_within2_w,alpha=alpha)

        emb_crosspro1_w = []
        emb_crosspro2_w = []

        for ntype in G.ntypes:
            n_id = G.node_dict[ntype]
            emb_crosspro1_w.append( whitening( self.crosspros[n_id](G.nodes[ntype].data['h1'])))
            emb_crosspro2_w.append( whitening( self.crosspros[n_id](G.nodes[ntype].data['h2'])))

        losscross = 0
        for ntype in G.ntypes:
            losscross = losscross + consistency_loss(emb_crosspro1_w[int(ntype)],emb_crosspro2_w[int(ntype)],alpha=alpha)

        feat_recon1 = self.deco_layers(emb_mask1)
        feat_recon2 = self.deco_layers(emb_mask2)

        loss_recon = self.reconloss(feat_recon1, features_list[int(mask_key)]) + self.reconloss(feat_recon2, features_list[int(mask_key)])

        embfuse = torch.cat([G.nodes[target_key].data['h1'], G.nodes[target_key].data['h2']],dim=-1)

        return embfuse, losswithin, losscross, loss_recon

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None, istest=False):
        if istest==1:
            return 0.0
        else:
            try:
                return float(input_mask_rate)
            except ValueError:
                if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                    mask_rate = [float(i) for i in input_mask_rate.split('~')]
                    assert len(mask_rate) == 2
                    if get_min:
                        return mask_rate[0]
                    else:
                        return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
                elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                    mask_rate = [float(i) for i in input_mask_rate.split(',')]
                    assert len(mask_rate) == 3
                    start = mask_rate[0]
                    step = mask_rate[1]
                    end = mask_rate[2]
                    if get_min:
                        return min(start, end)
                    else:
                        cur_mask_rate = start + epoch * step
                        if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                            return end
                        return cur_mask_rate
                else:
                    raise NotImplementedError

    def mask_attr(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        num_keep_nodes = num_nodes-num_mask_nodes
        mask_idx = perm[: num_mask_nodes].long()
        keep_idx = perm[num_mask_nodes:].long()
        mask_idx2 = perm[num_keep_nodes:].long()
        keep_idx2 = perm[:num_keep_nodes].long()

        out_x_mask = x.clone()
        out_x_mask[mask_idx] = 0.0
        out_x_mask2 = x.clone()
        out_x_mask2[mask_idx2] = 0.0

        return out_x_mask, out_x_mask2, (mask_idx, keep_idx), (mask_idx2, keep_idx2)

def whitening(x, eps=0.01):
    x = F.normalize(x)
    x = x-x.mean(dim=0)
    batch_size, feature_dim = x.size()
    f_cov = torch.mm(x.transpose(0, 1), x) / (batch_size-1) # d * d
    eye = torch.eye(feature_dim).float().to(f_cov.device)
    f_cov_shrink = (1 - eps) * f_cov + eps * eye
    inv_sqrt = torch.triangular_solve(eye, torch.linalg.cholesky(f_cov_shrink.float()), upper=False)[0]
    inv_sqrt = inv_sqrt.contiguous().view(feature_dim, feature_dim).detach()
    return torch.mm(inv_sqrt, x.transpose(0, 1)).transpose(0, 1)    # N * d
