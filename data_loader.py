import os
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict


# from sklearn.metrics import f1_score

class data_loader_acm:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i

    def list_to_sp_mat(self, li,shape1,shape2):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(shape1,shape2)).tocsr()

    def list_to_sp_mat00(self, li,shape1,shape2,mod):
        data = [x[2] for x in li]
        if (mod == 1):
            i = [x[0] for x in li]
            j = [x[1]-4019 for x in li]
        if (mod == 2):
            i = [x[0]-4019 for x in li]
            j = [x[1] for x in li]
        if (mod == 3):
            i = [x[0] for x in li]
            j = [x[1]-11186 for x in li]
        if (mod == 4):
            i = [x[0]-11186 for x in li]
            j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(shape1,shape2)).tocsr()

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list),'edge': defaultdict(list)}
        # links['meta'][0] = (0,0)
        # links['meta'][1] = (0,0)
        links['meta'][0] = (0,1)
        links['meta'][1] = (1,0)
        links['meta'][2] = (0,2)
        links['meta'][3] = (2,0)
        with open(os.path.join(self.path, 'pa.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id = int(th[0]), int(th[1])
                t_id = t_id + 4019
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 1):
                    links['data'][0].append((h_id, t_id, 1))
                    links['data'][1].append((t_id, h_id, 1))
                    links['count'][0] += 1
                    links['count'][1] += 1
                    links['total'] += 2
                    links['edge']['pa'].append((h_id, t_id, 1))
                    links['edge']['ap'].append((t_id, h_id, 1))
                if (t_type == 2):
                    links['data'][2].append((h_id, t_id, 1))
                    links['data'][3].append((t_id, h_id, 1))
                    links['count'][2] += 1
                    links['count'][3] += 1
                    links['total'] += 2
                    links['edge']['ps'].append((h_id, t_id, 1))
                    links['edge']['sp'].append((t_id, h_id, 1))
        ### to create apa


        with open(os.path.join(self.path, 'ps.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id = int(th[0]), int(th[1])
                t_id = t_id + 11186
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type==0):
                    links['data'][0].append((h_id, t_id, 1))
                    links['data'][1].append((t_id, h_id, 1))
                    links['count'][0] += 1
                    links['count'][1] += 1
                    links['total'] += 2
                if (t_type == 1):
                    links['data'][0].append((h_id, t_id, 1))
                    links['data'][1].append((t_id, h_id, 1))
                    links['count'][0] += 1
                    links['count'][1] += 1
                    links['total'] += 2
                    links['edge']['pa'].append((h_id, t_id, 1))
                    links['edge']['ap'].append((t_id, h_id, 1))
                if (t_type == 2):
                    links['data'][2].append((h_id, t_id, 1))
                    links['data'][3].append((t_id, h_id, 1))
                    links['count'][2] += 1
                    links['count'][3] += 1
                    links['total'] += 2
                    links['edge']['ps'].append((h_id, t_id, 1))
                    links['edge']['sp'].append((t_id, h_id, 1))

        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id],self.nodes['total'],self.nodes['total'])
        links['data'] = new_data

        links['edge']['pa'] = self.list_to_sp_mat00(links['edge']['pa'],4019,7167,1)
        links['edge']['ap'] = self.list_to_sp_mat00(links['edge']['ap'],7167,4019,2)
        links['edge']['ps'] = self.list_to_sp_mat00(links['edge']['ps'],4019,60,3)
        links['edge']['sp'] = self.list_to_sp_mat00(links['edge']['sp'],60,4019,4)
        return links


    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        nodes['total'] = 11246

        nodes['count'][0] = 4019
        nodes['count'][1] = 7167
        nodes['count'][2] = 60

        nodes['shift'][0] = 0
        nodes['shift'][1] = 4019
        nodes['shift'][2] = 11186

        nodes['attr'] = None
        return nodes


class data_loader_dblp:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()


    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i


    def list_to_sp_mat(self, li,shape1,shape2):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(shape1,shape2)).tocsr()

    def list_to_sp_mat00(self, li,shape1,shape2,mod):
        data = [x[2] for x in li]
        if (mod == 1):
            i = [x[0] for x in li]
            j = [x[1]-4019 for x in li]
        if (mod == 2):
            i = [x[0]-4019 for x in li]
            j = [x[1] for x in li]
        if (mod == 3):
            i = [x[0] for x in li]
            j = [x[1]-11186 for x in li]
        if (mod == 4):
            i = [x[0]-11186 for x in li]
            j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(shape1,shape2)).tocsr()

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list),'edge': defaultdict(list),'nei_t':{ }}
        # links['meta'][0] = (0,0)
        # links['meta'][1] = (0,0)
        links['meta'][0] = (1,0)
        links['meta'][1] = (0,1)
        links['meta'][2] = (1,2)
        links['meta'][3] = (2,1)
        links['meta'][4] = (1,3)
        links['meta'][5] = (3,1)
        # note: tail type can be 0 2 3 head type is only 1
        with open(os.path.join(self.path, 'pa.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id = int(th[0]), int(th[1])
                h_id = h_id + 4057      ### all P nodes are heads
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 0):
                    links['data'][0].append((h_id, t_id, 1))
                    links['data'][1].append((t_id, h_id, 1))
                    links['count'][0] += 1
                    links['count'][1] += 1
                    links['total'] += 2

        with open(os.path.join(self.path, 'pc.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id = int(th[0]), int(th[1])
                h_id = h_id + 4057
                t_id = t_id + 18385
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 2):
                    links['data'][2].append((h_id, t_id, 1))
                    links['data'][3].append((t_id, h_id, 1))
                    links['count'][2] += 1
                    links['count'][3] += 1
                    links['total'] += 2
                    # links['edge']['pc'].append((h_id, t_id, 1))
                    # links['edge']['cp'].append((t_id, h_id, 1))
        links['nei_t']={}
        for i in range(14328):
            links['nei_t'][i] = []
        with open(os.path.join(self.path, 'pt.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id = int(th[0]), int(th[1])
                links['nei_t'][h_id].append(t_id)
                h_id = h_id + 4057
                t_id = t_id + 18405
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 3):
                    links['data'][4].append((h_id, t_id, 1))
                    links['data'][5].append((t_id, h_id, 1))
                    links['count'][4] += 1
                    links['count'][5] += 1
                    links['total'] += 2
                    # links['edge']['pt'].append((h_id, t_id, 1))
                    # links['edge']['tp'].append((t_id, h_id, 1))

        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id],self.nodes['total'],self.nodes['total'])
        links['data'] = new_data

        return links


    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        nodes['total'] = 26128

        nodes['count'][0] = 4057
        nodes['count'][1] = 14328
        nodes['count'][2] = 20
        nodes['count'][3] = 7723

        nodes['shift'][0] = 0
        nodes['shift'][1] = 4057
        nodes['shift'][2] = 18385
        nodes['shift'][3] = 18405

        nodes['attr'] = None
        return nodes


class data_loader_imdb:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li, shape1, shape2):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(shape1, shape2)).tocsr()

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list), 'edge': defaultdict(list),
                 'nei_t': {}}
        # links['meta'][0] = (0,0)
        # links['meta'][1] = (0,0)
        links['meta'][0] = (0, 1)
        links['meta'][1] = (1, 0)
        links['meta'][2] = (0, 2)
        links['meta'][3] = (2, 0)
        # note: tail type can be 0 2 3 head type is only 1
        with open(os.path.join(self.path, 'IMDBnei_d.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split(' ')
                h_id, t_id = int(th[0]), int(th[1])
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 1):
                    links['data'][0].append((h_id, t_id, 1))
                    links['data'][1].append((t_id, h_id, 1))
                    links['count'][0] += 1
                    links['count'][1] += 1
                    links['total'] += 2

        with open(os.path.join(self.path, 'IMDBnei_a.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split(' ')
                h_id, t_id = int(th[0]), int(th[1])
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 2):
                    links['data'][2].append((h_id, t_id, 1))
                    links['data'][3].append((t_id, h_id, 1))
                    links['count'][2] += 1
                    links['count'][3] += 1
                    links['total'] += 2
                    # links['edge']['pc'].append((h_id, t_id, 1))
                    # links['edge']['cp'].append((t_id, h_id, 1))

        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id], self.nodes['total'], self.nodes['total'])
        links['data'] = new_data

        return links

    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        nodes['total'] = 11616

        nodes['count'][0] = 4278
        nodes['count'][1] = 2081
        nodes['count'][2] = 5257

        nodes['shift'][0] = 0
        nodes['shift'][1] = 4278
        nodes['shift'][2] = 6359

        nodes['attr'] = None
        return nodes


class data_loader_yelp:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li, shape1, shape2):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(shape1, shape2)).tocsr()

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list), 'edge': defaultdict(list),
                 'nei_t': {}}
        # links['meta'][0] = (0,0)
        # links['meta'][1] = (0,0)
        links['meta'][0] = (0, 1)
        links['meta'][1] = (1, 0)
        links['meta'][2] = (0, 2)
        links['meta'][3] = (2, 0)
        links['meta'][4] = (0, 3)
        links['meta'][5] = (3, 0)
        # note: tail type can be 0 2 3 head type is only 1
        with open(os.path.join(self.path, 'nei_u.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split(' ')
                h_id, t_id = int(th[0]), int(th[1])
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 1):
                    links['data'][0].append((h_id, t_id, 1))
                    links['data'][1].append((t_id, h_id, 1))
                    links['count'][0] += 1
                    links['count'][1] += 1
                    links['total'] += 2

        with open(os.path.join(self.path, 'nei_s.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split(' ')
                h_id, t_id = int(th[0]), int(th[1])
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 2):
                    links['data'][2].append((h_id, t_id, 1))
                    links['data'][3].append((t_id, h_id, 1))
                    links['count'][2] += 1
                    links['count'][3] += 1
                    links['total'] += 2

        with open(os.path.join(self.path, 'nei_l.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split(' ')
                h_id, t_id = int(th[0]), int(th[1])
                h_type = self.get_node_type(h_id)
                t_type = self.get_node_type(t_id)
                if (t_type == 3):
                    links['data'][4].append((h_id, t_id, 1))
                    links['data'][5].append((t_id, h_id, 1))
                    links['count'][4] += 1
                    links['count'][5] += 1
                    links['total'] += 2

        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id], self.nodes['total'], self.nodes['total'])
        links['data'] = new_data

        return links

    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        nodes['total'] = 3913

        nodes['count'][0] = 2614
        nodes['count'][1] = 1286
        nodes['count'][2] = 4
        nodes['count'][3] = 9

        nodes['shift'][0] = 0
        nodes['shift'][1] = 2614
        nodes['shift'][2] = 3900
        nodes['shift'][3] = 3904

        nodes['attr'] = None
        return nodes


def data_loader(dataset):
    if dataset == "acm":
        return data_loader_acm('./data/' + 'acm')
    elif dataset == "dblp":
        return data_loader_dblp('./data/' + 'dblp')
    elif dataset == "imdb":
        return data_loader_imdb('./data/' + 'imdb')
    elif dataset == "yelp":
        return data_loader_yelp('./data/' + 'yelp')