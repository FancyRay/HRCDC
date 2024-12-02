import numpy as np


temp = np.load(
    '../../../data/IMDB/0/0-1-0_idx.npy', allow_pickle=True)
temp1 = np.load(
    '../../../data/IMDB/0/0-2-0_idx.npy', allow_pickle=True)

#
# dict_nei_d = {}
# for i in range(4278):
#     dict_nei_d[i] = set()
# for a in temp:
#     start = a[0]
#     mid = a[1]
#     end = a[2]
#     dict_nei_d[start].add(mid)
#     dict_nei_d[end].add(mid)
#
# dict_nei_a = {}
# for i in range(4278):
#     dict_nei_a[i] = set()
# for a in temp1:
#     start = a[0]
#     mid = a[1]
#     end = a[2]
#     dict_nei_a[start].add(mid)
#     dict_nei_a[end].add(mid)
#
# file_name = 'IMDBnei_d.txt'
# file_name1 = 'IMDBnei_a.txt'
# #
# # # 打开文件以写入数据
# with open(file_name, 'w') as file:
#     # 遍历字典中的键值对
#     for key, value_set in dict_nei_d.items():
#         # 对于集合中的每个元素，将键和元素写入文件
#         for element in value_set:
#             file.write(f'{key} {element}\n')
#
# with open(file_name1, 'w') as file:
#     # 遍历字典中的键值对
#     for key, value_set in dict_nei_a.items():
#         # 对于集合中的每个元素，将键和元素写入文件
#         for element in value_set:
#             file.write(f'{key} {element}\n')
#
#
# # dict_nei_s = {}
# # for i in range(2614):
# #     dict_nei_s[i] = set()
# # for a in temp1:
# #     start = a[0]
# #     mid = a[1]
# #     end = a[2]
# #     dict_nei_s[start].add(mid)
# #     dict_nei_s[end].add(mid)
# #
# # file_name = 'nei_s.txt'
# #
# # # 打开文件以写入数据
# # with open(file_name, 'w') as file:
# #     # 遍历字典中的键值对
# #     for key, value_set in dict_nei_s.items():
# #         # 对于集合中的每个元素，将键和元素写入文件
# #         for element in value_set:
# #             file.write(f'{key} {element}\n')
# #
# # dict_nei_l = {}
# # for i in range(2614):
# #     dict_nei_l[i] = set()
# # for a in temp2:
# #     start = a[0]
# #     mid = a[1]
# #     end = a[2]
# #     dict_nei_l[start].add(mid)
# #     dict_nei_l[end].add(mid)
# #
# # file_name1 = 'nei_l.txt'
# #
# # # 打开文件以写入数据
# # with open(file_name1, 'w') as file:
# #     # 遍历字典中的键值对
# #     for key, value_set in dict_nei_l.items():
# #         # 对于集合中的每个元素，将键和元素写入文件
# #         for element in value_set:
# #             file.write(f'{key} {element}\n')
#
import pickle
with open('../../../data/imdbhgt/node_features.pkl', 'rb') as f:
    node_features = pickle.load(f)
with open('../../../data/imdbhgt/edges.pkl' , 'rb') as f:
    edges = pickle.load(f)
with open('../../../data/imdbhgt/labels.pkl' , 'rb') as f:
    labels = pickle.load(f)

labels = 0

