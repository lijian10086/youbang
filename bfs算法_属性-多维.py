
"""
https://blog.csdn.net/hezuijiudexiaobai/article/details/104451980
MovieLens 1M 数据集
https://cloud.tencent.com/developer/article/1474293
利用 Python 分析 MovieLens 1M 数据集
https://blog.csdn.net/Sakura55/article/details/81364961?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link
项目实战----基于协同过滤的电影推荐系统
https://blog.csdn.net/qq_25948717/article/details/81839463
超级详细的协同过滤推荐系统+完整Python实现及结果

https://zhuanlan.zhihu.com/p/152005338
用Python实现基于物品的协同过滤（附代码）
"""

# 基础
import numpy as np # 处理数组
import pandas as pd # 读取数据&&DataFrame
import matplotlib.pyplot as plt # 制图
import seaborn as sns
from matplotlib import rcParams # 定义参数

import warnings
warnings.filterwarnings('ignore') # 忽略警告信息
np.set_printoptions(precision=4) # 小数点后
import pickle
import random
import queue

data_train = pd.read_excel('data_train.xlsx',engine='openpyxl')
data_test = pd.read_excel('data_test.xlsx',engine='openpyxl')
# data_train = data_train.drop_duplicates(subset=['user_id'], keep='first')
# data_test = data_test.drop_duplicates(subset=['user_id'], keep='first')

#对属性进行改名，增加属性名字

data_train['user_id'] = data_train['user_id'].map(lambda x:'user_id#'+str(x))
data_test['user_id'] = data_test['user_id'].map(lambda x:'user_id#'+str(x))

data_train['movie_id'] = data_train['movie_id'].map(lambda x:'movie_id#'+str(x))
data_test['movie_id'] = data_test['movie_id'].map(lambda x:'movie_id#'+str(x))

data_train['gender'] = data_train['gender'].map(lambda x:'gender#'+str(x))
data_test['gender'] = data_test['gender'].map(lambda x:'gender#'+str(x))

data_train['age'] = data_train['age'].map(lambda x:'age#'+str(x))
data_test['age'] = data_test['age'].map(lambda x:'age#'+str(x))

data_train['occupation'] = data_train['occupation'].map(lambda x:'occupation#'+str(x))
data_test['occupation'] = data_test['occupation'].map(lambda x:'occupation#'+str(x))

data_train['zip'] = data_train['zip'].map(lambda x:'zip#'+str(x))
data_test['zip'] = data_test['zip'].map(lambda x:'zip#'+str(x))

data_train['year'] = data_train['year'].map(lambda x:'year#'+str(x))
data_test['year'] = data_test['year'].map(lambda x:'year#'+str(x))

data_train['genres'] = data_train['genres'].map(lambda x:'genres#'+str(x).split('|')[0])
data_test['genres'] = data_test['genres'].map(lambda x:'genres#'+str(x).split('|')[0])

user_list = list(data_train['movie_id'].unique())
product_list = list(data_train['movie_id'].unique())
gender_list =  list(data_train['gender'].unique())
age_list =  list(data_train['age'].unique())
occupation_list =  list(data_train['occupation'].unique())
zip_list =  list(data_train['zip'].unique())
year_list =  list(data_train['year'].unique())

##################构建共现矩阵
# pp_graph = dict()
# df_tmp = data_train[data_train['label']==1].head(1)
# user_cols = ['gender','age','occupation','zip']
# movie_cols = ['year''genres']
#
# all_cols = ['user_id','gender','age','occupation','zip','year','genres','movie_id']
# for col0 in all_cols[:-1]:
#     r_pid = df_tmp[col0].values[0]
#     pp_graph[r_pid] = dict()
#     for col1 in all_cols[1:]:
#         pid = df_tmp[col1].values[0]
#         if pid not in pp_graph[r_pid]:
#             pp_graph[r_pid][pid] = 1
#         else:
#             pp_graph[r_pid][pid] += 1
#
#         if pid not in pp_graph:
#             pp_graph[pid] = dict()
#         if r_pid not in pp_graph[pid]:
#             pp_graph[pid][r_pid] = 1
#         else:
#             pp_graph[pid][r_pid] += 1
# print('123')


pp_graph = dict()
def get_pp_graph1(df_tmp):
    ######item_bfs: hit1_rate= 0.1285  hit3_rate= 0.1844
    all_cols = ['user_id', 'gender', 'age', 'occupation', 'zip', 'year', 'genres', 'movie_id']
    for col0 in all_cols[:-1]:
        r_pid = df_tmp[col0].values[0]
        pp_graph[r_pid] = dict()
        for col1 in all_cols[1:]:
            pid = df_tmp[col1].values[0]
            if pid not in pp_graph[r_pid]:
                pp_graph[r_pid][pid] = 1
            else:
                pp_graph[r_pid][pid] += 1

            if pid not in pp_graph:
                pp_graph[pid] = dict()
            if r_pid not in pp_graph[pid]:
                pp_graph[pid][r_pid] = 1
            else:
                pp_graph[pid][r_pid] += 1

def get_pp_graph(df_tmp):
    ######item_bfs: hit1_rate= 0.1285  hit3_rate= 0.1844
    if df_tmp.shape[0] ==1:
        all_cols = ['user_id', 'gender', 'age', 'occupation', 'zip', 'year', 'genres', 'movie_id']
        for col0 in all_cols[:-1]:
            r_pid = df_tmp[col0].values[0]
            pp_graph[r_pid] = dict()
            for col1 in all_cols[1:]:
                pid = df_tmp[col1].values[0]
                if pid not in pp_graph[r_pid]:
                    pp_graph[r_pid][pid] = 1
                else:
                    pp_graph[r_pid][pid] += 1

                if pid not in pp_graph:
                    pp_graph[pid] = dict()
                if r_pid not in pp_graph[pid]:
                    pp_graph[pid][r_pid] = 1
                else:
                    pp_graph[pid][r_pid] += 1
    else:
        ######item_bfs: hit1_rate= 0.0615  hit3_rate= 0.162
        # for i in range(df_tmp.shape[0]-1):
        #     all_cols = ['user_id', 'gender', 'age', 'occupation', 'zip', 'year', 'genres', 'movie_id']
        #     for col0 in all_cols[:-1]:
        #         r_pid = df_tmp[col0].values[i]
        #         pp_graph[r_pid] = dict()
        #         for col1 in all_cols[1:]:
        #             pid = df_tmp[col1].values[i+1]
        #             if pid not in pp_graph[r_pid]:
        #                 pp_graph[r_pid][pid] = 1
        #             else:
        #                 pp_graph[r_pid][pid] += 1
        #
        #             if pid not in pp_graph:
        #                 pp_graph[pid] = dict()
        #             if r_pid not in pp_graph[pid]:
        #                 pp_graph[pid][r_pid] = 1
        #             else:
        #                 pp_graph[pid][r_pid] += 1

        for i in range(df_tmp.shape[0]):
            all_cols = ['user_id', 'gender', 'age', 'occupation', 'zip', 'year', 'genres', 'movie_id']
            for col0 in all_cols[:-1]:
                r_pid = df_tmp[col0].values[i]
                pp_graph[r_pid] = dict()
                for col1 in all_cols[1:]:
                    pid = df_tmp[col1].values[i]
                    if pid not in pp_graph[r_pid]:
                        pp_graph[r_pid][pid] = 1
                    else:
                        pp_graph[r_pid][pid] += 1

                    if pid not in pp_graph:
                        pp_graph[pid] = dict()
                    if r_pid not in pp_graph[pid]:
                        pp_graph[pid][r_pid] = 1
                    else:
                        pp_graph[pid][r_pid] += 1




tmp1 = data_train[data_train['label'] == 1].groupby('user_id').apply(get_pp_graph)

#计算item相似度
item_sim_dict = {}

def takeSecond(elem): # 获取列表的第二个元素
	return elem[1]

for i in range(len(product_list)):
	a = product_list[i]
	if a in pp_graph:
		item_sim_dict[a] = []
		print('###计算item=', a)
		for b,a_b_sim in pp_graph[a].items():
			item_sim_dict[a].append((b,a_b_sim))

		# 根据item相似度,由大到小排序
		item_sim_dict[a].sort(key=takeSecond, reverse=True)

def prune_the_graph(graph):
    """
    # 1，剪掉weight=1的边
    2，剪掉自己到自己的边
    # 3，weight平滑化 (为了凸显不同数据类型的权重，不需要做平滑化)
    """
    vertex = set()
    g = dict()
    for p_i, p_j_set in graph.items():
        vertex.add(p_i)
        g[p_i] = dict()
        for p_j, w in p_j_set.items():
            if p_i == p_j:
                continue
            else:
                g[p_i][p_j] = w
            # if w == 1:
            #     g[p_i][p_j] = 0.1
            # else:
            #     g[p_i][p_j] = round(log(w, 2), 3)
            vertex.add(p_j)
        if not g[p_i]:
            del g[p_i]
    return g, list(vertex)

def bfs_rank(root, pp_graph, max_count=500): #root为目标商品，要对目标商品召回1000个
    res_dict = dict()
    q = queue.Queue()
    q_set = set()       # q的集合形式，用于判断新元素是否已经存在于队列中
    # 初始化
    p_dict = pp_graph[root] #对有向共现矩阵（邻接表）,取出目标商品的带权重邻居表——》问题：如果某个目标sku，不在图r_pid中怎么办？冷启动解决
    sorted_list = sorted(p_dict.items(), key=lambda x: x[1], reverse=True) #根据权重排序,整理邻居
    sum_w = 0
    for i, w in sorted_list: #i是邻居(召回商品id),w是邻居权重（就是边）
        sum_w += w
    for i, w in sorted_list:
        if i not in q_set: #如果是未访问过的顶点(召回商品)
            q.put((i, w/sum_w)) #作为节点进入队列,用于做下一层的节点搜索的父节点
            q_set.add(i)
    for i, w in sorted_list:
        res_dict[i] = w / sum_w  #取目标商品root的第一层子节点

    while len(res_dict) < max_count and not q.empty(): #如果不够，取第二层子节点
        p, w0 = q.get()
        if p not in pp_graph: #第一层子节点如果没有子节点，跳过
            continue
        p_dict = pp_graph[p] #取出第一层子节点的子节点
        sorted_list = sorted(p_dict.items(), key=lambda x: x[1], reverse=True)
        sum_w = 0
        for i, w in sorted_list:
            sum_w += w
        for i, w in sorted_list:
            if i not in q_set:#未访问过的顶点(召回商品)
                q.put((i, w/sum_w))
                q_set.add(i)
        for i, w in sorted_list:
            if i not in res_dict:#未访问过的顶点
                res_dict[i] = (w / sum_w) * w0
            else: #已经访问过的顶点，累加权重——》入度越多，出度和越小——》这个点越重要
                res_dict[i] += (w / sum_w) * w0
    sorted_dict = dict()
    sorted_dict[root] = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)[:10000]

    pids_recall = []
    for pid_recall,w in sorted_dict[root]:
        if pid_recall in  product_list: #只保留电影id
            pids_recall.append((pid_recall,round(w,5)))
    return pids_recall

    # str_list = list()
    # sorted_dict = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)[:200]
    # for ele in sorted_dict: #排序取前面的200个 问题：如果上面取完，还是不够200个怎么办，为了避免过少，直接过滤掉少于50个召回结果的商品
    #     str_list.append(str(ele[0]) + '|' + str(round(ele[1], 4)))
    # str_t = ' '.join(str_list)
    # return str(root) + ',' + str_t


G, vertex = prune_the_graph(pp_graph)
top_prods = list(G.keys())

item_sim_dict = {}
for r_pid_test in top_prods:
	item_sim_dict[r_pid_test] = bfs_rank(r_pid_test, G)
	print('排名：',r_pid_test,item_sim_dict[r_pid_test][:10])

#测试命中率

# data_test = pd.read_excel('data_test.xlsx',engine='openpyxl')
data_test = data_test[data_test['label']==1]

data_test['1_num'] = data_test.groupby('user_id')["label"].transform('count')
data_test = data_test[(data_test['1_num']>1)]
del data_test['1_num']

test_list = []

def get_test(x):
  a = x.values[0]
  b = x.values[1]
  test_list.append((a,b))
tmp = data_test.groupby('user_id')["movie_id"].apply(get_test)
product_list = list(data_train['movie_id'].unique())

# f1 = open('./item_bfs_sim.pk', 'rb')
# item_cf_sim = pickle.load(f1)
item_cf_sim = item_sim_dict

test_num = 0

#使用cf算法选商品
cf1_num = 0
cf3_num = 0

#随机选商品
ramdom1_num = 0
ramdom3_num = 0

#选热品
hot1_num = 0
hot3_num = 0
hot_list = []
hot_movie = []
for movie_id in product_list:
	num_all = data_train[data_train['movie_id']==movie_id].shape[0]
	num_1 = data_train[(data_train['movie_id'] == movie_id)&(data_train['label'] == 1)].shape[0]
	rate = 0
	if num_1>0:
		rate = num_1/(num_all+10)
	hot_list.append((movie_id,rate))
hot_list.sort(key=takeSecond,reverse=True)
hot_movie = [ x[0] for x in hot_list ][:3]

##开始测试
result_cf = []
result_hot = []
for (a,b) in test_list:
	if a in item_cf_sim:

		test_num+=1
		#CF测试
		top3 = [x[0] for x in item_cf_sim[a][:3]]
		if b in [top3[0]]:
			cf1_num += 1
			result_cf.append(b)
		if b in top3:
			cf3_num += 1

        #随机测试
		ramdom_chose = random.sample(product_list, 3)
		if b in [ramdom_chose[0]]:
			ramdom1_num += 1
		if b in ramdom_chose:
			ramdom3_num += 1

		#热品测试
		if b in [hot_movie[0]] :
			hot1_num += 1
			result_hot.append(b)
		if b in hot_movie:
			hot3_num += 1

hit1_rate = round(cf1_num/test_num,4)
hit3_rate = round(cf3_num/test_num,4)
print('######item_bfs: hit1_rate=',hit1_rate,' hit3_rate=',hit3_rate)

hit1_rate = round(ramdom1_num/test_num,4)
hit3_rate = round(ramdom3_num/test_num,4)
print('######ramdom: hit1_rate=',hit1_rate,' hit3_rate=',hit3_rate)

hit1_rate = round(hot1_num/test_num,4)
hit3_rate = round(hot3_num/test_num,4)
print('######hot: hit1_rate=',hit1_rate,' hit3_rate=',hit3_rate)

'''
['movie_id', 'year', 'age','gender']
######item_bfs_属性: hit1_rate= 0.1611  hit3_rate= 0.3222
######ramdom: hit1_rate= 0.0056  hit3_rate= 0.0111
######hot: hit1_rate= 0.1444  hit3_rate= 0.2444

######item_bfs: hit1_rate= 0.1285  hit3_rate= 0.1844
######ramdom: hit1_rate= 0.0112  hit3_rate= 0.0223
######hot: hit1_rate= 0.1453  hit3_rate= 0.2458

'''

print('123')





