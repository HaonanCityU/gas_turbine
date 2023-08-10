# coding: utf-8
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from _function.steady_state_detection import steady_division, steady_detection
import matplotlib as mpl

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
data1 = pd.read_csv('./data/3yue.csv')
data2 = pd.read_csv('./data/6yue.csv')
data3 = pd.read_csv('./data/9yue.csv')
data4 = pd.read_csv('./data/12yue.csv')

# In[1] 数据读取
d1 = data1.iloc[7199:14399]
d2 = data2.iloc[600:7200]
d3 = data3.iloc[7199:14399]
d4 = data4.iloc[7199:14399]
data = [d1,d2,d3,d4]
data_train = pd.concat(data,ignore_index = True)
power_train = data_train['Power']
boundary_train = data_train[['Power','T']]
# 压气机进口温度、压力 ；压气机出口温度、压力；透平出口温度、压力；排气流量；天然气流量，共8个变量
variables_train = data_train[['t1','p1','t2','p2','t4','p4','m4','m_gas']]

# 训练样本中稳态数据与非稳态数据的划分
index,delta_power = steady_division(power_train,interval = 20)
index_steady,index_unsteady = index
delta_power_steady,delta_power_unsteady = delta_power
# 利用标签index,取出稳态数据样本
data_steady = data_train.loc[index_steady]
boundary_steady = boundary_train.loc[index_steady]
variables_steady = variables_train.loc[index_steady]
power_steady = power_train.loc[index_steady]


def label_data(data_train, boundary_train):
    # 把聚类结果为0-3的数据筛选出来并打上标签
    working_condition = KMeans(n_clusters=4, random_state=42).fit(boundary_train)
    centers = working_condition.cluster_centers_  # 聚类中心点
    sorted_indices = np.argsort(np.sum(centers, axis=1))[::-1]  # 根据聚类中心的大小排序，返回排序后的索引
    labels = sorted_indices[working_condition.labels_]  # 为每个样本分配标签

    data_steady = data_train.loc[boundary_train.index]
    data_labels = [data_steady[labels == i] for i in range(4)]

    return data_labels

def get_min_values(label_count):
    #获得每个类别的区间
    label_data = {}  # 创建一个字典来保存标签数据
    min_values = []  # 创建一个列表来保存最小值

    for i in range(label_count):
        file_path = f'./data/label_{i}_data.csv'  # 构建文件路径
        label_data[i] = pd.read_csv(file_path)  # 加载标签数据

        first_column = label_data[i].iloc[:, 0]
        min_value = first_column.min()
        min_values.append(min_value)  # 将最小值添加到列表中

    return min_values  # 返回四个最小值的列表

def get_labels_for_data(file_path):
    labels = []

    data_column = pd.read_csv(file_path, usecols=[0]).values  # 只需要用到第一列
    intervals = get_min_values(4)
    for value in data_column:
        if value >= intervals[0] and value < intervals[1]:
            labels.append(0)
        elif value >= intervals[1] and value < intervals[2]:
            labels.append(1)
        elif value >= intervals[2] and value < intervals[3]:
            labels.append(2)
        elif value >= intervals[2] and value<=312:
            labels.append(3)
        else:
            print('Input Error!')

    return labels

def detect_steady_and_anomaly(file_path):
    #判断是否是稳态数据
    test_data = pd.read_csv(file_path)
    test_power = test_data['Power']
    unsteady_model = joblib.load('./model/unsteady_model.pkl')
    steady_model = joblib.load('./model/steady_model.pkl')
    steady_ratio = joblib.load('./model/steady_ratio.pkl')
    delta_power = np.array(test_power[1:]) - np.array(test_power[:-1])
    delta_power = delta_power.reshape(-1, 1)

    indicators = [0]
    for delta_power_value in delta_power:
        indicator = steady_detection(delta_power_value.reshape(-1, 1), steady_model, unsteady_model, steady_ratio)
        indicators.append(indicator)
    #稳态：输出0；非稳态：输出1
    return indicators


if __name__ == "__main__":

    #data_labels = label_data(data_steady, boundary_steady)
    #for i, data_steady_i in enumerate(data_labels):
    #   data_steady_i.to_csv(f'./data/label_{i}_data.csv', index=False)

    file_path ='./data/test.csv'
    labels = get_labels_for_data(file_path) # 获取每个数据的label列表
    indicates = detect_steady_and_anomaly((file_path))
    print(labels)    # 输出labels列表
    print(indicates)