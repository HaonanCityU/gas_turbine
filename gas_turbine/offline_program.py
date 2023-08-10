# coding: utf-8
"""
Off_line Module for training the models before online monitoring.
1. 数据的读取
2. 数据的清洗与划分：功率、边界条件、特征变量
3. 历史数据的稳态划分
4. 稳态模型与非稳态模型的训练
5. 工况划分
6. 基准值回归模型的训练
7. 结果展示：画图
"""

import sys
sys.path.append('E:/Python/gas_turbine')
import numpy as np
import pandas as pd
from _function.steady_state_detection import steady_division,steady_training 
from _function.working_condition_classification import condition_clustering_MultistepK
from _function.reference_determination import reference_regression
from _function.fault_detection import network_construction

# In[1] 数据读取
data1 = pd.read_csv('./data/3yue_z.csv')
data2 = pd.read_csv('./data/6yue_z.csv')
data3 = pd.read_csv('./data/9yue_z.csv')
data4 = pd.read_csv('./data/12yue_z.csv')
d1 = data1.iloc[7199:14399]
d2 = data2.iloc[0:7200]
d3 = data3.iloc[7199:14399]
d4 = data4.iloc[7199:14399]
data = [d1,d2,d3,d4]
data_train = pd.concat(data,ignore_index = True)
# data classfication by power,boudary,and characteristic variables
power_train = data_train['Power']
boundary_train = data_train[['Power','T']]
# 压气机进口温度、压力 ；压气机出口温度、压力；透平出口温度、压力；排气流量；天然气流量，共8个变量        
variables_train = data_train[['t1','p1','t2','p2','t4','p4','m4','m_gas']]

# In[2] 稳态划分
# 训练样本中稳态数据与非稳态数据的划分
index,delta_power = steady_division(power_train,interval = 20)
index_steady,index_unsteady = index
delta_power_steady,delta_power_unsteady = delta_power
# 利用标签index,取出稳态数据样本
data_steady = data_train.loc[index_steady]
boundary_steady = boundary_train.loc[index_steady]
variables_steady = variables_train.loc[index_steady] 
power_steady = power_train.loc[index_steady]


# In[3]稳态与非稳态模型的训练
steady_training(delta_power_steady,delta_power_unsteady, number = 5)

# In[4] 稳态下的工况划分
number_M,clusters_M = condition_clustering_MultistepK(boundary_steady,number_up = 10)#聚类
length = len(clusters_M)
print('length长度为',length)
ref = {'power':np.zeros(length),'H':np.zeros(length),'T':np.zeros(length),'P':np.zeros(length),
     't1':np.zeros(length),'p1':np.zeros(length),'t2':np.zeros(length),'p2':np.zeros(length),
     't4':np.zeros(length),'p4':np.zeros(length),'m4':np.zeros(length),'m_gas':np.zeros(length),
     'c_pai':np.zeros(length),'c_efficiency':np.zeros(length),'t_efficiency':np.zeros(length),'h_efficiency':np.zeros(length),'h_consumption':np.zeros(length)}
std = {'power':np.zeros(length),'H':np.zeros(length),'T':np.zeros(length),'P':np.zeros(length),
     't1':np.zeros(length),'p1':np.zeros(length),'t2':np.zeros(length),'p2':np.zeros(length),
     't4':np.zeros(length),'p4':np.zeros(length),'m4':np.zeros(length),'m_gas':np.zeros(length),
     'c_pai':np.zeros(length),'c_efficiency':np.zeros(length),'t_efficiency':np.zeros(length),'h_efficiency':np.zeros(length),'h_consumption':np.zeros(length)}
for i in range(length):
     clusters= data_steady.loc[clusters_M[str(i)][0].index]
     Mean = clusters.astype(float).mean() 
     Standard = clusters.astype(float).std()
     # 计算数据期望
     ref['power'][i] = Mean['Power']
     ref['H'][i] = Mean['H']
     ref['T'][i] = Mean['T']
     ref['P'][i] = Mean['P']
     ref['t1'][i] = Mean['t1']
     ref['p1'][i] = Mean['p1']
     ref['t2'][i] = Mean['t2']
     ref['p2'][i] = Mean['p2']
     ref['t4'][i] = Mean['t4']
     ref['p4'][i] = Mean['p4']
     ref['m4'][i] = Mean['m4']
     ref['m_gas'][i] = Mean['m_gas'] 
     ref['c_pai'][i] = Mean['pai'] 
     ref['c_efficiency'][i] = Mean['c_efficiency']
     ref['t_efficiency'][i] = Mean['t_efficiency']
     ref['h_efficiency'][i] = Mean['h_efficiency']
     ref['h_consumption'][i] = Mean['h_consumption']    
     # 计算数据标准差
     std['power'][i] = Standard['Power']
     std['H'][i] = Standard['H']
     std['T'][i] = Standard['T']
     std['P'][i] = Standard['P']
     std['t1'][i] = Standard['t1']
     std['p1'][i] = Standard['p1']
     std['t2'][i] = Standard['t2']
     std['p2'][i] = Standard['p2']
     std['t4'][i] = Standard['t4']
     std['p4'][i] = Standard['p4']
     std['m4'][i] = Standard['m4']
     std['m_gas'][i] = Standard['m_gas']
     std['c_pai'][i] = Standard['pai'] 
     std['c_efficiency'][i] = Standard['c_efficiency']
     std['t_efficiency'][i] = Standard['t_efficiency']
     std['h_efficiency'][i] = Standard['h_efficiency']
     std['h_consumption'][i] = Standard['h_consumption']   
     
# In[5]基准值回归模型
reference_value = pd.DataFrame(ref)
reference_std = pd.DataFrame(std)
reference_value.to_csv('./data/reference_samples.csv')#把数学期望和标准差存入csv文件
reference_std.to_csv('./data/reference_std_samples.csv')
for v in ['t1', 'p1', 't2', 'p2', 't4', 'p4', 'm4','m_gas','c_pai','c_efficiency','t_efficiency','h_efficiency','h_consumption']:
     model_ref,coef_ref,intercept_ref,mse_ref,r2_ref = reference_regression('ref',v,reference_value[['power','T']],reference_value[v],2)
     model_std,coef_std,intercept_std,mse_std,r2_std = reference_regression('std',v,reference_value[['power','T']],reference_std[v],1)  # 注意：功率基准值和参数方差回归
# In[6]贝叶斯网络模型训练
fault_model = network_construction() 



