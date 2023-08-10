# -*- coding: utf-8 -*-
"""
Module for working condition classification
KMeans聚类
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score
from collections import defaultdict

def clustering_criterion(boundary,labels):   
    """
    计算每个样本的平均轮廓系数，度量距离为马氏距离
    """
    silhouette_avg = silhouette_score(boundary,labels, metric = 'mahalanobis')
    
    return silhouette_avg

def condition_clustering_kmeans(boundary,number_upper = 10):
    """
    working condition based on the k-means,the clustering number optimized by silhouette criterion.
    
    Input:
    - boundary variables:power and ambient temperature and(ambient humidity).样本
    - number_upper：the number limit of clusters for optimization
    Output:
    - optimal_num_clusters: cluster number after optimization by silhouette 
    - labels: cluster labels of the training data
    - centers: cluster centers
    """
    range_n_clusters = range(2,number_upper) #寻优确定最佳工况分类数range(start, stop)
    silhouette_avg = np.zeros((len(range_n_clusters),2))  #创建一个x行x列的数组
    i = 0
    for num_clusters  in range_n_clusters:  
        clusters=KMeans(n_clusters=num_clusters).fit(boundary) 
        cluster_labels = clusters.labels_      
        silhouette_avg[i,1] = silhouette_score(boundary, cluster_labels)  # 计算样本在silhouette准则下的得分
        silhouette_avg[i,0] = num_clusters #第0列存num_clusters, 第1列存silhouette的平均值
        i = i+1
    index_max = np.argmax(silhouette_avg[:,1])#找出最大得分
    optimal_num_clusters = silhouette_avg[index_max,0].astype(int)  #找到最佳聚类数
    # K-means under the optimal_num_clusters
    working_condition = KMeans(n_clusters=optimal_num_clusters).fit(boundary) 
    labels = working_condition.labels_    # 聚类样本标签
    centers = working_condition.cluster_centers_   #聚类中心点
    return optimal_num_clusters,labels,centers,silhouette_avg   # 最佳聚类数，标签, 聚类中心，轮廓系数平均值



def condition_clustering_MultistepK(boundary, number_up = 10):
    """
    number_up：the number limit in each step cluster
    """   
    number = defaultdict(list)     #记录每个子簇的数量
    sub_cluster = defaultdict(list)#记录每个子簇的数据
    k = 0
    name = boundary.columns
    sub_cluster_0 = boundary
    n_0,labels_0,centers_0,s_0 = condition_clustering_kmeans(np.array(sub_cluster_0[name[0]]).reshape(-1,1),number_upper = number_up)
    for i in range(n_0):
        sub_cluster_1 = sub_cluster_0[labels_0 == i]
        n_1,labels_1,centers_1,s_1 = condition_clustering_kmeans(np.array(sub_cluster_1[name[1]]).reshape(-1,1),number_upper = number_up)
        number[str(i)].append(n_1)
        for j in range(n_1):
            sub_cluster_2 = sub_cluster_1[labels_1 == j]
            sub_cluster[str(k)].append(sub_cluster_2)
            k = k+1
    return number,sub_cluster