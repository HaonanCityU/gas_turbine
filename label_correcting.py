import numpy as np

# 初始化网络拓扑结构
# edges[i][j] 表示从节点 i 到节点 j 的边的容量
n = 5
edges = np.zeros((n, n))
edges[0][1] = 10
edges[0][2] = 15
edges[1][2] = 25
edges[1][3] = 0
edges[2][1] = -20
edges[1][4] = 5
edges[3][4] = -5
edges[4][3] = 10
edges[4][2] = 30

# 初始化每个节点的标签值
labels = np.full(n, np.inf)
labels[0] = 0

# Label correcting 算法的核心部分
while True:
    updated = False

    # 更新所有节点的标签值
    for i in range(n):
        for j in range(n):
            if edges[i][j] > 0:
                new_label = labels[i] + edges[i][j]
                if labels[j] > new_label:
                    labels[j] = new_label
                    updated = True

    # 如果所有节点的标签值都不再更新，则结束迭代
    if not updated:
        break

# 输出最短路径
print("The shortest path from node 0 to node 4 is:", labels[4])
