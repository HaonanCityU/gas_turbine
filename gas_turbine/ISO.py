import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. 从CSV文件中读取数据
data = pd.read_csv('./data/label_data.csv')

# 2. 创建特征数据集
X = data.values

# 新进来的样本
new_sample = np.array([[145.22, 32.21, 22.23, 100.91, 21.91, 295.05, -0.07, 0.11, 394.31, 667.45, 1.11, 1.21, 1539.71, 1.21, 27.78, 300.93, 0.23, 0.11, 1089449.01, 47195.01, 80.94, 11.96, 0.81, 1.65, 0.33, 10822.01]])

# 3. 训练Isolation Forest模型
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
isolation_forest.fit(X)

# 4. 异常检测
prediction = isolation_forest.predict(new_sample)

if prediction[0] == 1:
    print("样本正常")
else:
    print("样本异常")

    # 获取新样本的异常评分
    new_sample_score = isolation_forest.score_samples(new_sample)
    print(new_sample_score)

    # 将特征名称与其对应的异常评分关联起来，并按异常评分排序
    feature_scores_dict = dict(zip(data.columns, new_sample_score))
    sorted_feature_scores = sorted(feature_scores_dict.items(), key=lambda x: x[1], reverse=True)

    # 打印异常参数及其异常评分
    for feature, score in sorted_feature_scores:
        print(f"特征 {feature} 的异常评分：{score:.5f}")  # 注意保留小数点后五位
