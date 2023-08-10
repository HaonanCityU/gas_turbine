import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# 1. 从CSV文件中读取数据
data = pd.read_csv('./data/label_data.csv')

# 2. 创建特征数据集
x_train, x_test = train_test_split(data, test_size=0.1, random_state=42)
# 3. 训练One-Class SVM模型
one_class_svm = OneClassSVM(kernel='rbf', gamma=1e-9, nu=0.02)  # nu是一个超参数，可以根据需要进行调整
one_class_svm.fit(x_train)

# 新进来的样本，假设是数据中的最后一行
new_sample = pd.DataFrame([[1035.22, 32.21, 22.23, 100.91, 21.91, 295.05, -0.07,  0.11, 394.31, 667.45, 1.11, 1.21, 1539.71, 1.21, 27.78, 300.93, 0.23, 0.11, 1089449.01, 47195.01, 80.94, 11.96, 0.81, 1.65, 0.33, 10822.01]])

# 4. 异常检测
prediction = one_class_svm.predict(x_test)
score = 0
for idx in range(x_test.shape[0]):
    if prediction[idx] == 1:
        score +=1
        #print("样本正常")
    else:
        print("样本异常")
        # 获取特征的重要性评分（使用随机森林算法）
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # 将目标标签设置为0（异常样本）
        y_train_rf = np.ones(x_test.shape[0]) * -1
        rf.fit(x_test, y_train_rf)  # 使用目标标签训练随机森林模型

        # 获取特征的重要性评分
        feature_importances = rf.feature_importances_

        # 将特征名称与其对应的重要性评分关联起来
        feature_scores_dict = dict(zip(data.columns, feature_importances))

        # 打印特征及其重要性评分
        #for feature, importance in feature_scores_dict.items():
            #print(f"特征 {feature} 的重要性评分：{importance:.5f}")  # 注意保留小数点后五位


accuracy = score / x_test.shape[0]
print('Accuracy:{:.2%}'.format(accuracy))

pred = one_class_svm.predict(new_sample)
print((pred))