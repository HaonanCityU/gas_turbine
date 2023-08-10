import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

#划分训练集，测试集，验证集
dataset = pd.read_csv('./data/label_data.csv')
dataset = dataset.values
x_train, x_test = train_test_split(dataset, test_size=0.1, random_state=42)
x_test = x_train
# 创建一个IsolationForest模型
clf = IsolationForest(contamination=0.02)  # 可以调整contamination参数来控制异常数据的比例
# 训练模型
clf.fit(x_train)

#计算准确率
pred = clf.predict(x_test)#预测数据
print(pred)
score = 0
for idx in range(x_test.shape[0]):
    if pred[idx] == 1:
        score += 1
accuracy = score / x_test.shape[0]
print('Accuracy:{:.2%}'.format(accuracy))

#保存模型
with open('model/clf_iso.pickle','wb') as f: pickle.dump(clf,f)

#加载模型
with open('model/clf_iso.pickle','rb') as f: clf_new = pickle.load(f)
new_sample = pd.read_csv('./data/test.csv')
print((new_sample))
new_sample = new_sample.values
#print(new_sample)
print(clf_new.predict(new_sample))
