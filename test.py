import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import os
from data_process import get_labels_for_data, detect_steady_and_anomaly
import torch
from gevent import pywsgi

app = Flask(__name__)

# 定义搜索范围
param_grid = {
    'gamma': np.logspace(-9, 3, 13),  # gamma取值范围为10的-9次方到10的3次方之间的13个对数均匀间隔值
    'nu': np.linspace(0.01, 0.1, 10)  # nu取值范围为0.01到0.1之间的10个均匀间隔值
}

def find_best_ocsvm_params(X, param_grid):
    # 手动进行交叉验证和超参数搜索
    best_f1 = -1  # 初始化最优F1分数
    best_params = {}  # 保存最优参数组合
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折交叉验证
    for gamma in param_grid['gamma']:
        for nu in param_grid['nu']:
            f1_scores = []  # 存储每折的F1分数
            ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
            for train_idx, test_idx in kf.split(X):
                ocsvm.fit(X[train_idx])  # 使用训练集进行训练
                y_pred = ocsvm.predict(X[test_idx])  # 在测试集上进行预测
                # 假设1为正样本（正常），-1为异常样本（异常）
                y_true = np.ones(len(test_idx))
                # 计算精确率、召回率和F1分数
                precision = precision_score(y_true, y_pred, pos_label=1)
                recall = recall_score(y_true, y_pred, pos_label=1)
                f1 = f1_score(y_true, y_pred, pos_label=1)
                f1_scores.append(f1)
            avg_f1 = np.mean(f1_scores)  # 计算平均F1分数
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = {'gamma': gamma, 'nu': nu}

    return best_params, best_f1


def train_and_save_ocsvm(data_path, model_path, gamma, nu):
    # 从CSV文件中读取数据
    data = pd.read_csv(data_path)
    X = data.values  # 创建特征数据集
    x_train, x_test = train_test_split(X, test_size=0.1, random_state=42)
    # 调用函数查找最优参数和评分
    #best_params, best_f1 = find_best_ocsvm_params(X, param_grid)

    # 创建OneClassSVM分类器
    #best_gamma = best_params['gamma']
    #best_nu = best_params['nu']
    #best_gamma = 1e-09
    #best_nu =0.02
    clf = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
    clf.fit(X)

    pred = clf.predict(x_test)
    score = 0
    for idx in range(x_test.shape[0]):
        if pred[idx] == 1:
            score += 1
    accuracy = score / x_test.shape[0]
    print('Accuracy:{:.2%}'.format(accuracy))

    # 保存模型
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    #print(f"Model trained and saved successfully to {model_path}")


def load_and_predict_ocsvm(model_path, test_data):
    # 加载模型
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    # 将 DataFrame 转换为 NumPy 数组
    X_test = test_data.values

    # 预测并输出结果
    predictions = clf.predict(X_test)
    return predictions


def predict(test_data_path):
    parameters = ["Power", "H", "T", "P", "t1", "p1", "t2", "p2", "t4", "p4", "m4", "m_gas", "CLCSO"]
    # Pre-class of data in 4 work status (in 3 kinds, observing result)
    # c1: Power, p2, m_gas
    # c2: T, t1, t2, t4
    # c3: P, p1, p4
    class_numb = 3
    c1 = [0, 7, 11]
    c2 = [2, 4, 6, 8]
    c3 = [3, 5, 10]
    class_list = [c1, c2, c3]

    # 训练和保存四个OCSVM模型
    label_files = ['label_0_data.csv', 'label_1_data.csv', 'label_2_data.csv', 'label_3_data.csv']

    # checking if exist saved models
    # if so, skip training process;
    # else, start training.
    files_to_check = []
    missing_files = []

    # defination of checking models name
    for i in range(3):
        files_to_check.append(f'clf_ocsvm_label_{i}.pickle')
        for j in range(0, class_numb):
            files_to_check.append(f'clf_ocsvm_label_{i}-{j}.pickle')

    for file_name in files_to_check:
        file_path = os.path.join("./model", file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    # missing models, start training
    if len(missing_files) != 0:
        print("missing models：", missing_files, "start training")

        # main models
        for i, label_file in enumerate(label_files):
            data_path = f'./data/{label_file}'
            model_path = f'./model/clf_ocsvm_label_{i}.pickle'
            train_and_save_ocsvm(data_path, model_path, 1e-09, 0.02)

        # classified models
        for i, label_file in enumerate(label_files):
            for j in range(class_numb):

                # choose selected columns to form new data(classfied)
                # ensure source data file
                input_file = f'./data/{label_file}'
                df = pd.read_csv(input_file)

                # select certain columns by class
                selected_col = class_list[j]
                selected_data = df.iloc[:, selected_col]

                # define new data file's name & write
                output_file_name = f'./data/class_data/{label_file[6]}-{str(j)}.csv'
                output_file = pd.DataFrame(selected_data)
                selected_data.to_csv(output_file_name, index=False)

                # start training part models
                data_path = output_file_name
                model_path = f'./model/clf_ocsvm_label_{i}-{j}.pickle'
                print("用", data_path, "训练", model_path, "完成")
                if j == 1:
                    train_and_save_ocsvm(data_path, model_path, 1e-06, 0.01)
                else:
                    train_and_save_ocsvm(data_path, model_path, 1e-09, 0.02)

    else:
        print("all models exists, start prediction")#训练好检测模型

    test_data = pd.read_csv(test_data_path)
    indicators = []
    indicators = detect_steady_and_anomaly(test_data_path)  # 判断是否是跳变数据
    labels = get_labels_for_data(test_data_path)  # 获取每个数据的label列表

    # print(labels)  # 输出labels列表

    # 调用对应的OCSVM模型检测
    all_predictions = []
    error_kind = []
    all_test_result = []
    log = []

    for i, label in enumerate(labels):
        if label < len(label_files):
            model_path = f'model/clf_ocsvm_label_{label}.pickle'
            row_data = pd.read_csv(test_data_path, nrows=1, skiprows=i)  # 仅读取第i行数据，判断是否有错误
            prediction = load_and_predict_ocsvm(model_path, row_data)
            all_predictions.append(prediction[0])  # 添加预测结果到all_predictions

            if prediction[0] == -1:  # 定位错误参数
                for j in range(0, class_numb):
                    model_path = f'model/clf_ocsvm_label_{label}-{j}.pickle'
                    row_data = pd.read_csv(test_data_path).iloc[:, class_list[j]]
                    prediction = load_and_predict_ocsvm(model_path, row_data)
                    if prediction[0] == -1:
                        if j == 0 & 1:
                            error_kind.append(j)
                        else:
                            error_kind.append(2)
                            # print("error", error_kind)

                all_test_result.append(error_kind)
                error_kind = []

            else:
                all_test_result.append("K")

        # 构建日志信息
        log_entry = {}
        log_entry["index"] = i
        if indicators[i] == 1:
            log_entry["status"] = "Non-stationary data"
            log_entry["details"] = f"Test data {i}: Non-stationary data, discarded"
        elif all_predictions[i] == 1:
            log_entry["status"] = "Normal"
            log_entry["details"] = f"Test data {i}: Condition type {labels[i]}, Working status: Normal"
        else:
            log_entry["status"] = "Abnormal"
            error = []
            for k in range(len(all_test_result[i])):  # k is numb of errors in test sample i
                for j in range(len(class_list[all_test_result[i][k]])):  # j is for output erroe list
                    # 第i个test结果中，第k个错误类型，对应用j来遍历错误列表添加至错误输出的数组
                    error.append(parameters[class_list[all_test_result[i][k]][j]])
            log_entry[
                "details"] = f"Test data {i}: Condition type {labels[i]}, Working status: Abnormal, Error types: {error}"
        log.append(log_entry)

    # 构建最终log_data字典
    log_data = {
        "predictions": all_predictions,
        "error_classification": all_test_result,
        "log": log
    }
    return log_data

# 在predict_endpoint()中调用predict()函数并构建response
@app.route('/predict_endpoint', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    test_data_path = './temp.csv'
    file.save(test_data_path) # 将上传的文件保存到服务器的临时路径
    result_data = predict(test_data_path)

    # 构建最终response字典
    response = {
        "data": result_data
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,port=8080)
