import numpy as np

# 表格的初始化
# 输入一个初始的表
d = np.array([[2, 1, 0, 0, 0, 0], [0, 5, 1, 0, 0, 15], [6, 2, 0, 1, 0, 24], [1, 1, 0, 0, 1, 5]]).astype(float)
# 初始的参数表进行分离
c = d[0, :-1]
a = d[1:, :-1]
b = d[1:, -1]
# 确定基向量
j = [2, 3, 4]  # 基向量


# 进行循环
def pivot(j, a, b):
    Cb = [c[i] for i in j]  # 更新Cb

    # 对sigma进行求解
    sigma = np.array([0] * a.shape[1]).astype(float)  # 求解sigma，用做后续的判定
    for i in range(a.shape[1]):
        temp = 0
        for k in range(a.shape[0]):
            temp += a[k, i] * Cb[k]
        sigma[i] = c[i] - temp

    # 得到换入变量，从sigam中最大值的索引为换入基变量
    vetorIn = np.argmax(sigma)  # 这里的vetorIn是横着的索引

    # 得到换出基变量
    theta = []
    for i in range(len(b)):
        if a[i, vetorIn] == 0:  # 如果为0，则不能当作除数
            theta.append(10000000000)
        else:
            theta.append(b[i] / a[i, vetorIn])
    index = np.argmin(theta)  # 确定换出基变量，这里的index时竖着的索引
    vetorOut = j[np.argmin(theta)]  # 这里得到的是换出基变量，但是后面一般都用其索引index

    # 行变换，进行高斯变换
    temp = a[index, vetorIn]
    a[index, :] = a[index, :] / a[index, vetorIn]  # 对换出基向量进行变换-----矩阵A
    b[index] = (b[index] / temp)  # 对换出基向量进行变换+-----举证b
    for i in range(a.shape[0]):  # 对其他的行向量高斯变换
        if i != index:
            if a[i, vetorIn] != 0:  # 如果为换入基向量列中有0，则无需变换
                temp = a[i, vetorIn]
                a[i, :] = a[i, :] - a[i, vetorIn] * a[index, :]
                b[i] = b[i] - temp * b[index]

    # 基向量的替换
    j[index] = vetorIn  # 基变量
    return j, a, b, sigma


"最优性检验"


def check(j, a, b, c, d):
    """
    这里的主要的求解步骤
    :param j: 基向量的索引
    :param a: 约束中的矩阵
    :param b: 约束中的条件
    :param c: 目标函数中的c
    :param d: 初始化的表
    :return:返回基向量，a，b
    """
    sigma = np.array([1, 1, 1, 1, 1]).astype(float)
    while (sigma > 0).any():
        j, a, b, sigma = pivot(j, a, b)
    return j, a, b


"解的输出"


def output(j, a, b):
    result = [0] * len(b)
    for i in range(len(b)):
        result[j[i]] = b[i]
    return result


if __name__ == '__main__':
    J, A, B = check(j, a, b, c, d)
    Result = output(J, A, B)
    for i in range(len(Result)):
        print('x', str(i + 1), '为', Result[i])