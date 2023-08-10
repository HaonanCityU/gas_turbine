# 首先，定义物品的类
class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

# 然后，定义函数来求解背包问题
def knapsack(items, max_weight):
    n = len(items)

    # 初始化 dp 数组
    dp = [[0] * (max_weight + 1) for _ in range(n + 1)]

    # 按照动态规划的方法，求解背包问题
    for i in range(1, n + 1):
        for j in range(1, max_weight + 1):
            if j >= items[i - 1].weight:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - items[i - 1].weight] + items[i - 1].value)
            else:
                dp[i][j] = dp[i - 1][j]

    # 返回最优解
    return dp[n][max_weight]

# 定义背包中的物品
items = [Item(3, 12), Item(5, 15), Item(6, 12), Item(7, 14), Item(2, 5),Item(8, 15), Item(1, 10), Item(4, 17)]

# 求解背包问题
max_value = knapsack(items, 10)

# 输出最优解
#print(max_value)
print("最大价值总量为",max_value)
