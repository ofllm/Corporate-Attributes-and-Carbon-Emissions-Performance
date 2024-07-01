
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load data
data = pd.read_csv('原始数据.csv')

# Split data into input and output variables
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Define the model
model = RandomForestRegressor()
model.fit(X, y)  # 确保在这里调用 fit 方法


# 特征列表

features = ["Technology","Liquid", "Size", "Cashflow", "StrategyTarget", "Female", "Board", "Management", "ListAge", "Balance", "AssetGrowth", "AdvocacyTrain"]
# 计算行和列的数量
n_features = len(features)
n_cols = 4  # 每行显示4个图
n_rows = (n_features + n_cols - 1) // n_cols  # 计算行数

# 设置每个子图的宽度和高度
single_plot_width = 3
single_plot_height = 2.5

# 创建子图
fig, axs = plt.subplots(n_rows, n_cols, figsize=(single_plot_width * n_cols, single_plot_height * n_rows))
axs = axs.flatten()  # 将axs数组扁平化，方便索引

# 绘制 Partial Dependence Plot
for i, feature in enumerate(features):
    PartialDependenceDisplay.from_estimator(model, X, features=[feature], ax=axs[i], grid_resolution=50, feature_names=X.columns)
    #axs[i].set_title(f'Partial Dependence Plot of {feature}')
    axs[i].set_ylabel('Predicted target value')

# 删除多余的子图
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

