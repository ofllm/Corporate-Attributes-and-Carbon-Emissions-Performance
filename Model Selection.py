import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

# 读取数据
file_path = '原始数据.csv'
data = pd.read_csv(file_path)

# 数据预处理
data['Log_ListAge'] = np.log1p(data['ListAge'])
data['Log_Board'] = np.log1p(data['Board'])
data['Log_Balance'] = np.log1p(data['Balance'])
data['Log_Liquid'] = np.log1p(data['Liquid'])

# 使用Yeo-Johnson变换替代平方根变换
pt = PowerTransformer(method='yeo-johnson')
data['YJ_Cashflow'] = pt.fit_transform(data[['Cashflow']])

# 选择特征和目标变量
X = data[['Log_ListAge', 'Log_Board', 'Log_Balance', 'Log_Liquid', 'YJ_Cashflow'] + list(data.columns[5:12])]
y = data.iloc[:, 12]  # 目标变量假设在第13列

# Z-score变换
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()


# 定义评价函数
def evaluate_model_cv(model, X, y, cv):
    mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv)

    mse = -np.mean(mse_scores)
    mae = -np.mean(mae_scores)
    r2 = np.mean(r2_scores)
    mape = np.mean(np.abs((y - model.predict(X)) / y)) * 100

    return mse, mae, r2, mape


# 训练和评估模型
models = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    model.fit(X, y)
    results[name] = evaluate_model_cv(model, X, y, kf)

# 显示结果
results_df = pd.DataFrame(results, index=['MSE', 'MAE', 'R2', 'MAPE']).T
print(results_df)

# 保存结果到文件
results_df.to_csv('model_evaluation_results.csv')

