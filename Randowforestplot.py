import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
import shap
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load data
data = pd.read_csv('West.csv')

# Split data into input and output variables
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Define the metrics
scoring = {
    'MSE': make_scorer(mean_squared_error),
    'MAE': make_scorer(mean_absolute_error),
    'R2': make_scorer(r2_score),
    'MAPE': make_scorer(mean_absolute_percentage_error)
}

# Define the cross-validation method
cv = KFold(n_splits=5, random_state=42, shuffle=True)

# Define the model
model = RandomForestRegressor(random_state=42)

# Evaluate model using cross-validation
scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
print("Cross-validation scores:")
for key, values in scores.items():
    print(f"{key}: {np.mean(values)}")

# Train model on the entire dataset
model.fit(X, y)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

#### Plot SHAP Summary Plot
#shap.summary_plot(shap_values, X, max_display=23, cmap=plt.get_cmap("viridis"))


####条形图
# 计算每个特征的平均绝对 SHAP 值
feature_importance = np.abs(shap_values).mean(axis=0)

# 对特征重要性进行排序
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = X.columns[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

# 绘制自定义条形图
plt.figure(figsize=(9.5, 4))
plt.barh(sorted_features[:12], sorted_importance[:12], color="olive")  # 修改颜色,行业用'darkolivegreen'，区域用olive
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('Feature Importance based on SHAP Values(West)')
plt.gca().invert_yaxis()
plt.show()

