import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load data
data = pd.read_csv('C31.csv')

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

# Compute mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
indices = np.argsort(mean_abs_shap_values)[::-1]  # Sort indices by SHAP values in descending order

# Get the top 5 features
top_features = [X.columns[indices[i]] for i in range(4)]

# Number of features to plot
n_features = len(top_features)
n_cols = 4  # We want to display 5 plots in a row
n_rows = 1  # Only one row

# Set the size of each subplot
single_plot_width = 3
single_plot_height = 2.5

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(single_plot_width * n_cols, single_plot_height * n_rows))
axs = axs.flatten()

# Plot Partial Dependence Plots for the top features
for i, feature in enumerate(top_features):
    PartialDependenceDisplay.from_estimator(model, X, features=[feature], ax=axs[i], grid_resolution=50, feature_names=X.columns)
    axs[i].set_ylabel('Predicted target value')

# Delete extra subplots if any
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()