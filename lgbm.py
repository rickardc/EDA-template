import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Example: Generate synthetic time series data
np.random.seed(42)
n_samples = 2000
X = np.random.randn(n_samples, 5)
y = np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples) * 0.1
y = (y > 0).astype(int)


# Convert to DataFrame for clarity
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
y = pd.Series(y, name='target')

data = pd.read_csv('./datasets/creditcard.csv')

X = data.drop(columns='Class')
y = data['Class']

print(f"len: {len(X)}")

# Create pipeline
pipeline = Pipeline([
    ('lgbm', LGBMClassifier(random_state=42, verbose=-1))
])

# Define hyperparameter grid
param_grid = {
    # 1. Tree Structure (Keep num_leaves proportional to max_depth)
    'lgbm__max_depth': [3, 6, 10],
    'lgbm__num_leaves': [15, 31, 63],

    # 2. Learning Speed vs Tree Capacity
    'lgbm__learning_rate': [0.01, 0.05, 0.1],
    'lgbm__n_estimators': [100, 200, 300],

    # 3. Regularization (Essential for noisy/time-series targets)
    'lgbm__reg_alpha': [0.0, 0.1, 1.0],     # L1 regularization
    'lgbm__reg_lambda': [0.0, 1.0, 10.0],    # L2 regularization

    # 4. Stochastic Sampling (Subsampling features & rows reduces overfitting)
    # 'lgbm__subsample': [0.7, 1.0],           # % of data rows sampled per tree
    # 'lgbm__colsample_bytree': [0.7, 1.0],    # % of features sampled per tree
    # 'lgbm__min_child_samples': [20, 50]      # Min data points in a leaf
}

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)

# GridSearchCV with time series CV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=tscv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Fit the model
print("Starting GridSearch with TimeSeriesSplit...")
grid_search.fit(X, y)

# Results
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best MSE (negative): {grid_search.best_score_:.4f}")
print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# Access best model
best_model = grid_search.best_estimator_
print(f"\nBest model:\n{best_model}")

# Feature importance from the best LightGBM model
feature_importance = best_model.named_steps['lgbm'].feature_importances_
print(f"\nFeature importances:\n{pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)}")
