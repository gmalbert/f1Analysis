import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Simulated example F1 dataset
# -----------------------------
data = {
    'driver': ['Verstappen', 'Hamilton', 'Leclerc', 'Russell', 'Norris'] * 20,
    'team': ['Red Bull', 'Mercedes', 'Ferrari', 'Mercedes', 'McLaren'] * 20,
    'circuit': ['Monza', 'Silverstone', 'Monaco', 'Spa', 'Bahrain'] * 20,
    'session': ['Q1', 'Q2', 'Q3', 'Race', 'Race'] * 20,
    'lap_number': np.random.randint(1, 60, 100),
    'track_temp': np.random.uniform(25, 45, 100),
    'air_temp': np.random.uniform(20, 35, 100),
    'lap_time': np.random.uniform(75, 100, 100)  # in seconds
}

df = pd.DataFrame(data)
print(df.head())

# -----------------------------
# Feature selection
# -----------------------------
features = ['driver', 'team', 'circuit', 'session', 'lap_number', 'track_temp', 'air_temp']
target = 'lap_time'

X = df[features]
y = df[target]

# -----------------------------
# Preprocessing
# -----------------------------
categorical_features = ['driver', 'team', 'circuit', 'session']
numerical_features = ['lap_number', 'track_temp', 'air_temp']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# -----------------------------
# Gradient Boosting Model Pipeline
# -----------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ))
])

# -----------------------------
# Split data & train
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

preprocessor = model.named_steps['preprocessor']
regressor = model.named_steps['regressor']

# Get feature names from each transformer
numeric_names = numerical_features
categorical_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Combine all feature names
all_feature_names = np.concatenate([numeric_names, categorical_names])

# Step 3: Get feature importances from the regressor
importances = regressor.feature_importances_
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
})

# Normalize to percentage
importance_df['Importance (%)'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()

# Sort by importance
importance_df = importance_df.sort_values('Importance (%)', ascending=False)

# Display
print(importance_df.head(15))  # Show top 15 features

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R^2 Score: {r2:.3f}")
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")


# Assume you have the final transformed features (after preprocessing)
# For example:
X_transformed = model.named_steps['preprocessor'].transform(X)
feature_names = np.concatenate([
    numerical_features,
    model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
])

# Convert to DataFrame
X_df = pd.DataFrame(X_transformed, columns=feature_names)

# Compute correlation matrix
corr_matrix = X_df.corr().abs()

# Show heatmap of correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", vmax=1.0, square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# -------------------------------------------
# 🔍 Feature Importance Breakdown (in %)
# -------------------------------------------

# Find highly correlated pairs
threshold = 0.9
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr = [(col1, col2, upper_triangle.loc[col1, col2])
             for col1 in upper_triangle.columns
             for col2 in upper_triangle.columns
             if pd.notnull(upper_triangle.loc[col1, col2]) and upper_triangle.loc[col1, col2] > threshold]

# Display results
if high_corr:
    print("Highly correlated feature pairs (r > 0.9):")
    for f1, f2, score in high_corr:
        print(f"{f1} <-> {f2}: {score:.3f}")
else:
    print("No highly correlated features found above the threshold.")

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importances = result.importances_mean

for i in importances.argsort()[::-1]:
    print(f"{all_feature_names[i]}: {importances[i]:.4f}")

# Get feature names from preprocessor

cat_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numerical_features, cat_names])

# Get importances
importances = model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
})
importance_df['Importance (%)'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()
importance_df = importance_df.sort_values(by='Importance (%)', ascending=False)

# Display top 15 features
print("\nTop Feature Importances:")
print(importance_df.head(15))

# Optional: Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'].head(15), importance_df['Importance (%)'].head(15))
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (% Contribution)")
plt.xlabel("Importance (%)")
plt.tight_layout()
plt.show()