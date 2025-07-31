# House pricing predictor with multiple linear regression project!
# Data set link; https://www.kaggle.com/datasets/swarupsudulaganti/uk-house-price-prediction-dataset-2015-to-2024

# importing modules and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Grabbing Dataset from Kaggle
api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    "swarupsudulaganti/uk-house-price-prediction-dataset-2015-to-2024", path="./HP_data"
)
df = pd.read_csv("./HP_data/UK_House_Price_Prediction_dataset_2015_to_2024.csv")

# Adding year collum
df["Year"] = df["date"].str[:4]

# Adding a average house price collum for each propert type in an county
avg_price = df.groupby(["county", "property_type"])["price"].mean().reset_index()
avg_price.rename(columns={"price": "avg_price_county_type"}, inplace=True)
df = df.merge(avg_price, on=["county", "property_type"], how="left")

# One-hot encode 'property_type'
df = pd.get_dummies(df, columns=["property_type"], prefix="type", drop_first=True)
df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)

# One-hot encoding newbuild status
df["new_build"] = df["new_build"].map({"Y": 1, "N": 0})
df["freehold"] = df["freehold"].map({"F": 1, "L": 0})

# Log-transforming the price to make model handle skew better
df["log price"] = np.log1p(df["price"])

df = df.drop(
    ["date", "town", "street", "postcode", "locality", "county", "district"], axis=1
)
print(df.head())

# Remove top 1% most expensive properties
upper_limit = df["log price"].quantile(0.99)
df = df[df["log price"] <= upper_limit]


X = df.drop("log price", axis=1)
Y = df["log price"]

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=101
)


def run_linear_regression(X_train, X_test, y_train, y_test):
    # creating a regression model
    model = LinearRegression()

    # fitting the model
    model.fit(X_train, y_train)

    # make predictions
    log_predictions = model.predict(X_test)

    # Convert predictions back to pounds values
    predictions = np.expm1(log_predictions)
    actuals = np.expm1(y_test)

    # model evaluation
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = mse**0.5

    # Printing metric
    print("Linear Regression Model")
    print("Mean Squared Error (in £):", mse)
    print("Mean Absolute Error (in £):", mae)
    print("Root Mean Squared Error (in £):", rmse)
    return model


def run_random_forest(X_train, X_test, y_train, y_test):
    # Creating forst model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    # fitting the model
    model.fit(X_train, y_train)

    # make predictions
    log_predictions = model.predict(X_test)

    # Convert predictions back to pounds values
    predictions = np.expm1(log_predictions)
    actuals = np.expm1(y_test)
    # model evaluation
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = mse**0.5

    # Plot: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], "r--")
    plt.xlabel("Actual Price (£)")
    plt.ylabel("Predicted Price (£)")
    plt.title("Random Forest: Actual vs Predicted House Prices")
    plt.grid(True)
    plt.tight_layout()

    # Get feature importances and match with feature names
    importances = model.feature_importances_
    features = X_train.columns
    feat_importance = pd.Series(importances, index=features).sort_values(
        ascending=False
    )

    # Top 10 features as a named Series
    top_features = feat_importance.head(10)

    # Plot
    plt.figure(figsize=(8, 6))
    top_features.plot(kind="barh", color="steelblue")
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    # Printing metric
    print("Forest Regressor Model")
    print("Mean Squared Error (in £):", mse)
    print("Mean Absolute Error (in £):", mae)
    print("Root Mean Squared Error (in £):", rmse)
    return model, feat_importance


# Run both models to compare
# run_linear_regression(X_train, X_test, y_train, y_test)
model, feat_importance = run_random_forest(X_train, X_test, y_train, y_test)

# Removing low importance features and retraining to compare
low_importance_features = feat_importance[feat_importance < 0.01].index.tolist()
print("Candidates for removal:", low_importance_features)

# Drop low-importance features
X_reduced = X.drop(columns=low_importance_features)

# Train/test split
X_train_red, X_test_red, y_train, y_test = train_test_split(
    X_reduced, Y, test_size=0.3, random_state=101
)

# Run model
run_random_forest(X_train_red, X_test_red, y_train, y_test)

# Metrics
labels = ["MSE", "MAE", "RMSE"]
before = [220864138.09, 3121.74, 14861.50]
after = [54206051.29, 1129.32, 7362.48]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axs):
    ax.bar(["Before", "After"], [before[i], after[i]], color=["steelblue", "coral"])
    ax.set_title(labels[i])
    ax.set_ylabel("Error (£)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

fig.suptitle("Model Error Comparison: Before vs After Feature Removal")
plt.tight_layout()
# plt.show()
