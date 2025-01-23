import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder

# Load the default dataset
default_df = pd.read_csv("studentmat.csv")
# Dataset info and constants
DATASET_INFO = """
This dataset contains student grades and various demographic, social, and school-related features.
It can be used to predict final grades (G3) based on these features. The dataset has 395 rows and 33 columns.
Learn more about the dataset [here](https://archive.ics.uci.edu/dataset/320/student+performance).
"""
TARGET = "G3"
# -------------------------
# Helper Function for Metrics
# -------------------------
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2
    }

# -------------------------
# Visualization Helper Functions
# -------------------------
def plot_feature_importance(importances, preprocessor, X, scenario_name, model_name):
    """Plots feature importance for tree-based models."""
    categorical_features = preprocessor.transformers_[0][2]
    numeric_features = preprocessor.transformers_[1][2]
    feature_names = list(categorical_features) + list(numeric_features)

    sorted_idx = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=sorted_feature_names, palette="viridis")
    plt.title(f"Feature Importance for {model_name} ({scenario_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(plt)

def plot_residuals(y_true, y_pred, model_name, scenario_name):
    """Plots residuals for model predictions."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, color="purple")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.title(f"Residuals for {model_name} ({scenario_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    st.pyplot(plt)

# -------------------------
# Main App Logic
# -------------------------
def main():
    st.title("Student Performance Prediction with Enhanced Features")

    # Dataset info and constants
    DATASET_INFO = """
    This dataset contains student grades and various demographic, social, and school-related features.
    It can be used to predict final grades (G3) based on these features. The dataset has 395 rows and 33 columns.
    Learn more about the dataset [UCI](https://archive.ics.uci.edu/dataset/320/student+performance).
    """
    TARGET = "G3"
    # Display dataset information
    st.header("Dataset Information")
    st.markdown(DATASET_INFO)

    # Correlation Heatmap (Numeric Columns Only)
    st.header("Correlation Heatmap")
    numeric_df = default_df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    if st.button("Show Correlation Heatmap"):
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        st.pyplot(plt)

    # Dataset Exploration Tools
    st.header("Dataset Exploration Tools")
    st.subheader("Visualize Dataset Columns")
    column_to_explore = st.selectbox("Select a column to visualize", default_df.columns)
    visualization_type = st.radio(
        "Choose a visualization type:",
        ["Histogram", "Boxplot", "Scatterplot (with Target)", "Countplot (for Categorical Columns)"]
    )

    if st.button("Show Graph"):
        plt.figure(figsize=(8, 6))
        if visualization_type == "Histogram":
            if default_df[column_to_explore].dtype != "object":
                sns.histplot(default_df[column_to_explore], kde=True, color="blue")
                plt.title(f"Histogram of {column_to_explore}")
            else:
                st.warning("Histogram is only applicable for numeric columns.")
        elif visualization_type == "Boxplot":
            if default_df[column_to_explore].dtype != "object":
                sns.boxplot(y=default_df[column_to_explore], palette="coolwarm")
                plt.title(f"Boxplot of {column_to_explore}")
            else:
                st.warning("Boxplot is only applicable for numeric columns.")
        elif visualization_type == "Scatterplot (with Target)":
            if default_df[column_to_explore].dtype != "object":
                sns.scatterplot(x=default_df[column_to_explore], y=default_df['G3'], color="green")
                plt.title(f"Scatterplot of {column_to_explore} vs G3")
                plt.xlabel(column_to_explore)
                plt.ylabel("G3")
            else:
                st.warning("Scatterplot is only applicable for numeric columns.")
        elif visualization_type == "Countplot (for Categorical Columns)":
            if default_df[column_to_explore].dtype == "object":
                sns.countplot(y=default_df[column_to_explore], palette="viridis")
                plt.title(f"Countplot of {column_to_explore}")
            else:
                st.warning("Countplot is only applicable for categorical columns.")
        st.pyplot(plt)

    # Model Training and Performance
    st.header("Model Training and Performance")
    scenarios = {
        "Earliest Prevention": default_df.drop(columns=["G1", "G2", "G3"]),
        "Middle Intervention": default_df.drop(columns=["G2", "G3"]),
        "Last Stage Prediction": default_df.drop(columns=["G3"])
    }
    target = default_df["G3"]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
    }

    # Initialize variables for comparison
    all_results = {}

    # Train and Evaluate Models for Each Scenario
    for scenario_name, X in scenarios.items():
        st.subheader(f"{scenario_name} Scenario")
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns),
                ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns)
            ],
            remainder='passthrough'
        )

        scenario_results = []

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Calculate metrics
            metrics = evaluate_model(y_test, y_pred, model_name)
            scenario_results.append({
                "Model": model_name,
                "R²": metrics["R²"],
                "MAE": metrics["MAE"]
            })

            # Visualize Residuals
            if st.checkbox(f"Show Residuals for {model_name} ({scenario_name})"):
                plot_residuals(y_test, y_pred, model_name, scenario_name)

        all_results[scenario_name] = scenario_results

    # Enhanced Table Representation with AgGrid
    st.header("Interactive Model Comparison Table")
    comparison_data = []

    for scenario_name, results in all_results.items():
        for result in results:
            comparison_data.append({
                "Scenario": scenario_name,
                "Model": result["Model"],
                "R²": result["R²"],
                "MAE": result["MAE"]
            })

    comparison_df = pd.DataFrame(comparison_data)
    styled_df = comparison_df.style.background_gradient(cmap="coolwarm", subset=["R²", "MAE"])
    st.dataframe(styled_df, use_container_width=True)

    # Bar Graph for Model Performance
    st.header("Model Performance Comparison Across Scenarios")
    comparison_pivot = comparison_df.pivot(index="Model", columns="Scenario", values="R²")
    comparison_pivot.plot(kind="bar", figsize=(10, 6), colormap="viridis")
    plt.title("Model R² Scores Across Scenarios")
    plt.ylabel("R² Score")
    plt.xlabel("Model")
    st.pyplot(plt)


if __name__ == "__main__":
    main()