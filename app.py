
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Prediction System")

st.markdown(
    "A machine learning application using Support Vector Machine "
    "(SVM) to predict diabetes outcomes."
)

st.info(
    "Educational project only. This application is not a medical "
    "diagnostic tool."
)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv("diabetes.csv")

    return data


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("diabetes_model.pkl")

    return model


# --------------------------------------------------
# 4. LOAD DATA AND MODEL
# --------------------------------------------------

try:

    data = load_data()
    model = load_model()

except FileNotFoundError as e:

    st.error(
        f"Required file not found: {e.filename}. "
        "Ensure diabetes.csv and diabetes_model.pkl "
        "are in the project folder."
    )

    st.stop()

except Exception as e:

    st.error(f"Error loading the application: {e}")

    st.stop()


# --------------------------------------------------
# 5. VALIDATE DATASET
# --------------------------------------------------

required_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

if not all(column in data.columns for column in required_columns):

    st.error("The dataset does not contain all required columns.")

    st.stop()


feature_columns = [col for col in required_columns if col != "Outcome"]

if list(model.feature_names_in_) != feature_columns:

    st.error(
        "The trained model's feature columns do not match "
        "the current dataset."
    )

    st.stop()


# --------------------------------------------------
# 6. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Diabetes Prediction",
        "Model Performance",
        "Dataset Insights"
    ]
)


# --------------------------------------------------
# 7. DIABETES PREDICTION
# --------------------------------------------------

if page == "Diabetes Prediction":

    st.header("Patient Information")

    st.write(
        "Enter the following medical measurements "
        "to generate a prediction."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                step=1
            )

            glucose = st.number_input(
                "Glucose Level",
                min_value=0.0,
                max_value=300.0,
                value=120.0
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=100.0,
                value=20.0
            )

        with col2:

            insulin = st.number_input(
                "Insulin Level",
                min_value=0.0,
                max_value=900.0,
                value=79.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.0
            )

            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30,
                step=1
            )

        st.caption(
            "A zero value for Glucose, Blood Pressure, Skin Thickness, "
            "Insulin or BMI is treated as missing and imputed by the "
            "trained pipeline."
        )

        submitted = st.form_submit_button(
            "Predict Diabetes Outcome",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=feature_columns
        )

        # Replace invalid zero measurements
        columns_to_check = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI"
        ]

        input_data[columns_to_check] = (
            input_data[columns_to_check].replace(0, np.nan)
        )

        # Predict using the saved pipeline
        prediction = model.predict(input_data)[0]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 0:

            st.success(
                "Prediction: No Diabetes Outcome (Class 0)"
            )

            st.write(
                "The model predicts class 0 for the entered measurements."
            )

        else:

            st.error(
                "Prediction: Diabetes Outcome (Class 1)"
            )

            st.write(
                "The model predicts class 1 for the entered measurements."
            )

        st.subheader("Patient Input Summary")

        st.dataframe(
            input_data.fillna("Missing (imputed by model)"),
            use_container_width=True
        )

        # Download prediction
        result_df = input_data.copy()

        result_df["Predicted Outcome"] = int(prediction)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Summary",
            data=csv,
            file_name="diabetes_prediction.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# 8. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance Dashboard")

    st.write(
        "Evaluation of the saved SVM model using the same "
        "train-test split as your original ML implementation."
    )

    # Recreate the original train-test split
    X = data.drop(columns="Outcome").copy()

    Y = data["Outcome"]

    columns_to_check = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    X[columns_to_check] = X[columns_to_check].replace(0, np.nan)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=1,
        stratify=Y
    )

    # Predictions from saved model
    train_predictions = model.predict(X_train)

    test_predictions = model.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(
        Y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        Y_test,
        test_predictions
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Accuracy",
        f"{train_accuracy * 100:.2f}%"
    )

    col2.metric(
        "Testing Accuracy",
        f"{test_accuracy * 100:.2f}%"
    )

    col3.metric(
        "Test Samples",
        len(Y_test)
    )

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        Y_test,
        test_predictions
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
        ax=ax
    )

    ax.set_xlabel("Predicted Outcome")
    ax.set_ylabel("Actual Outcome")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        Y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.caption(
        "Recall measures how many actual positive cases are identified. "
        "Precision measures the correctness of positive predictions. "
        "F1-score balances precision and recall."
    )


# --------------------------------------------------
# 9. DATASET INSIGHTS
# --------------------------------------------------

elif page == "Dataset Insights":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(data)
    )

    col2.metric(
        "Total Features",
        len(feature_columns)
    )

    col3.metric(
        "Target Variable",
        "Outcome"
    )

    st.divider()

    # Dataset preview
    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    # Descriptive statistics
    st.subheader("Descriptive Statistics")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True
    )

    st.divider()

    # Outcome distribution
    st.subheader("Diabetes Outcome Distribution")

    outcome_counts = data["Outcome"].value_counts().sort_index()

    col1, col2 = st.columns(2)

    col1.metric(
        "No Diabetes (0)",
        int(outcome_counts.get(0, 0))
    )

    col2.metric(
        "Diabetes (1)",
        int(outcome_counts.get(1, 0))
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        data=data,
        x="Outcome",
        hue="Outcome",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Diabetes Outcome Distribution")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Between Features")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Glucose boxplot
    st.subheader("Glucose Distribution by Outcome")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.boxplot(
        data=data,
        x="Outcome",
        y="Glucose",
        hue="Outcome",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Glucose Levels Across Outcome Groups")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Glucose")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Feature distributions
    st.subheader("Feature Distributions")

    selected_features = st.multiselect(
        "Select features to visualize",
        options=feature_columns,
        default=["Glucose", "BMI", "Age"]
    )

    for feature in selected_features:

        fig, ax = plt.subplots(figsize=(8, 3))

        sns.histplot(
            data=data,
            x=feature,
            kde=True,
            ax=ax
        )

        ax.set_title(f"Distribution of {feature}")

        st.pyplot(fig)

        plt.close(fig)


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Diabetes Prediction System | Python | Scikit-learn | Streamlit"
)
