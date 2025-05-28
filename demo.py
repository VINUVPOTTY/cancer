import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("Cancer Prediction App")
st.write("This app predicts whether a tumor is **Benign (0)** or **Malignant (1)** using KNN.")

# Load dataset
df = pd.read_csv("cancer.csv")
df.drop(['Name', 'Surname'], axis=1, inplace=True)

# Features and labels
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Scale features
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)

# Accuracy
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

# Input fields for user
st.header("Input Tumor Features")
feature1 = st.number_input("Feature 1", min_value=0.0, max_value=100.0, value=52.0)
feature2 = st.number_input("Feature 2", min_value=0.0, max_value=100.0, value=18.0)
feature3 = st.number_input("Feature 3", min_value=0.0, max_value=100.0, value=4.0)
feature4 = st.number_input("Feature 4", min_value=0.0, max_value=100.0, value=5.0)

# Prediction and result display
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    input_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_scaled)[0]
    result = "🧪 Malignant (1) – Needs Attention" if prediction == 1 else "✅ Benign (0) – No Concern"
    st.subheader("Prediction Result")
    st.success(result)
