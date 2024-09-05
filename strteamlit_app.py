pip show scikit-learn

import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd

import joblib

model = joblib.load('K-Nearest Neighborsmodel.pkl')

with open('accuracy.txt', 'r') as file:
    accuracy = file.read()

st.title("Model accuracy and Real-Time Prediction")
st.write(f"model {accuracy}")

#user input for real time prediction
st.header("Reral-Time prediction")


#Load the test data

test_data = pd.read_csv("mobile_price_range_data.csv")

#Assuming the last column is the target

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

#Assume the model expects the same input features as X_test

input_data = []
for col in X_test.columns:
    input_value = st.number_input(f"Input for {col}", value = 0.0)
    input_data.append(input_value)

#convert to DataFrame for prediction
input_df = pd.DataFrame([input_data], columns = X_test.columns)

#Make predictions

if st.button("predict"):
   prediction = model.predict(input_df)
   st.write(f"prediction: {prediction[0]}")

#plot accuracy

st.header("Accuracy plot")
st.bar_chart([float(accuracy.split(': ')[1])])

