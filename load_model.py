import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('K-Nearest Neighborsmodel.pkl')

#load dataset

test_data = pd.read_csv("mobile_price_range_data.csv")
#Assuming the last column is the target

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

#generate test predictions

y_pred = model.predict(X_test)

#calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

#save accuracy to a file

with open('accuracy.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}')
    