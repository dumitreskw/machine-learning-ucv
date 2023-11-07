import joblib
from pandas import pandas as pd
from sklearn.preprocessing import LabelEncoder


new_data = pd.read_csv('instance.csv')
model = joblib.load('decision_tree_model.pkl')

prediction = model.predict(new_data)

print(prediction)

