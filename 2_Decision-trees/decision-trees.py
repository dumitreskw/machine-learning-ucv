from pandas import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from utils import compute_accuracy_and_error_rate 
from matplotlib import pyplot as plt
from sklearn import tree
import joblib

data = pd.read_csv('dataset.csv')

# delete JobRole column
data_copy = data
deleted_column = ['JobRole']
data_copy = data_copy.drop(deleted_column, axis = 1)
columns = data_copy.columns

print(data_copy.info())

# encoding for categorical var
data_copy = pd.get_dummies(data_copy)
data['JobRole'].unique()
job_roles_number = data['JobRole'].value_counts()

print('# Job roles and their numbers: \n')
print(job_roles_number)

job_roles_before_encoding = data['JobRole'].unique()

# encode roles 
label_encoder = LabelEncoder()
data['JobRoleCategory'] = label_encoder.fit_transform(data['JobRole'])
job_roles_after_encoding = data['JobRoleCategory'].unique()
print('\n--- Job roles after label encoding: \n', job_roles_after_encoding)

features = data_copy.columns
X = data_copy[features]
y = data['JobRoleCategory']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.07, random_state=1)

print('Initial size of the dataset:', len(X))
print('The size of the dataset used for training the model: ',len(train_X))
print('The size of the dataset used for validating the model: ',len(test_X))

print('\n\n*** Decision tree regressor ***')
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(train_X, train_y)

regressor_predictions = regressor.predict(test_X)
      
accuracy_dec_tree, error_rate_dec_tree = compute_accuracy_and_error_rate(test_y, regressor_predictions, len(regressor_predictions))

print('\n--- Job roles before label encoding: \n', job_roles_before_encoding)
print('\n--- Job roles after label encoding: \n', job_roles_after_encoding)
print('\n --- Predictions: \n', regressor_predictions)
print('\n --- Actual: \n', test_y)
print('\n--- Accuracy: ', accuracy_dec_tree)
print('\n--- Error rate: ', error_rate_dec_tree)


fig_dec_tree = plt.figure(figsize=(25,20))
_ = tree.plot_tree(regressor, 
                   feature_names=features,  
                   class_names=data['JobRoleCategory'],
                   filled=True)


# fig_dec_tree.savefig("decision_tree.png")
# joblib.dump(regressor, 'decision_tree_model.pkl')