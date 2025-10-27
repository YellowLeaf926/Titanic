import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_path_train = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
data_path_train = os.path.normpath(data_path_train)

data_path_test = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")
data_path_test = os.path.normpath(data_path_test)

train = pd.read_csv(data_path_train)
print("Load Training Data")
test = pd.read_csv(data_path_test)
print("Load Testing Data")

# Q14. Explore, add, and adjust the data as you see fit. 
train['Age'] = train.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.median()))
print("Train Dataset Age Column (impute with median)", train['Age'])

# Q15. Build a logistic regression model to predict survivability on the training set using any features that you see fit.
y_train = train['Survived']
x_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Define categorical and numerical columns
categorical_features = ['Sex']
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
print(clf.fit(x_train, y_train))

# Q16. Measure the accuracy of your model on the training set.
y_pred_train = clf.predict(x_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Train Dataset Accuracy: ", train_accuracy)

# Q17. Load `test.csv` and predict your model on the test set.
gender_submission = pd.read_csv(r"C:\Users\lpy20\Desktop\NU_MLDS\2025Fall\MLDS400-DataEngineering\HW3\titanic\gender_submission.csv")
test = pd.read_csv(r"C:\Users\lpy20\Desktop\NU_MLDS\2025Fall\MLDS400-DataEngineering\HW3\titanic\test.csv")
test = test.merge(gender_submission, on = "PassengerId", how = "left")
print("Combined Test Dataset with Survival Data: ", test.head())

test['Fare'] = test['Fare'].transform(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.median()))
print("Test Dataset Fare Column (impute with median)", test['Fare'])
print("Test Dataset Age Column (impute with median)", test['Age'])

# Q18. Measure the accuracy of your model on the test set.
x_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y_test = test['Survived']
y_pred_test = clf.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy: ", test_accuracy)