# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load Data
# --------------------------------------------
train = pd.read_csv("data/train.csv")
print("Training data loaded successfully.")

test = pd.read_csv("data/test.csv")
print("Testing data loaded successfully.")

# Data Cleaning & Feature Engineering
# --------------------------------------------

# Impute missing Age values with median age grouped by Sex
train['Age'] = train.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.median()))
print("Train dataset: Missing 'Age' values imputed using group median by Sex.")

# Define features and target variable
y_train = train['Survived']
x_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Define categorical and numerical feature sets
categorical_features = ['Sex']
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Preprocessing and Model Building
# --------------------------------------------

# Preprocessing pipeline:
# - Standardize numerical features
# - One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Combine preprocessing and model into one pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit logistic regression model
clf.fit(x_train, y_train)
print("Logistic regression model trained successfully.")

# Model Evaluation on Training Data
# --------------------------------------------

# Predict on training set
y_pred_train = clf.predict(x_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Test Data Preparation
# --------------------------------------------

# Impute missing Fare and Age values using medians
test['Fare'] = test['Fare'].transform(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.median()))
print("Test dataset: Missing 'Fare' and 'Age' values imputed using median(s).")

# Prepare test features
x_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Generate Predictions on Test Data
# --------------------------------------------

# Predict survivability on test dataset
test["Pred_Survived"] = clf.predict(x_test)
test[['PassengerId', 'Pred_Survived']].to_csv("python_app/py_predictions.csv", index=False)

print("Predictions saved to python_app/py_predictions.csv.")
print("Titanic model pipeline completed successfully.")