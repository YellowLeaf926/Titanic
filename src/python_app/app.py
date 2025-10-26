import pandas as pd
import numpy as np
import sklearn
import os

data_path_train = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
data_path_train = os.path.normpath(data_path_train)

data_path_test = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")
data_path_test = os.path.normpath(data_path_test)

train = pd.read_csv(data_path_train)
print("Load Training Data")
test = pd.read_csv(data_path_test)
print("Load Testing Data")