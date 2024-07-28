import time
from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to load and preprocess the dataset
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv(r"C:/Users/lavgu/OneDrive/Desktop/dbmsreview/Agri.csv")
    # Preprocess the data (e.g., feature selection, encoding)
    # This is a placeholder for your actual preprocessing steps
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

# Function to train and evaluate models
def train_and_evaluate_models(X, y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define classifiers
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=29, criterion='entropy', random_state=0),
        'Naive Bayes': GaussianNB()
    }
    
    # Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(f'{clf_name} Report:\n{report}')

# Serial execution time measurement
if rank == 0:
    start_time = time.time()
    X, y = load_and_preprocess_data()
    train_and_evaluate_models(X, y)
    end_time = time.time()
    serial_time = end_time - start_time
    print(f"Serial execution time: {serial_time} seconds")

# Parallel execution time measurement
start_time = time.time()
X, y = load_and_preprocess_data()
train_and_evaluate_models(X, y)
end_time = time.time()
parallel_time = end_time - start_time

# Speedup calculation
if rank == 0:
    speedup = serial_time / parallel_time
    print(f"Speedup: {speedup}")

# Print execution times for both serial and parallel
if rank == 0:
    print(f"Serial execution time: {serial_time} seconds")
    print(f"Parallel execution time: {parallel_time} seconds")
