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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Read the dataset on the root process
if rank == 0:
    data = pd.read_csv(r"C:/Users/lavgu/OneDrive/Desktop/dbmsreview/Agri.csv")
    # Determine the number of data points each process will handle
    chunk_size = len(data) // size
else:
    data = None
    chunk_size = None

# Broadcast chunk size to all processes
chunk_size = comm.bcast(chunk_size, root=0)

# Scatter indices of the dataset
if rank == 0:
    indices = np.arange(len(data))
    np.random.shuffle(indices)  # Shuffle indices for better load balancing
else:
    indices = None
local_indices = np.empty(chunk_size, dtype=int)
comm.Scatter(indices, local_indices, root=0)

# Load corresponding data based on indices
local_data = data.iloc[local_indices]

# Combine data on the root process
all_data = comm.gather(local_data, root=0)

# Plotting on the root process
if rank == 0:
    combined_data = pd.concat(all_data)

    # Temperature graph for all crops
    plt.figure(figsize=(10, 6))
    sns.barplot(x='label', y='temperature', data=combined_data)
    plt.title('Average Temperature for Different Crops')
    plt.xlabel('Crops')
    plt.ylabel('Average Temperature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Soil attributes graph for all crops
    attributes = ['humidity', 'ph', 'N', 'P', 'K', 'rainfall']
    plt.figure(figsize=(15, 8))
    for i, attr in enumerate(attributes, start=1):
        plt.subplot(2, 3, i)
        sns.barplot(x='label', y=attr, data=combined_data)
        plt.title(f'Average {attr.capitalize()} for Different Crops')
        plt.xlabel('Crops')
        plt.ylabel(f'Average {attr.capitalize()}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Comparison graph of classifiers
    X = combined_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = combined_data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    classifiers = {
        'Decision Tree': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=29, criterion='entropy', random_state=0),
        'Naive Bayes': GaussianNB()
    }

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(f'{clf_name} Report:\n{report}')

    plt.figure(figsize=(10, 6))
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = np.mean(y_pred == y_test)
        plt.bar(clf_name, acc, label=clf_name)
    plt.title('Classifier Comparison')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()











