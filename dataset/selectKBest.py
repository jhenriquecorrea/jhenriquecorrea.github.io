#! /usr/bin/python3.7
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Change for mean negative number in DB
def subsNegative(data):
    searchNegative = np.where(data == -1)
    for i in range(len(searchNegative[0])):
        mean = data[searchNegative[0][i]-1] + data[searchNegative[0][i]-2]
        mean = mean / 2
        data[searchNegative[0][i]] = mean
    return data


names = ['cpu','memory','memory_swap_in','memory_swap_out','disk_request_read','disk_request_write','interface_bytes_in','interface_bytes_out','interface_packets_in','interface_packets_out','timestamp','label','label2']

base_metrics = pd.read_csv('./training.txt', header=None, names=names)

X = np.array(base_metrics.iloc[1:, [0,1,2,3,4,5,6,7,8,9]])

# Detection phase
y = np.array(base_metrics.iloc[1:,11])
# Identify phase
#y = np.array(base_metrics.iloc[1:,12])

X = X.astype(np.float)

X = subsNegative(X)

selector = SelectKBest(k=3)
selector.fit(X, y)

X_new = selector.transform(X)
X_new.shape

for i in range(len(names)-1):
    selector = SelectKBest(chi2,k=i+1)
    selector.fit(X, y)
    print(selector.get_support(indices=True))
    X_actual = np.array(base_metrics.iloc[1:, selector.get_support(indices=True)])

    dataTraining, dataTest, yTraining, yTest = train_test_split(X_actual, y, test_size=0.3, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(dataTraining,yTraining)
    pred_knn_teste = knn.predict(dataTest)

    print("k = %d" %(i+1))
    print (accuracy_score(yTest, pred_knn_teste))
    print("-----------------")
