#! /usr/bin/python3.7
# General imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ML Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
 
# Change for mean negative number in DB
def subsNegative(data):
    searchNegative = np.where(data == -1)
    for i in range(len(searchNegative[0])):
        mean = data[searchNegative[0][i]-1] + data[searchNegative[0][i]-2]
        mean = mean / 2
        data[searchNegative[0][i]] = mean
    return data

def plot_confusion_matrix(cm, 
                          labels,
                          archive_name,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    import itertools
    plt.cla()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Real Class')
    plt.xlabel('Prediction Class')
    plt.tight_layout()
    plt.show()
    #plt.savefig('./plots/'+archive_name+'.pdf',format='pdf')
    plt.clf()

# DB label
names = ['cpu','memory','memory_swap_in','memory_swap_out','disk_request_read','disk_request_write','interface_bytes_in','interface_bytes_out','interface_packets_in','interface_packets_out','timestamp','label','label2']
identify_dos = ['cpu','memory','memory_swap_out','disk_request_read','disk_request_write','interface_bytes_in','interface_bytes_out','interface_packets_in','interface_packets_out']


# Read DB
base_metrics_training = pd.read_csv('./training.txt', header=None, names=names)
base_metrics_test = pd.read_csv('./test.txt', header=None, names=names)

X = np.array(base_metrics_training.iloc[1:, [0,1,3,4,5,6,7,8,9]])
X2 = np.array(base_metrics_training.iloc[1:, [0,1,3,4,5,6,7,8,9]])

y = np.array(base_metrics_training.iloc[1:,11])
y2 = np.array(base_metrics_training.iloc[1:,12])

X = X.astype(np.float)
X2 = X2.astype(np.float)

X = subsNegative(X)
X2 = subsNegative(X2)

X_test = np.array(base_metrics_test.iloc[1:, [0,1,3,4,5,6,7,8,9]])
y_test = np.array(base_metrics_test.iloc[1:,11])

X_test = X_test.astype(np.float)
X_test = subsNegative(X_test)

#ML Algorithms
print('--------------\nkNN\n')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
pred_knn_teste = knn.predict(X_test)

print (accuracy_score(y_test, pred_knn_teste))
print(confusion_matrix(y_test, pred_knn_teste))  
print(classification_report(y_test, pred_knn_teste,digits=4))

cm = confusion_matrix(y_test, pred_knn_teste)
sns.set_style('white')
plot_confusion_matrix(cm, ['Only Client', 'DDoS Attack'], title="Confusion Matrix - Only Client or DDoS Attack - kNN",archive_name='kNN-matrix-1')
sns.set()

# Plot ROC curve
metrics.plot_roc_curve(knn,X_test,y_test)
plt.show()
#plt.savefig('./knn-roc-curve.pdf',format='pdf')

print('Identify phase - kNN')
searchDoS = np.where(pred_knn_teste == '1')
searchDoS = searchDoS[0].tolist()

X_dos = np.array(base_metrics_test.loc[searchDoS,identify_dos])
X_dos = X_dos[1:len(X_dos)].astype(np.float)
X_dos = subsNegative(X_dos)
y_dos = np.array(base_metrics_test.loc[searchDoS,'label2'])
y_dos = y_dos[1:len(y_dos)]

knn.fit(X2,y2)
pred_knn_dos_teste = knn.predict(X_dos)

print (accuracy_score(y_dos, pred_knn_dos_teste))
print("\n")
print(confusion_matrix(y_dos, pred_knn_dos_teste))
print("\n")
print(classification_report(y_dos, pred_knn_dos_teste,digits=4))

cm = confusion_matrix(y_dos, pred_knn_dos_teste)
sns.set_style('white')
plot_confusion_matrix(cm, ['Only Client', 'SYN Flood', 'HTTP Flood'], title="Confusion Matrix - Client, SYN or HTTP Flood - kNN",archive_name='kNN-matrix-2')
sns.set()

print('--------------\nRandom Forest\n')
classifier_RF = RandomForestClassifier(n_estimators = 100)
classifier_RF.fit(X,y)
pred_RF_teste = classifier_RF.predict(X_test)

print (accuracy_score(y_test, pred_RF_teste))
print("\n")
print(confusion_matrix(y_test, pred_RF_teste))
print("\n")
print(classification_report(y_test, pred_RF_teste,digits=4))

cm = confusion_matrix(y_test, pred_RF_teste)
sns.set_style('white')
plot_confusion_matrix(cm, ['Only Client', 'DDoS Attack'], title="Confusion Matrix - Only Client or DDoS Attack - Random Forest",archive_name='RF-matrix-1')
sns.set()

# Plot ROC curve
metrics.plot_roc_curve(classifier_RF,X_test,y_test)
plt.show()
#plt.savefig('./rf-roc-curve.pdf',format='pdf')

print('Identify phase - RF')
searchDoS = np.where(pred_RF_teste == '1')
searchDoS = searchDoS[0].tolist()

X_dos = np.array(base_metrics_test.loc[searchDoS,identify_dos])
X_dos = X_dos[1:len(X_dos)].astype(np.float)
X_dos = subsNegative(X_dos)
y_dos = np.array(base_metrics_test.loc[searchDoS,'label2'])
y_dos = y_dos[1:len(y_dos)]

classifier_RF.fit(X2,y2)
pred_RF_dos_teste = classifier_RF.predict(X_dos)

print (accuracy_score(y_dos, pred_RF_dos_teste))
print("\n")
print(confusion_matrix(y_dos, pred_RF_dos_teste))
print("\n")
print(classification_report(y_dos, pred_RF_dos_teste,digits=4))

cm = confusion_matrix(y_dos, pred_RF_dos_teste)
sns.set_style('white')
plot_confusion_matrix(cm, ['Only Client', 'SYN Flood', 'HTTP Flood'], title="Confusion Matrix - Client, SYN or HTTP Flood - Random Forest",archive_name='RF-matrix-2')
sns.set()
