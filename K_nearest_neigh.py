"""
Using K Nearest neighbors to classify telecommunication company customers into 4 groups: basic service, E-service
plus service and full service with the features including region, age, income, gender and marital
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Get an overview of data and clean up data
df = pd.read_csv('teleCust1000t.csv')
df.head()
df['custcat'].value_counts()
df.columns
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'employ', 'gender', 'reside']].values
X[0:5]
y = df['custcat'].values
y[0:5]
# data standardized
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]
# use 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# train model and find the best K number
K = 10
mean_acc = np.empty(K - 1)
std_acc = np.empty(K - 1)
for n in range(1, K):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, y_hat)
    std_acc[n - 1] = np.std(y_hat == y_test) / np.sqrt(y_hat.shape[0])

# plot different K numbers' the accuracy score
plt.plot(range(1,K),mean_acc)
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.show()
# summarized the finding
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
