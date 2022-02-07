"""
Used K-means for clustering bank data into groups like transactors, revolvers, new customers and VIPs for marketing
purpose, used elbow methods to decide number of clusters K. Variables including balance, purchases frequency,
purchase average, tenure, credit limit, cash advance and payments
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

# Get an overview of data and clean up data
creditcard_df = pd.read_csv('Marketing_data.csv')
creditcard_df
creditcard_df.info()
print('The average, minimum and maximum balance amount are:', creditcard_df['BALANCE'].mean(),
      creditcard_df['BALANCE'].min(), creditcard_df['BALANCE'].max())
creditcard_df.describe()
creditcard_df['CASH_ADVANCE'].max()
creditcard_df.isnull().sum()
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] \
    = creditcard_df['MINIMUM_PAYMENTS'].mean()
creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] \
    = creditcard_df['CREDIT_LIMIT'].mean()
creditcard_df.duplicated().sum()
creditcard_df.drop("CUST_ID", axis = 1, inplace = True)
n = len(creditcard_df.columns)
n
creditcard_df.columns
# Standardize data
creditcard_df_scaled = StandardScaler().fit_transform(creditcard_df)
creditcard_df_scaled.shape
creditcard_df_scaled

# use elbow method to choose the number of clusters k
scores_1 = []
range_values = range(1, 20)
for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(creditcard_df_scaled)
  scores_1.append(kmeans.inertia_)
plt.plot(scores_1, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores')
plt.show()

# apply k-means method and inverse data
kmeans = KMeans(7)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_
kmeans.cluster_centers_.shape
cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard_df.columns])
cluster_centers
cluster_centers = StandardScaler().inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])
cluster_centers
labels.shape
labels.max()
labels.min()
# concat table and label
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster.head()
# Plot the histogram of various clusters
for i in creditcard_df.columns:
  plt.figure(figsize = (35, 5))
  for j in range(7):
    plt.subplot(1,7,j+1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 10)
    plt.title('{}    \nCluster {} '.format(i,j))
  plt.show()
