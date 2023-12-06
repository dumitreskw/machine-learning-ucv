import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

#Hierarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

pd.options.mode.chained_assignment = None  # default='warn'


data = pd.read_csv("data.csv")
X = data[['Annual Income (k$)', 'Age', 'Spending Score (1-100)']]


#Find optimum number of cluster
sse = [] #SUM OF SQUARED ERROR
for k in range(1,11):
    km = KMeans(n_clusters=k, n_init=10, random_state=2)
    km.fit(X)
    sse.append(km.inertia_)
    
sns.set_style("whitegrid")
g=sns.lineplot(x=range(1,11), y=sse)

g.set(xlabel ="Number of cluster (k)",
      ylabel = "Sum Squared Error",
      title ='Elbow Method')

plt.show()


kmeans = KMeans(n_clusters = 3, n_init=10, random_state = 2)
kmeans.fit(X)
kmeans.cluster_centers_
pred = kmeans.fit_predict(X)


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X['Annual Income (k$)'],X['Age'],c = pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center = center[:2]
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Age")
plt.show()


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X['Annual Income (k$)'], X['Age'], X['Spending Score (1-100)'], c=pred, cmap=cm.Accent)
for center in kmeans.cluster_centers_:
    ax.scatter(center[0], center[1], center[2], marker='^', c='red', s=100)

ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Age')
ax.set_zlabel('Spending Score')
ax.set_title('3D Cluster Plot')
plt.show()


X_subset = X[['Age', 'Spending Score (1-100)']]

age_filter = (X_subset['Age'] >= 0) & (X_subset['Age'] <= 25)
X_subset_filtered = X_subset[age_filter]

spend_filter = (X_subset['Spending Score (1-100)'] >= 50) & (X_subset['Spending Score (1-100)'] <= 55)
X_subset_filtered_spend_age = X_subset_filtered[spend_filter]

clustering = AgglomerativeClustering(n_clusters=2).fit(X_subset_filtered_spend_age)

Z = linkage(X_subset_filtered_spend_age, 'ward')

# Plot dendrogram
dendrogram(Z)

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Age')
plt.ylabel('Gender')
plt.show()