import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics

# Încarcarea datelor de antrenare
df_train = pd.read_csv('data.csv')
df1_train = [col for col in df_train.columns if "call" not in col]
df_train = df_train[df1_train]

# Aplicarea PCA pentru datele de antrenare
df_train = df_train.T
df2_train = df_train.drop(['Gene Description', 'Gene Accession Number'], axis=0)
df2_train.index = pd.to_numeric(df2_train.index)
df2_train.sort_index(inplace=True)
df2_train['cat'] = list(pd.read_csv('actual.csv')[:38]['cancer'])
dic = {'ALL': 0, 'AML': 1}
df2_train.replace(dic, inplace=True)

X_std_train = StandardScaler().fit_transform(df2_train.drop('cat', axis=1))
sklearn_pca_train = sklearnPCA(n_components=3)
Y_sklearn_train = sklearn_pca_train.fit_transform(X_std_train)

# Salvarea rezultatelor PCA în variabila X_reduced_train
X_reduced_train = Y_sklearn_train

# Evaluarea impactului PCA asupra varianței pentru datele de antrenare
cum_sum_train = sklearn_pca_train.explained_variance_ratio_.cumsum() * 100

# Vizualizarea impactului PCA asupra varianței pentru datele de antrenare
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(range(1, len(cum_sum_train) + 1), cum_sum_train, marker='o', linestyle='-', color='b')
ax.set_title("Cumulative Sum of Explained Variance (Training Data)")
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Sum of Explained Variance (%)")
plt.show()

# Reducerea succesivă a numărului de atribute și evaluarea performanței pentru datele de antrenare
for n_components in range(1, 31):
    # Antrenarea modelului KNN pentru datele de antrenare
    clf_knn = KNeighborsClassifier(n_neighbors=10)
    clf_knn.fit(sklearn_pca_train.fit_transform(X_std_train)[:, :n_components], df2_train['cat'])

    # Evaluarea performanței pe datele de antrenare pentru KNN
    pred_knn_train = clf_knn.predict(sklearn_pca_train.transform(X_std_train)[:, :n_components])
    true_knn_train = df2_train['cat']

    # Afisarea matricei de confuzie pentru KNN
    print(f"Number of Components (KNN - Training): {n_components}")
    print(metrics.confusion_matrix(true_knn_train, pred_knn_train))
    print()

    # Antrenarea modelului Decision Tree pentru datele de antrenare
    clf_tree = DecisionTreeClassifier(min_samples_split=2)
    clf_tree.fit(sklearn_pca_train.fit_transform(X_std_train)[:, :n_components], df2_train['cat'])

    # Evaluarea performanței pe datele de antrenare pentru Decision Tree
    pred_tree_train = clf_tree.predict(sklearn_pca_train.transform(X_std_train)[:, :n_components])

    # Afisarea matricei de confuzie pentru Decision Tree
    print(f"Number of Components (Decision Tree - Training): {n_components}")
    print(metrics.confusion_matrix(true_knn_train, pred_tree_train))
    print()

    # Antrenarea modelului SVM pentru datele de antrenare
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(sklearn_pca_train.fit_transform(X_std_train)[:, :n_components], df2_train['cat'])

    # Evaluarea performanței pe datele de antrenare pentru SVM
    pred_svm_train = clf_svm.predict(sklearn_pca_train.transform(X_std_train)[:, :n_components])

    # Afisarea matricei de confuzie pentru SVM
    print(f"Number of Components (SVM - Training): {n_components}")
    print(metrics.confusion_matrix(true_knn_train, pred_svm_train))
    print()

# Restul codului pentru vizualizarea rezultatelor
fig = plt.figure(1, figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_reduced_train[:, 0], X_reduced_train[:, 1], X_reduced_train[:, 2], c=df2_train['cat'], cmap=plt.cm.Paired, s=60)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")
plt.show()
