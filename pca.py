import mnist_parse
from sklearn.decomposition import PCA
import numpy as np

X_train,y_train = mnist_parse.parse("train")

X_train = X_train/np.average(X_train)

pca = PCA()

pca.fit(X_train)

x = pca.explained_variance_/np.sum(pca.explained_variance_)

tot = 0.0

for i in range(len(x)):
    tot = tot + x[i]
    if tot > 0.95: 
        print("%d components account for 95 pct of variation"%(i))
        break

