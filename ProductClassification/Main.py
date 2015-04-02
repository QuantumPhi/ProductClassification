import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
import sklearn.metrics as metrics
import sklearn.ensemble as en
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.linear_model as ln

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

frame = pd.read_csv('C:\\Users\\Tarik\\Documents\\DATA\\product_dump\\train.csv')
mat = frame.as_matrix()

ids = mat[:,0]
pca = decomp.PCA(len(mat[0]) - 2)
X = pca.fit_transform(mat[:,1:len(mat[0]) - 1].astype(float))
y = np.array([float(x[-1]) for x in mat[:,len(mat[0]) - 1]])

shuffle_in_unison_scary(X,y)

fold = 0.5
leng = int(len(X) * fold)
X_train = X[:leng]
X_test = X[leng:]

y_train = y[:leng]
y_test = y[leng:]

print "Fitting"
model = en.BaggingClassifier(n_estimators = 200)
model.fit(X_train,y_train)

print "Testing"
val = metrics.log_loss(y_test,model.predict_proba(X_test))
print val