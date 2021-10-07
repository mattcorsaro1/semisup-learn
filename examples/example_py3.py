import copy
import numpy as np
import random
from frameworks.CPLELearning import CPLELearningModel
import sklearn.svm

n = 1000
labeled_N = 50
d = 7
X = np.random.random((n, d))
ytrue = np.array([random.choice([0, 1]) for i in range(n)])
ys = copy.deepcopy(ytrue)
labeled_indices = random.sample(range(n), labeled_N)
unlabeled_indices = [i for i in range(n) if i not in labeled_indices]

# label a few points 
for index in unlabeled_indices:
    ys[index] = -1

# semi-supervised score, RBF SVM model
ssmodel = CPLELearningModel(sklearn.svm.SVC(kernel="rbf", probability=True), predict_from_probabilities=True) # RBF SVM
ssmodel.fit(X, ys)
print("CPLE semi-supervised RBF SVM score", ssmodel.score(X, ytrue))
