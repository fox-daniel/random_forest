import numpy as np

X = np.array([[.1,.2,0],[.2,.7,1],[.3,.8,0]])
sample_indices = np.random.choice(np.arange(X.shape[0]), 2, replace = False)
print(sample_indices)
print(X[sample_indices])