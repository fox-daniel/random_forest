import sys
sys.path.append('/Users/Daniel/Code/decision_tree')
sys.path.append('/Users/Daniel/Code/random_forest')
import numpy as np
np.random.seed(seed = 11)
import importlib
import pytest

import decision_tree
importlib.reload(decision_tree)




def print_forest(forest):
    for tree in forest:
        for node in tree:
            print(node)

def forest_predict(forest, x, threshold = .5):
    """Predict the class from the random forest."""
    predictions = np.zeros(x.shape[0])
    for tree in forest:
        preds = decision_tree.predict(tree, x)
        predictions += preds
    predictions = predictions/len(forest)

    class_predictions = (predictions >= threshold).astype(int)
    return class_predictions

def grow_random_forest(X, y, num_trees = 10, 
                       max_features = 'sqrt', 
                       max_depth = 5, 
                       min_node_size = 5,  
                       max_samples = None,
                       bootstrap = True,
                       min_loss = .056,
                       print_progress = False 
                       ):
    """Create Random Forest Classifier"""
    # set default for max features
    if max_features == 'sqrt':
        max_features = int(np.floor(np.sqrt(X.shape[1])))
    if max_samples is None:
        max_samples = X.shape[0]
    forest = []
    for i in range(num_trees):
        sample_indices = np.random.choice(np.arange(X.shape[0]), size = max_samples, replace = bootstrap)
        sample_features = X[sample_indices]
        sample_targets = y[sample_indices]
        tree = decision_tree.grow_decision_tree(sample_features, sample_targets, max_depth, min_node_size, min_loss, max_features)
        forest.append(tree)
        if print_progress == True:
            print(f'{i+1} trees have been grown.')
    return forest





