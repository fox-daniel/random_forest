import numpy as np

np.random.seed(seed=11)

import decision_tree


def print_forest(forest):
    for tree in forest:
        for node in tree:
            print(node)


def forest_predict(forest, x, threshold=0.5):
    """
    Predict the class from the random forest.
    Input:
    forest - a list of trees (see decision_tree.py doc for data structure of a tree)
    x - numpy array of features
    threshold - the threshold for predicting 1 or 0; default is 0.5

    Output:
    class_predictions - 1D numpy array of integers (1 or 0)
    """
    predictions = np.zeros(x.shape[0])
    for tree in forest:
        preds = decision_tree.predict(tree, x)
        predictions += preds
    predictions = predictions / len(forest)

    class_predictions = (predictions >= threshold).astype(int)
    return class_predictions


def grow_random_forest(
    X,
    y,
    num_trees=10,
    max_features="sqrt",
    max_depth=5,
    min_node_size=5,
    max_samples=None,
    bootstrap=True,
    min_loss=0.056,
    print_progress=False,
):
    """
    Create Random Forest Classifier
    X, y - features, targets (np.arrays)
    num_trees - number of trees to grow, default = 10
      max_features - number of features randomly selected when splitting
      each node; default is 'sqrt' which uses the floor of the sqrt of
      the number of features
    max_depth - maximum allowed depth of any branch in the tree; default = 5,
    min_node_size - minimum allowed node size; default = 5,
    max_samples - the sample size to use - default = None
      results in using full data set,
    bootstrap - using bootstrapping results in using sampling with replacements; default = True
    min_loss - if in growing the tree min_loss is exceeded or reached,
      the growing process stops; default = .056 (1% accuracy for cross-entropy)
    print_progress - print statement indicating that a tree was grown; default = False
    """
    if max_features == "sqrt":
        max_features = int(np.floor(np.sqrt(X.shape[1])))
    if max_samples is None:
        max_samples = X.shape[0]
    forest = []
    for i in range(num_trees):
        sample_indices = np.random.choice(
            np.arange(X.shape[0]), size=max_samples, replace=bootstrap
        )
        sample_features = X[sample_indices]
        sample_targets = y[sample_indices]
        tree = decision_tree.grow_decision_tree(
            sample_features,
            sample_targets,
            max_depth,
            min_node_size,
            min_loss,
            max_features,
        )
        forest.append(tree)
        if print_progress == True:
            print(f"{i+1} trees have been grown.")
    return forest
