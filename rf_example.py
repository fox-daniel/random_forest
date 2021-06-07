import numpy as np
import decision_tree
import random_forest
import evaluation
import data_for_tests

num_points, dim, max_features = 1000, 10, 3

# generate data
xy_parent = data_for_tests.make_diagonal_ndim(num_points, dim).values
X = xy_parent[:, :-1]
y = xy_parent[:, -1]

# train the model -- grow the forest
forest = random_forest.grow_random_forest(
    X, y, num_trees=30, max_depth=20, max_features=max_features, min_node_size=1
)

# make predictions
predictions = random_forest.forest_predict(forest, X)

# calculate the numbers of true positives, false positives, true negatives, false negatives
tfpns = evaluation.tfpn(predictions, y)

# calculate the confusion matrix
cm = evaluation.make_confusion_matrix(*tfpns, percentage=True)

# calculate metrics: precision, sensitivity, false-positive-rate
metrics = np.array(
    [evaluation.precision(cm), evaluation.sensitivity(cm), evaluation.fpr(cm)]
)

print(
    f"{num_points} points are randomly generated in the unit cube in {dim}-dimensions.\n \
Those with the sum of coordinates >= {dim}/2 are labeled 1, \n those below are \
labeled 0."
)
print("The model achieves the following in sample metrics:")
print("precision:", metrics[0])
print("sensitivity:", metrics[1])
print("false-positive-rate:", metrics[2])
print("If the metrics are not 1,1,0, then there is a problem.")
# if (metrics[0] == 1) & (metrics[1] == 1) & (metrics[2] == 0):
# 	print(0)
# else:
# 	print(1)
