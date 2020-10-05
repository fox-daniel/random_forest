import numpy as np
import data_for_tests
import decision_tree
import random_forest
import evaluation

num_points, dim, max_features, expected, precision_bound = 1000, 5, 2, [1, 1, 0], 0.01


def test_diagonal_ndim(num_points, dim, max_features, expected, precision_bound):
    xy_parent = data_for_tests.make_diagonal_ndim(num_points, dim).values
    X = xy_parent[:, :-1]
    y = xy_parent[:, -1]

    forest = random_forest.grow_random_forest(
        X, y, num_trees=30, max_depth=20, max_features=max_features, min_node_size=1
    )
    predictions = random_forest.forest_predict(forest, X)
    targets = y
    tfpns = evaluation.tfpn(predictions, targets)
    cm = evaluation.make_confusion_matrix(*tfpns, percentage=True)
    result = np.array(
        [evaluation.precision(cm), evaluation.sensitivity(cm), evaluation.fpr(cm)]
    )
    expected = np.array(expected)
    print(
        f"{num_points} are randomly generated in the unit cube in {dim}-dimensions.\n \
    Those with the sum of coordinates >= {dim}/2 are labeled 1, those below are\n \
    labeled 0."
    )
    print("The model achieves the following results:")
    print("precision:", result[0])
    print("sensitivity:", result[1])
    print("false-positive-rate:", result[2])


test_diagonal_ndim(num_points, dim, max_features, expected, precision_bound)
