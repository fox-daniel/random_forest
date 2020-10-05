import sys

sys.path.append("/Users/Daniel/Code/decision_tree")
sys.path.append("/Users/Daniel/Code/random_forest")
import numpy as np

np.random.seed(seed=11)
import importlib
import pytest

import data_for_tests

importlib.reload(data_for_tests)
import plot_data

importlib.reload(plot_data)
import decision_tree

importlib.reload(decision_tree)
import random_forest

importlib.reload(random_forest)
import evaluation

importlib.reload(evaluation)


@pytest.mark.parametrize(
    "num_points, classes, xbound, ybound, zbound, max_depth, min_node_size, min_loss, expected",
    [
        (
            np.full(8, 1000),
            [0, 0, 0, 0, 0, 0, 0, 1],
            0.5,
            0.5,
            0.5,
            30,
            1,
            0,
            [1, 1, 0],
        ),
        (
            np.full(8, 1000),
            [1, 0, 0, 0, 0, 0, 0, 1],
            0.5,
            0.5,
            0.5,
            30,
            1,
            0,
            [1, 1, 0],
        ),
    ],
)
def test_octavo(
    num_points,
    classes,
    xbound,
    ybound,
    zbound,
    max_depth,
    min_node_size,
    min_loss,
    expected,
):
    xy_parent = data_for_tests.make_octavo(
        num_points, classes, xbound, ybound, zbound
    ).values
    X = xy_parent[:, :-1]
    y = xy_parent[:, -1]

    forest = random_forest.grow_random_forest(
        X, y, num_trees=20, max_features=2, min_node_size=1
    )
    predictions = random_forest.forest_predict(forest, X)
    targets = y
    tfpns = evaluation.tfpn(predictions, targets)
    cm = evaluation.make_confusion_matrix(*tfpns, percentage=True)
    result = np.array(
        [evaluation.precision(cm), evaluation.sensitivity(cm), evaluation.fpr(cm)]
    )
    expected = np.array(expected)
    assert np.any(np.abs(expected - result) < 0.01)


@pytest.mark.parametrize(
    "num_points, dim, max_features, expected, precision_bound",
    [
        (1000, 2, 2, [1, 1, 0], 0.01),
        (1000, 3, 2, [1, 1, 0], 0.01),
        (1000, 4, 2, [1, 1, 0], 0.01)
        # (10000, 10, 'sqrt', [1, 1, 0], .05), # commented out because they take a long time to run
        # (10000, 20, 'sqrt', [1, 1, 0], .05) # commented out because they take a long time to run
    ],
)
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
    print(precision_bound)
    assert np.any(np.abs(expected - result) < precision_bound)
