import sys
sys.path.append('/Users/Daniel/Code/decision_tree')
sys.path.append('/Users/Daniel/Code/random_forest')
import numpy as np
np.random.seed(seed = 11)
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

# num_points = 1000
# max_samples = 1000
# xy = data_for_tests.make_diagonal(num_points).values
# X = xy[:,:-1]
# y = xy[:,-1]
	
# forest = random_forest.grow_random_forest(X, y, 
# 						num_trees = 10, 
#                         max_features = 1, 
#                         max_depth = 5, 
#                         min_node_size = 5,  
#                         max_samples = max_samples,
#                         bootstrap = False,
#                         min_loss = 0,
#                         print_progress = True )

# # random_forest.print_forest(forest)

# predictions = random_forest.forest_predict(forest, X)

# mse = np.mean((predictions - y)**2)
# print(f'mean squared error: {mse}')

@pytest.mark.parametrize('num_points, classes, xbound, ybound, zbound, max_depth, min_node_size, min_loss, expected',
	[
	(np.full(8,100), [0,0,0,0,0,0,0,1], .5, .5, .5, 30, 1, 0, [1, 1, 0]),
	(np.full(8,100), [1,0,0,0,0,0,0,1], .5, .5, .5, 30, 1, 0, [1, 1, 0])
	])
def test_octavo(num_points, classes, xbound, ybound, zbound, max_depth, min_node_size, min_loss, expected):
	xy_parent = data_for_tests.make_octavo(num_points, classes, xbound, ybound, zbound).values
	X = xy_parent[:,:-1]
	y = xy_parent[:,-1]

	forest = random_forest.grow_random_forest(X, y, max_depth, min_node_size, min_loss)
	predictions = forest_predict(forest, X)
	targets = y
	tfpns = tfpn(predictions, targets)
	cm = make_confusion_matrix(*tfpns, percentage = True)
	result = [precision(cm), recall(cm), fpr(cm)]
	assert expected == result