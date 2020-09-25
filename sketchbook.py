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
import evaluation
importlib.reload(evaluation)

for i in range(5):
	data = data_for_tests.make_diagonal_ndim(100, 5)
	# print(data)
	print(data.iloc[:,-1].sum())
# plot_data.plot_data(data)