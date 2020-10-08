import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
import importlib

np.random.seed(seed=0)

import decision_tree

# importlib.reload(decision_tree)

import random_forest

# importlib.reload(random_forest)

from data_for_tests import *
from evaluation import *

import time
from datetime import datetime

report = open("random_forest_test_report", "a")
report.write("***** New Run of Test *****\n")

report.write(f"Time of run: {datetime.now()}\n")

note = "Notes: numba: njit on loss and info_gain & best_cut \
	 \n"

if note:
    report.write(note)

# select type of test data
testing_data = make_diagonal_ndim
report.write(f"data type: {testing_data.__name__}\n")

# select number of data points
if testing_data == make_diagonal_ndim:
    num_points = 10000
    dim = 8
    report.write(f"{num_points} points in {dim} dimensions.\n")

data = testing_data(num_points, dim)

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

# plot_data(data)

# model parameters
max_depth = 20
min_node_size = 1
report.write(f"max_depth = {max_depth}\n")
report.write(f"min_node_size = {min_node_size}\n")

# train model - don't report!
tree = random_forest.grow_random_forest(
    X, y, 
    num_trees=10,
    max_features="sqrt",
    max_depth=max_depth,
    min_node_size=min_node_size,
    max_samples=None,
    bootstrap=True,
    min_loss=0.056,
    print_progress=False,
)

# train model - now that numba has compiled - report
start = time.time()
tree = random_forest.grow_random_forest(
    X, y, 
    num_trees=10,
    max_features="sqrt",
    max_depth=max_depth,
    min_node_size=min_node_size,
    max_samples=None,
    bootstrap=True,
    min_loss=0.056,
    print_progress=False,
)
finish = time.time()
duration = finish - start
report.write(f"Growing the Random Forest took: {duration}\n")

# report.write(f'Tree:\n')
# for node in tree:
# 	report.write('\n \n New Node: \n')
# 	for cond in node[0]:
# 		report.write(f'{cond}')
# 		report.write('\n')

# predict model
start = time.time()
preds = random_forest.forest_predict(tree, X)
finish = time.time()
duration = finish - start
report.write(f"Predicting with the Random Forest took: {duration}\n")

tfpns = tfpn(preds, y)
cm = make_confusion_matrix(*tfpns, percentage=True)
# plot_cm(cm)
report.write((f"Precision: {precision(cm)}, Recall: {sensitivity(cm)}\n"))

report.write("***** End ****\n\n\n")


# To profile a script and put the stats in a file:
# python -m cProfile -o file_with_stats myscript.py

# to visualize
# snakeviz file_with_stats
