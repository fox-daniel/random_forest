# random_forest

random_forest is a python module for training and predicting with a random forest on data with float features and binary labels (0 or 1).

## Features

* Train a random forest
* Predict with a random forest
* Print the trees of the forest

## Data Structure

* A random forest is a list of decision trees
* Trees are defined by lists of nodes. These are all terminal nodes. Internal nodes are not saved.
* Each node is a tuple, (path, loss, predict, num):
    * path: a list that defines the region of the node, using the data structure: 
        * [(feature_column_ind, value, orientation),...] where
            * feature_column_ind: the index of a column of the array or dataframe
            * value: the value at which the split of the region is made
            * orientation: 'upper' (>) or 'lower' (<=) 
    * loss: the value of the loss function for the region defined by the path
    * predict: the class prediction for the region defined by the path
    * num: number of training points in the node 
    

## Installation

Save the following modules into your working directory:
* decision_tree.py
* random_forest.py
* evaluation.py
* data_for_tests.py

## Usage

Run the following code in a python script:
```
import numpy as np
import decision_tree
import random_forest
import evaluation
import data_for_tests

num_points, dim, max_features = 1000, 5, 2

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
    f"{num_points} are randomly generated in the unit cube in {dim}-dimensions.\n \
Those with the sum of coordinates >= {dim}/2 are labeled 1, \n those below are \
labeled 0."
)
print("The model achieves the following in sample metrics:")
print("precision:", metrics[0])
print("sensitivity:", metrics[1])
print("false-positive-rate:", metrics[2])
```

The function grow_random_forest() accepts the following arguments (default values listed):

* X -- numpy array of features (floats)
* y -- numpy array of labels (ints: 0 or 1)
* num_trees=10,
* max_features="sqrt",
* max_depth=5,
* min_node_size=5,
* max_samples=None,
* bootstrap=True,
* min_loss=0.056,
* print_progress=False,


## Contributing

## License
