"""
This module implements training and predicting for a decision tree on data with binary targets encoded as {0,1}.

Data Structure:

Trees are defined by lists of nodes. These are all terminal nodes. Internal nodes are not saved. 
Each node is a tuple, (path, loss, predict, num): 
        loss: the value of the loss function for the region defined by the path
        predict: the class prediction for the region defined by the path
        num: number of training points in the node 
        path: a list that defines the region of the node, using the data structure:
            [(feature_column_ind, value, orientation),...] where
                feature_column_ind: the index of a column of the array or dataframe
                value: the value at which the split of the region is made
                orientation: 'upper' (>) or 'lower' (<=) 
"""

import numpy as np
from numba import njit


def predict(tree, x):
    """Predict the Outcome for the decision tree model.

    Input:
    tree: tree is the list of nodes defining the tree
    x: the input features to predict the outcome of;
        x is an np.array

    Output:
    outcome: an array of 1's and 0's predicting the class of each row of x
    """
    pred = np.zeros(x.shape[0])
    for node in tree:
        pred += node[2] * region_indicator(node[0], x)
    pred = pred.astype(int)
    return pred


def region_indicator(path, x):
    """Indicator Function for the region defined by the path.
    Input: x np.array of features
    path: a list of tuples; each tuple is
    (column_name of data or index of array, split point, specification of upper or lower half-plan )
    x: the data points; np.array
    Output:
    1 if x is in the region (for each row)
    0 if x is not in the region (for each row)
    """
    result = np.full(x.shape[0], 1)
    if path is None:
        return result
    else:
        for tup in path:
            pred = half_plane_indicator(tup[0], tup[1], tup[2], x)
            result *= pred
            result = result.astype(int)
        return result


def half_plane_indicator(j, s, orientation, x):
    def lower(j, s, x):
        # indicator function for lower half-plane
        mask = x[:, j] <= s
        mask = mask.astype(int)
        return mask

    def upper(j, s, x):
        # indicator function for lower half-plane
        mask = x[:, j] > s
        mask = mask.astype(int)
        return mask

    if orientation == "lower":
        return lower(j, s, x)
    elif orientation == "upper":
        return upper(j, s, x)


def grow_decision_tree(
    X, y, max_depth=10, min_node_size=1, min_loss=0, num_features=False
):
    """
    Grow a decision tree from numpy arrays of features and
    targets with max_depth, min_node_size, min_loss specified.

    Input:
    X - numpy array of features
    y - numpy array of targets
    max_depth - no branch of the tree is allowed to exceed this paramter
    min_node_size - no node in the tree is allowed to drop below this parameter
    min_loss - the growth of the tree stops if the loss of it drops below min_loss
    num_features - (optional) the number of features to select for each split

    Output:
    tree - a list of nodes; each node is a tuple (path, loss, predict, num);
    each path is a list of tuples (feature_column_ind, value, orientation)
    """
    min_depth = 0
    tree = initialize_tree(X, y)
    max_node_size = update_max_node_size(tree)
    current_loss = tree[0][1]
    while (
        (max_node_size >= 2 * min_node_size)
        & (min_depth < max_depth)
        & (current_loss > min_loss)
    ):
        old_loss = current_loss
        max_node_size, min_depth, current_loss = branch(
            X,
            y,
            max_depth,
            min_node_size,
            min_loss,
            tree,
            max_node_size,
            min_depth,
            current_loss,
            num_features,
        )
        if current_loss == old_loss:
            break
    return tree


def initialize_tree(X, y):
    """Creates the initial tree as a single node based on a dataframe"""
    num_0, num_1, num = count_classes(y)
    path_initial = []
    loss_initial = loss_fnc(num_0, num_1)
    pred_initial = 0 if num_0 >= num_1 else 1
    num_initial = y.shape[0]

    node_initial = [path_initial, loss_initial, pred_initial, num_initial]
    tree_initial = [node_initial]
    return tree_initial


def print_tree_path_sizes(tree):
    for node in tree:
        print(
            f"path size: {len(node[0])}, loss: {node[1]}, pred: {node[2]}, num_points: {node[3]}"
        )


def branch(
    X,
    y,
    max_depth,
    min_node_size,
    min_loss,
    tree,
    max_node_size,
    min_depth,
    current_loss,
    num_features=False,
):
    for i, node in enumerate(tree):
        path, loss_parent, predict, num = node
        if (num >= 2 * min_node_size) & (len(path) < max_depth):
            xy_parent = select_node_data(path, X, y)
            if num_features:
                features = np.random.choice(
                    np.arange(xy_parent[:, :-1].shape[1]), num_features, replace=False
                )
            else:
                features = range(xy_parent.shape[1] - 1)
            node_lower, node_upper, current_igain = best_variable(
                xy_parent, loss_parent, min_node_size, features
            )
            if current_igain is not None:
                num_lower = node_lower[3]
                num_upper = node_upper[3]
                if (current_igain > 0) & (min(num_lower, num_upper) >= min_node_size):
                    node_index = i
                    loss_lower = node_lower[1]
                    loss_upper = node_upper[1]
                    new_tree_loss = calculate_new_loss(
                        tree, node_index, loss_lower, loss_upper
                    )
                    if new_tree_loss >= min_loss:
                        update_tree(tree, node_index, node_lower, node_upper)
                        max_node_size = update_max_node_size(tree)
                        min_depth = update_min_depth(tree)
                        current_loss = new_tree_loss
    return max_node_size, min_depth, current_loss


def create_list_of_losses(tree):
    losses = []
    for node in tree:
        losses.append(node[1])
    return losses


def select_node_data(path, X, y):
    """
    Select the rows of data (X,y) that fall into the region defined by the path.
    Input:
        path: a list of conditions defining a region
        data: data whose columns are referenced in the path: np.array
    """
    xy_parent = np.concatenate([X, y[:, np.newaxis]], axis=1)
    for cond in path:
        col, value, orientation = cond
        if orientation == "upper":
            xy_parent = xy_parent[xy_parent[:, col] > value]
        else:
            xy_parent = xy_parent[xy_parent[:, col] <= value]
    return xy_parent.copy()


def calculate_training_loss_weighted(tree):
    num_points = 0
    for node in tree:
        num_points += node[3]
    loss = 0
    for node in tree:
        loss += node[3] / num_points * node[1]
    return loss


def calculate_new_loss(tree, node_index, loss_lower, loss_upper):
    loss = 0
    temp_tree = tree.copy()
    temp_tree.pop(node_index)
    for node in temp_tree:
        loss += node[1]
    loss += loss_lower
    loss += loss_upper
    return loss


def update_tree(tree, node_index, new_lower, new_upper):
    """
    This branches the tree in-place. It is called in branch().
    """
    if node_index is not None:
        old_node = tree.pop(node_index)
        new_node_lower, new_node_upper = build_nodes(old_node, new_lower, new_upper)
        tree.append(new_node_lower)
        tree.append(new_node_upper)


def build_nodes(old_node, new_lower, new_upper):
    """
    Build the two new nodes to replace the one that was split. The
    node data structure is: (path, loss, pred) with path a list of
    tuples of the form (col, cut_value, orientation)

    Input:
    old_node, new_lower, new_upper -- each is an instance of a node
    data structure

    Output:
    new_node_lower, new_node_upper -- the node data structures for the
    new children nodes that replace the old_node in the tree
    """
    new_path_lower = old_node[0].copy()
    new_path_upper = old_node[0].copy()
    new_path_lower.append(new_lower[0])
    new_path_upper.append(new_upper[0])
    new_node_lower = new_path_lower, new_lower[1], new_lower[2], new_lower[3]
    new_node_upper = new_path_upper, new_upper[1], new_upper[2], new_upper[3]
    return new_node_lower, new_node_upper


def update_max_node_size(tree):
    max_size = 0
    for node in tree:
        if node[3] > max_size:
            max_size = node[3]
    return max_size


def update_min_depth(tree):
    path_0 = tree[0][0]
    min_depth = len(path_0)
    for node in tree:
        path = node[0]
        if len(path) < min_depth:
            min_depth = len(path)
    return min_depth


def best_variable(xy_parent, loss_parent, min_node_size, features):
    """This function finds the optimal split among all variables for the data in x_parent.
    Input: the data, the list of feature columns, the name of the output column
    Output: variables for the pair of nodes:
                    ((j,s, 'lower'), loss, pred),
                    ((j,s, 'upper'), loss, pred),
                    information_gain
    """
    # initialize output values in case no split reduces loss
    igain = 0
    new_col = None
    new_variables = None
    for col in features:  # don't use the last column of contains targets
        # find best cut for current col
        # variable order is: loss_lower, loss_upper, pred_lower, pred_upper, num_lower, num_upper, cut
        current_variables = best_cut(xy_parent, col, loss_parent, min_node_size)
        if current_variables is not None:
            (
                loss_lower,
                loss_upper,
                pred_lower,
                pred_upper,
                num_lower,
                num_upper,
                cut_value,
            ) = current_variables
            # update variables if infogain is bigger for this variable/col
            num_parent = xy_parent.shape[0]
            current_info_gain = info_gain(
                loss_parent, loss_upper, loss_lower, num_parent, num_upper, num_lower
            )
            if current_info_gain > igain:
                new_variables = (
                    loss_lower,
                    loss_upper,
                    pred_lower,
                    pred_upper,
                    num_lower,
                    num_upper,
                    cut_value,
                )
                new_col = col
                igain = current_info_gain
    if new_col is None:
        return None, None, None
    elif new_variables is None:
        return None, None, None
    else:
        (
            loss_lower,
            loss_upper,
            pred_lower,
            pred_upper,
            num_lower,
            num_upper,
            cut_value,
        ) = new_variables
        return (
            ((new_col, cut_value, "lower"), loss_lower, pred_lower, num_lower),
            ((new_col, cut_value, "upper"), loss_upper, pred_upper, num_upper),
            igain,
        )


def best_cut(xy_parent, col, loss_parent, min_node_size):
    """find the cut that maximizes the information gain

    Input:
    xy_parent - np array of data in parent node
    col - the col index to cut along
    loss_parent - the loss of the parent node
    min_node_size - the minimum size of a node allowed

    Output:
    new_variables = loss_lower, loss_upper, pred_lower, pred_upper, num_lower, num_upper, cut_value
    """
    num_parent = xy_parent.shape[0]
    xy_parent = sort_parent(xy_parent, col)
    igain = 0
    new_variables = None
    num_parent = xy_parent.shape[0]

    one_counts = np.cumsum(xy_parent[:, -1])
    total_counts = np.arange(1, xy_parent.shape[0] + 1)
    zero_counts = total_counts - one_counts

    lower_nums = total_counts
    upper_nums = num_parent - lower_nums
    lower_one_counts = one_counts
    lower_zero_counts = zero_counts
    upper_one_counts = one_counts[-1] - lower_one_counts
    upper_zero_counts = zero_counts[-1] - lower_zero_counts

    # create mask to satisfy min_node_size bound:
    nums_mask = (lower_nums >= min_node_size) & (upper_nums >= min_node_size)
    # array of indices for the True values of the mask
    if np.any(nums_mask) == False:
        return None
    elif np.max(xy_parent[:, col]) == np.min(xy_parent[:, col]):
        return None
    else:
        inds = np.nonzero(nums_mask)[0]
        # mask the children data
        lower_nums = lower_nums[nums_mask]
        upper_nums = upper_nums[nums_mask]
        lower_one_counts = lower_one_counts[nums_mask]
        lower_zero_counts = lower_zero_counts[nums_mask]
        upper_one_counts = upper_one_counts[nums_mask]
        upper_zero_counts = upper_zero_counts[nums_mask]

        with np.errstate(divide="ignore", invalid="ignore"):
            lower_losses = np.where(
                (lower_one_counts == 0) | (lower_zero_counts == 0),
                0,
                loss_fnc(lower_zero_counts, lower_one_counts),
            )
            upper_losses = np.where(
                (upper_one_counts == 0) | (upper_zero_counts == 0),
                0,
                loss_fnc(upper_zero_counts, upper_one_counts),
            )
            igains = np.where(
                (num_parent == 0),
                0,
                info_gain(
                    loss_parent,
                    upper_losses,
                    lower_losses,
                    num_parent,
                    upper_nums,
                    lower_nums,
                ),
            )
        if np.max(igains) > 0:
            ind = np.argmax(igains)
            cut_value = find_cut_value(xy_parent, col, inds[ind])
            num_lower = lower_nums[ind]
            num_upper = upper_nums[ind]
            loss_lower = lower_losses[ind]
            loss_upper = upper_losses[ind]
            lower_zero_count = lower_zero_counts[ind]
            lower_one_count = lower_one_counts[ind]
            upper_zero_count = upper_zero_counts[ind]
            upper_one_count = upper_one_counts[ind]
            pred_lower = predict_from_counts(lower_zero_count, lower_one_count)
            pred_upper = predict_from_counts(upper_zero_count, upper_one_count)
            return (
                loss_lower,
                loss_upper,
                pred_lower,
                pred_upper,
                num_lower,
                num_upper,
                cut_value,
            )
        else:
            return None


def count_classes(y):
    # takes a numpy array
    num = y.shape[0]
    count_1 = y.sum()
    count_0 = num - count_1
    return count_0, count_1, num


def find_cut_value(xy_parent, col, ind):
    """
    Find the value to make the cut for col that separates xy_parent
    into xy_lower and xy_upper."""
    return 0.5 * (xy_parent[ind, col] + xy_parent[ind + 1, col])


def predict_from_counts(
    count_0,
    count_1,
):
    """
    Predict binary class {0,1} based upon the counts of 0's and 1's in the data set.
    """
    if max(count_0, count_1) > 0:
        if count_0 >= count_1:
            pred = 0
        else:
            pred = 1
    else:
        pred = None
    return pred


def sort_parent(xy_parent, col):
    xy_parent = xy_parent[xy_parent[:, col].argsort()]
    return xy_parent

@njit
def info_gain(loss_parent, loss_upper, loss_lower, num_parent, num_upper, num_lower):
    """
    Calculate the information gain for splitting the data in parent node into upper and lower nodes.
    Input: loss_parent, loss_upper, loss_lower, num_parent, num_upper, num_lower
    Output: info gain for splitting the parent node
    """
    ig = (
        loss_parent
        - num_upper / num_parent * loss_upper
        - num_lower / num_parent * loss_lower
    )
    return ig

@njit
def loss_fnc(num_0, num_1):
    """loss function -- Only accepts nonzero inputs
    input:
    num_0, num_1 - NONZERO class counts
    output: a scalar loss -- cross entropy"""
    p = num_1 / (num_0 + num_1)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)  # cross-entropy
