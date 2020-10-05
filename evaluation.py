import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

"""
Positive = 1
Negative = 0
"""


def precision(cm):
    tp = cm.loc[1, 1]
    fp = cm.loc[1, 0]
    tn = cm.loc[0, 0]
    fn = cm.loc[0, 1]
    if tp == 0:
        return 0
    else:
        return tp / (tp + fp)


def sensitivity(cm):
    tp = cm.loc[1, 1]
    fp = cm.loc[1, 0]
    tn = cm.loc[0, 0]
    fn = cm.loc[0, 1]
    if tp == 0:
        return 0
    else:
        return tp / (tp + fn)


def fpr(cm):
    tp = cm.loc[1, 1]
    fp = cm.loc[1, 0]
    tn = cm.loc[0, 0]
    fn = cm.loc[0, 1]
    if fp == 0:
        return 0
    else:
        return fp / (fp + tn)


def tfpn(preds, target):
    """Generates Series for the TP, FN, TN, FP.
    Input:
    preds: 1/0 predictions from a model
    target: 1/0 true values from data

    Output: tuple of series: TP, FN, TN, FP
    Four series that have the indices of the relevant samples and all ones as values.
    """
    mask_positives = preds == 1
    positives = preds[mask_positives]
    target_pred_positive = target[mask_positives]
    true_positives = positives[positives == target_pred_positive]
    false_positives = positives[positives != target_pred_positive]

    mask_negatives = preds == 0
    negatives = preds[mask_negatives]
    target_pred_negative = target[mask_negatives]
    true_negatives = negatives[negatives == target_pred_negative]
    false_negatives = negatives[negatives != target_pred_negative]

    return true_positives, false_negatives, true_negatives, false_positives


def make_confusion_matrix(
    true_positives, false_negatives, true_negatives, false_positives, percentage=False
):
    """Generates a confusion matrix using the sizes of the input pd.Series."""
    cm = pd.DataFrame(columns=[1, 0], index=[1, 0])
    tp = true_positives.shape[0]
    fp = false_positives.shape[0]
    tn = true_negatives.shape[0]
    fn = false_negatives.shape[0]
    tfpn_list = [tp, fp, tn, fn]
    total = sum(tfpn_list)
    if percentage == True:
        tfpn_list = [round(x / total, 3) for x in tfpn_list]
    cm.at[1, 1] = tfpn_list[0]
    cm.at[1, 0] = tfpn_list[1]
    cm.at[0, 0] = tfpn_list[2]
    cm.at[0, 1] = tfpn_list[3]
    cm.index.name = "Predicted"
    cm.columns.name = "True"
    if percentage == True:
        cm = cm.astype(float)
    else:
        cm = cm.astype(int)
    return cm


def plot_cm(cm):
    """Plots the confusion matrix as a heatmap."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, cbar=True)
    plt.title("Confusion Matrix", fontsize=20)
    plt.show()
    # plt.savefig('', bbox_inches = 'tight')
