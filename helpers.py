# helper functions that are used in both scripts.
import numpy as np

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score
from collections import Counter, defaultdict
import random


def visualize_data(x, y, title_string):
    """Plots the PCA of the data.

    Parameters
    ----------
    x : input features.
    y : labels.
    title_string : a string used in the title of the plot e.g. "IDT".
    """
    pca = PCA(n_components=2)
    x_r = pca.fit(x).transform(x)

    plt.figure()
    colors = ['navy', 'turquoise']
    target_names = ['truth', 'lie']

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(x_r[y == i, 0], x_r[y == i, 1], color=color, alpha=.8, lw=2,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(f'Visualization of Truths and Lies in 2D from {title_string} features.')


def stratified_group_k_fold(x, y, groups, k, seed=None):
    """Splits the data into k folds.

    Folds are made by preserving the percentage of samples for each class.
    The same group will not appear in two different folds.

    This function is needed to ensure the same subject does not appear in different
    folds to avoid the algorithm degenerating to person re-identification.

    Taken from here: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation

    Parameters
    ----------
    x : input features.
    y : labels.
    groups : the groups the data points belong to (same length as y).
    k : numbers of folds.
    seed :

    Returns
    -------
    train_indices
    test_indices
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)
    
    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def cross_validation(x, y_lie, y_gender, subjects, run_gender, classifier,
                     use_predict=True, folds=10, iterations=25):
    """Performs cross validation.

    Parameters
    ----------
    x : input features.
    y_lie : lie/truth labels.
    y_gender : gender labels.
    subjects : the identities each data points belongs to.
    run_gender : whether to run cross validation using gender labels or
                 lie/truth labels.
    classifier : the classifier from sklearn.
    use_predict : whether to use "predict_proba" to find target predictions or
                  directly "predict". This is necessary for SVC of sklearn as
                  the two methods are inconsistent and, in the IDt experiment,
                  the two gave better results depending on the type of experiment.
    folds : numbers of folds.
    iterations : number of times cross validation should be performed.

    Returns
    -------
    auc_score_mean_list : a list of AUC metrics for each iteration where AUC
                          is averaged ignoring the fold sizes.
    auc_score_weighted_list : a list of AUC metrics for each iteration where AUC
                          is weighted by the fold sizes.
    accuracy_mean_list : a list of accuracy metrics for each iteration where
                         accuracy is averaged ignoring the fold sizes.
    accuracy_weighted_list : a list of accuracy metrics for each iteration where
                         accuracy is weighted by the fold sizes.
    """
    auc_score_mean_list = []
    auc_score_weighted_list = []
    accuracy_mean_list = []
    accuracy_weighted_list = []

    for _ in range(iterations):

        auc_score_mean = 0
        auc_score_weighted = 0
        accuracy_mean = 0
        accuracy_weighted = 0

        for fold_ind, (train_index, val_index) in enumerate(
                       stratified_group_k_fold(x, y_lie, subjects, k=folds)):
            
            y_train = y_gender[train_index] if run_gender else y_lie[train_index]
            y_val = y_lie[val_index]
            
            x_train, x_val = x[train_index, :], x[val_index, :]
            
            classifier.fit(x_train, y_train)
            
            predict_proba = classifier.predict_proba(x_val)[:, 1]
            if use_predict:
                predict_target = classifier.predict(x_val)
            else:
                predict_target = np.where(predict_proba > 0.5, 1, 0)
            
            auc_score_mean += roc_auc_score(y_val, predict_proba) / folds
            auc_score_weighted += roc_auc_score(y_val, predict_proba) * len(y_val) / len(y_lie)
            
            accuracy_mean += accuracy_score(y_val, predict_target) / folds
            accuracy_weighted += accuracy_score(y_val, predict_target) * len(y_val) / len(y_lie)
        
        auc_score_mean_list.append(auc_score_mean)
        auc_score_weighted_list.append(auc_score_weighted)
        accuracy_mean_list.append(accuracy_mean)
        accuracy_weighted_list.append(accuracy_weighted)  

    return auc_score_mean_list, auc_score_weighted_list, accuracy_mean_list, accuracy_weighted_list


def print_results(parameters, mean_auc, weighted_auc, mean_acc, weighted_acc):
    """Prints the results of cross_validation."""
    print(f"""
    ##########################################################
    CHECKING {parameters}
    ----------------------------------
    WHEN AVERAGING FOLDS: ACCURACY
    mean: {np.mean(mean_acc)}
    max: {np.max(mean_acc)}
    min: {np.min(mean_acc)}
    std: {np.std(mean_acc)}
    ----------------------------------
    WHEN WEIGHTING FOLDS: ACCURACY
    mean: {np.mean(weighted_acc)}
    max: {np.max(weighted_acc)}
    min: {np.min(weighted_acc)}
    std: {np.std(weighted_acc)}
    ----------------------------------
    WHEN AVERAGING FOLDS: AUC
    mean: {np.mean(mean_auc)}
    max: {np.max(mean_auc)}
    min: {np.min(mean_auc)}
    std: {np.std(mean_auc)}
    ----------------------------------
    WHEN WEIGHTING FOLDS: AUC
    mean: {np.mean(weighted_auc)}
    max: {np.max(weighted_auc)}
    min: {np.min(weighted_auc)}
    std: {np.std(weighted_auc)}
    """)


def collect_results(x, y_lie, y_gender, subjects, run_gender, parameters, classifier,
                    hyperparameter_map, use_predict=True, folds=10, iterations=25):
    """Prints the results of cross validation and puts them in a map."""
    mean_auc, weighted_auc, mean_acc, weighted_acc = cross_validation(x, y_lie, y_gender, subjects, run_gender,
                                                                      classifier, use_predict=use_predict,
                                                                      folds=folds, iterations=iterations)
    hyperparameter_map[parameters] = (mean_auc, weighted_auc, mean_acc, weighted_acc)
    print_results(parameters, mean_auc, weighted_auc, mean_acc, weighted_acc) 
