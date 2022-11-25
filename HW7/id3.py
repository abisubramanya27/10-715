import pandas as pd
import numpy as np
from math import log
from random import sample
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import KFold
from itertools import takewhile

class Node:
    def __init__(self, depth, node_type='normal', label=None, feature=None, class_dict=None):
        assert (class_dict is None and feature is None) or (class_dict is not None and feature is not None)
        self.depth = depth
        self.node_type = node_type
        self.feature_to_split_on = feature
        self.label = label
        self.children = {value: None for value in class_dict[feature]} if feature is not None else {}
    
    def add_split_feature(self, feature, class_dict):
        self.feature_to_split_on = feature
        self.children = {value: None for value in class_dict[feature]} if feature is not None else {}
        
    
class DecisionTree:
    def __init__(self, X, Y, max_depth, class_dict):
        self.max_depth = max_depth
        self.class_dict = class_dict
        self.root = self.build(X, Y, 0)
        self.features = list(X.columns)
    
    def build(self, X, Y, depth):
        node = Node(depth, 'root' if depth == 0 else 'normal')
        counter = Counter(Y)
        if counter.most_common(1)[0][1] == len(Y) or depth == self.max_depth:
            node.node_type = 'leaf'
            all_most_common = list(takewhile(lambda val: val[1] == counter.most_common(1)[0][1], counter.most_common()))
            all_most_common.sort()
            node.label = all_most_common[0][0]
            return node
        
        split_feature = self.get_best_feature_to_split_on(X, Y)
        node.add_split_feature(split_feature, self.class_dict)
        
        for value in self.class_dict[split_feature]:
            eff_idx = X[split_feature] == value
            X_eff, Y_eff = X[eff_idx].loc[:, X.columns != split_feature], Y[eff_idx]
            if len(Y_eff) == 0:
                node.children[value] = Node(
                    depth+1, 'leaf', counter.most_common(1)[0][0]
                )
            else:
                node.children[value] = self.build(X_eff, Y_eff, depth+1)
        
        return node             

    @staticmethod
    def gini_index(y_counts):
        sum_counts = np.sum(y_counts)
        return 1 - np.sum(list(map(lambda x: (x/sum_counts)**2, y_counts)))

    def get_best_feature_to_split_on(self, X, Y):
        features = list(X.columns)
        best_feature = None
        best_gini_gain = -np.inf
        
        gini_gain_base = self.gini_index([info[1] for info in Counter(Y).most_common()])
        for feature in features:
            gini_gain = gini_gain_base
            for value in self.class_dict[feature]:
                eff_idx = X[feature] == value
                Y_eff = Y[eff_idx]
                y_counts = [info[1] for info in Counter(Y_eff).most_common()]
                gini_gain -= len(Y_eff) * self.gini_index(y_counts) / len(Y)
            
            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_feature = feature
        
        return best_feature
    
    def predict(self, X):
        Y_hat = []
        for _, x in X.iterrows():
            x_features = {}
            for feature in self.features:
                x_features.update({feature: x[feature]})
        
            Y_hat.append(self.predict_obs(x_features, self.root))
        
        return Y_hat
    
    def predict_obs(self, x, node):
        if node.node_type == 'leaf':
            return node.label
    
        feature_value = x[node.feature_to_split_on]
        return self.predict_obs(x, node.children[feature_value])
    
    def accuracy(self, Y, Y_hat):
        return np.mean(np.array(Y, dtype=np.bool8) == np.array(Y_hat, dtype=np.bool8))

KFOLD = True
TEST = True

if __name__ == "__main__":
    train = pd.read_csv('Q1_data/train.csv')
    train['target'] = train['target'] == 'recurrence-events'
    test = pd.read_csv('Q1_data/test.csv')
    test['target'] = test['target'] == 'recurrence-events'
    attributes = ['age', 'menopause', ' tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad',
                  'irradiant']
    train['deg_malig'] = train['deg_malig'].astype(str)
    test['deg_malig'] = test['deg_malig'].astype(str)

    train = train.sample(frac=1).reset_index(drop=True)

    class_dict = {'age': ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
                  'menopause': ["lt40", "ge40", "premeno"],
                  ' tumor_size': ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                                  "50-54", "55-59"],
                  'inv_nodes': ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29",
                                "30-32", "33-35", "36-39"],
                  'node_caps': ["yes", "no"],
                  'deg_malig': ["1", "2", "3"],
                  'breast': ["left", "right"],
                  'breast_quad': ["left_up", "left_low", "right_up", "right_low", "central"],
                  'irradiant': ["yes", "no"]}
    
    if KFOLD:
        best_max_depth = -1
        best_val_acc = -1
        max_depths = [1, 3, 5, 7, 9]
        train_accuracies, val_accuracies = [], []
        for max_depth in max_depths:
            kfold_train_acc, kfold_val_acc = [], []
            kf = KFold(n_splits = 3, shuffle=True, random_state=2022)
            for train_index, val_index in kf.split(train):   
                X_train, Y_train = train.loc[train_index, train.columns != 'target'], train.loc[train_index, 'target']
                X_val, Y_val = train.loc[val_index, train.columns != 'target'], train.loc[val_index, 'target']
                dt = DecisionTree(X_train, Y_train, max_depth, class_dict)
                Y_train_hat = dt.predict(X_train)
                kfold_train_acc.append(dt.accuracy(Y_train, Y_train_hat))
                Y_val_hat = dt.predict(X_val)
                kfold_val_acc.append(dt.accuracy(Y_val, Y_val_hat))

            train_accuracies.append(np.mean(kfold_train_acc))
            val_accuracies.append(np.mean(kfold_val_acc))
            print(f'Train Accuracy for max-depth {max_depth}:', train_accuracies[-1])
            print(f'Val Accuracy for max-depth {max_depth}:', val_accuracies[-1])
            if val_accuracies[-1] > best_val_acc:
                best_val_acc = val_accuracies[-1]
                best_max_depth = max_depth

        plt.figure(0)
        plt.plot(max_depths, train_accuracies, linestyle='-.', color='b', marker='o', label='Train Accuracy')
        plt.plot(max_depths, val_accuracies, linestyle='-', color='orange', marker='x', label='Validation Accuracy')
        plt.title('Accuracy vs Max Depth of Decision Tree')
        plt.ylabel('Accuracy')
        plt.xlabel('Max Depth')
        plt.legend(loc='best')
        plt.savefig(f'./plots/dt_acc_2.png')
        plt.show()

    if TEST:
        MAX_DEPTH = 3 if not KFOLD else best_max_depth
        X_train, Y_train = train.loc[:, train.columns != 'target'], train['target']
        X_test, Y_test = test.loc[:, test.columns != 'target'], test['target']

        dt = DecisionTree(X_train, Y_train, MAX_DEPTH, class_dict)
        Y_train_hat = dt.predict(X_train)
        Y_test_hat = dt.predict(X_test)

        print(f'Final Train Accuracy (after training on all data) with max-depth {MAX_DEPTH}:', dt.accuracy(Y_train, Y_train_hat))
        print(f'Final Test Accuracy (after training on all data) with max-depth {MAX_DEPTH}:', dt.accuracy(Y_test, Y_test_hat))

        
    
        

