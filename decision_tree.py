from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def sum_squared_error(y_true, y_pred):
    # use sum as metric so that a split is automatically weighted by number of instances in each split
    if len(y_true) > 0:
        return np.square(y_true-y_pred).sum()
    else:
        return 0

# find losses for variable var for each posiible split
def get_all_losses_per_var(df: pd.DataFrame(), var: str):
    var_vals = np.sort(df[var].unique())
    sse_list = []
    l_vals = []
    r_vals = []
    # for each possible  value calc left and right sided errors
    for v in var_vals:
        idx = df[var] <= v
        l_vals.append(df.loc[idx, 'MEDV'].mean())
        r_vals.append(df.loc[~idx, 'MEDV'].mean())
        sse_l = sum_squared_error(df.loc[idx, 'MEDV'], l_vals[-1])
        sse_r = sum_squared_error(df.loc[~idx, 'MEDV'], r_vals[-1])
        sse_list.append(sse_l + sse_r)
    return np.array(sse_list), var_vals, l_vals, r_vals

def get_opt_split_per_var(df: pd.DataFrame(), var: str):
    sse_list, vals, l_vals, r_vals = get_all_losses_per_var(df, var)
    imin = np.argmin(sse_list)
    return sse_list[imin], vals[imin], l_vals[imin], r_vals[imin]

def get_opt_split(df: pd.DataFrame(), y_col):

    best_sse = 10e10
    for col in df.columns:
        if col == y_col:
            continue
        sse, split_val, l_val, r_val = get_opt_split_per_var(df, col)
        if sse < best_sse:
            best_sse = sse
            best_col = col
            best_split = split_val
            best_l_val = l_val
            best_r_val = r_val
    return best_col, best_split, best_l_val, best_r_val

class Node():
    def __init__(self, val):
        self.val = val # best variable, split_value, left mean val, right mean val
        self.left = None # left branch
        self.right = None

class DecisionTree():
    def __init__(self, max_depth=3, min_leaf_samples=20):
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.tree = None
    def splitting(self, df, y_col, depth=1):
        if len(df) < self.min_leaf_samples or depth > self.max_depth:
            return None
        print(df.shape, depth)
        best_col, best_split, best_l_val, best_r_val = get_opt_split(df, y_col)
        node = Node((best_col, best_split, best_l_val, best_r_val))

        idx = df[best_col] < best_split
        data_left = df[idx]
        data_right = df[~idx]
        node.left = self.splitting(data_left, y_col, depth+1)
        node.right = self.splitting(data_right, y_col, depth+1)

        return node

    def fit(self, df, y_col):
        self.tree = self.splitting(df, y_col)
        return self.tree

    def predict(self, row, tree=None):
        if not self.tree:
            raise ValueError("Call fit method first to create tree")
        if not tree:
            tree = self.tree

        # val = (best_col, best_split, best_l_val, best_r_val)
        col_split = tree.val[0]
        if row[col_split] <= tree.val[1]:
            if not tree.left:
                return tree.val[2]
            else:
                return self.predict(row, tree.left)
        else:
            if not tree.right:
                return tree.val[3]
            else:
                return self.predict(row, tree.right)


    def print_tree(self, tree, tab=""):
        print(tab + str(tree.val))
        if tree.left:
            self.print_tree(tree=tree.left, tab=tab+"\t")
        if tree.right:
            self.print_tree(tree=tree.right, tab=tab+"\t")


if __name__ == '__main__':

    # simplified conditions: depth = 3, min_split of leaf= 20

    data = load_boston()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = pd.DataFrame(data['target'], columns=['MEDV'])
    df = pd.concat([X, y], axis=1)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=1)

    DT = DecisionTreeRegressor(max_depth=3, min_samples_split=20)
    DT.fit(df_train[df_train.columns[:-1]], df_train['MEDV'])
    plt.figure(figsize=(12, 15))
    plot_tree(DT, feature_names=df_train.columns[:-1])
    plt.show()
    y_pred_test = DT.predict(df_test[df_train.columns[:-1]])
    print(mean_squared_error(df_test['MEDV'], y_pred_test))


    DT = DecisionTree(max_depth=3, min_leaf_samples=20)
    splitting_tree = DT.fit(df_train, 'MEDV')

    print(df_train.shape)
    DT.print_tree(splitting_tree)

    print(df_test.iloc[0])
    print(DT.predict(df_test.iloc[0]))

    y_pred_test = df_test.apply(DT.predict, axis=1)
    print(y_pred_test[:10])
    print(mean_squared_error(df_test['MEDV'], y_pred_test))