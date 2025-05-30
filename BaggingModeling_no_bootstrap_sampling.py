import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graphviz import Digraph
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.tree import export_graphviz

MIN_INFO_GAIN = -1000000
TITANIC_DS_TARGET = "survived"
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
UNKNOWN = 9999
TARGET = "survived"
TREES_OUTPUT_DIR = "output/trees"
CM_OUTPUT_DIR = "output/cm"


def entropy(data_col: pd.DataFrame):
    val_counts = data_col.value_counts()
    N = data_col.shape[0]
    # print(val_counts)
    entropy = 0.0
    for val, count in val_counts.items():
        """print(val)
        print(count)"""
        p = count / N
        entropy -= p * np.log2(p)

    return entropy


def feature_info_gain(data: pd.DataFrame, target: str, col: str):

    entropy_dataset = entropy(data[target])

    # pd.api.types.is_integer_dtype(df['A'])
    # pd.api.types.is_float_dtype(df['B'])
    thresholds = None

    if pd.api.types.is_float_dtype(data[col]):
        # print(f"float col - mean : {col} - {data[col].mean()}")
        thresholds = [data[col].median()]
    else:
        thresholds = data[col].unique()

    """data_left = pd.DataFrame([],columns=data.columns)
    data_right = pd.DataFrame([],columns=data.columns)"""

    data_left = None
    data_right = None

    info_gain = MIN_INFO_GAIN
    best_threshold = None

    if (len(thresholds) == 1) and not pd.api.types.is_float_dtype(data[col]):
        return info_gain, best_threshold, data_left, data_right

    for thres in thresholds:

        current_data_left = data[data[col] <= thres]
        current_data_right = data[data[col] > thres]

        left_weight = current_data_left.shape[0] / data.shape[0]
        right_weight = current_data_right.shape[0] / data.shape[0]

        left_entropy = entropy(current_data_left[target])
        right_entropy = entropy(current_data_right[target])

        current_info_gain = entropy_dataset - (
            left_weight * left_entropy + right_weight * right_entropy
        )

        if current_info_gain > info_gain:
            info_gain = current_info_gain
            best_threshold = thres
            data_left = current_data_left
            data_right = current_data_right

        # print(f"Split: {col} - {thres} - {current_data_left.shape} - {current_data_right.shape}")
    # print(f"Feature Split: {col} - {best_threshold} - {len(data_left)} - {len(data_right)}")
    return info_gain, best_threshold, data_left, data_right


def best_info_gain_feature_and_data_split(
    data: pd.DataFrame, target_col_name: str, features: list
):

    feat_igain, thres, data_left, data_right = MIN_INFO_GAIN, None, None, None

    target_feat = ""
    for feat in features:

        current_feat_igain, current_thres, current_data_left, current_data_right = (
            feature_info_gain(data, target_col_name, feat)
        )
        if current_feat_igain > feat_igain:
            target_feat = feat
            feat_igain, thres, data_left, data_right = (
                current_feat_igain,
                current_thres,
                current_data_left,
                current_data_right,
            )
    # print(f"best split: {target_feat} - {thres} - {data_left.shape} - {data_right.shape}")
    return target_feat, thres, data_left, data_right


class CustomDecisionTree:
    def __init__(self, target_col_name, feature_names, start_depth=0, max_depth=5):
        self.start_depth = start_depth
        self.max_depth = max_depth
        self.target = target_col_name
        self.features = feature_names
        self.left = None
        self.right = None
        self.leaf_val = None
        self.discriminant_feat = None
        self.threshold = None

    def fit(self, x_train):

        if x_train is None or len(x_train) <= 0:
            self.leaf_val = UNKNOWN
            return

        self.discriminant_feat, self.threshold, data_left, data_right = (
            best_info_gain_feature_and_data_split(x_train, self.target, self.features)
        )

        if (
            self.start_depth >= self.max_depth
            or self.discriminant_feat is None
            or self.threshold is None
        ):

            self.leaf_val = x_train[self.target].value_counts().idxmax()
            # print(f"leaf val setting : {self.leaf_val}")
            return

        # Recursive Case
        self.left = CustomDecisionTree(
            self.target, self.features, self.start_depth + 1, self.max_depth
        )
        if data_left is not None and len(data_left) > 0:
            data_left = data_left.reset_index(drop=True)
            self.left.fit(data_left)
        else:
            self.left.fit(None)

        self.right = CustomDecisionTree(
            self.target, self.features, self.start_depth + 1, self.max_depth
        )
        if data_right is not None and len(data_right) > 0:
            data_right = data_right.reset_index(drop=True)
            self.right.fit(data_right)
        else:
            self.right.fit(None)

    def predict(self, data):
        return pd.Series(
            [self.predict_one(row) for _, row in data.iterrows()], index=data.index
        )

    def predict_one(self, row):

        if self.left is None or self.right is None:
            return self.leaf_val

        if row[self.discriminant_feat] <= self.threshold:
            result = self.left.predict_one(row)
            """if result is None:
                print(f"This row returns nan for 'row[{self.discriminant_feat}] <= {self.threshold}' left criterion: {row}")"""
            return result
        else:
            result = self.right.predict_one(row)
            """if result is None:
                print(f"This row returns nan for 'row[{self.discriminant_feat}] > {self.threshold}' right criterion: {row}")"""
            return result


"""dot_data = export_graphviz(dt_model, out_file=None,feature_names=FEATURES,class_names=["dead", "survived"], impurity=False,filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())"""


def visualize_tree(node, graph=None, node_id=0):
    if graph is None:
        graph = Digraph()

    current_id = str(node_id)
    # print(node_id)
    if node.leaf_val is not None:
        graph.node(current_id, label=f"Leaf\nValue={node.leaf_val}")
    else:
        graph.node(current_id, label=f"X[{node.discriminant_feat}] <= {node.threshold}")
        left_id = str(2 * node_id + 1)
        right_id = str(2 * node_id + 2)
        if node.left is not None:
            graph.edge(current_id, left_id, label="True")
            visualize_tree(node.left, graph, 2 * node_id + 1)

        if node.right is not None:
            graph.edge(current_id, right_id, label="False")
            visualize_tree(node.right, graph, 2 * node_id + 2)

    return graph


# print(dt_model)


# data are dataframe columns, series or numpy arrays
def display_cm(y_test, y_pred, y_all, cm_title, pred_axis_label, true_axis_label):
    # Calculate and display the confusion matrix
    class_labels = np.unique(y_all)  # Use all unique target labels
    class_labels = np.append(class_labels, UNKNOWN)
    cm = confusion_matrix(
        y_test, y_pred, labels=class_labels
    )  # Use all target labels for the confusion matrix
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.title(cm_title)
    plt.xlabel(pred_axis_label)
    plt.ylabel(true_axis_label)
    os.makedirs(CM_OUTPUT_DIR, exist_ok=True)
    # Save the figure
    plt.savefig(
        os.path.join(CM_OUTPUT_DIR, f"{cm_title}.png"), dpi=300, bbox_inches="tight"
    )

    plt.show()
    # Optional: close the plot to free memory
    plt.close()


def get_acc_score(y_test, y_pred):

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    return acc


class CustomRandomForest:
    def __init__(
        self, num_of_trees, target_col_name, feature_names, start_depth=0, max_depth=5
    ):
        self.start_depth = start_depth
        self.max_depth = max_depth
        self.target = target_col_name
        self.features = feature_names
        self.num_of_trees = num_of_trees
        self.trees = None

        num_trees_is_odd = self.num_of_trees % 2 == 1

        if num_trees_is_odd:
            self.trees = [
                CustomDecisionTree(
                    self.target, self.features, self.start_depth, self.max_depth
                )
                for _ in range(self.num_of_trees)
            ]
        else:
            num_columns = len(self.features)
            column_split_idx = int(num_columns / 2)
            left_trees = [
                CustomDecisionTree(
                    self.target,
                    self.features[:column_split_idx],
                    self.start_depth,
                    self.max_depth,
                )
                for _ in range(int(self.num_of_trees / 2))
            ]
            right_trees = [
                CustomDecisionTree(
                    self.target,
                    self.features[column_split_idx:],
                    self.start_depth,
                    self.max_depth,
                )
                for _ in range(int(self.num_of_trees / 2))
            ]
            self.trees = left_trees + right_trees

    def train_data_split(self, x_train):

        num_trees_is_odd = self.num_of_trees % 2 == 1

        if num_trees_is_odd:

            # We split by rows only
            row_chunk_size = int(x_train.shape[0] / self.num_of_trees)
            row_chunks = [
                x_train.iloc[row_chunk_size * idx : row_chunk_size * (idx + 1), :]
                for idx in range(self.num_of_trees - 1)
            ] + [x_train.iloc[row_chunk_size * (self.num_of_trees - 1) :, :]]

            return row_chunks

        else:
            # We split by rows and 2 groups of columns
            num_columns = len(self.features)
            column_split_idx = int(num_columns / 2)
            num_of_row_chunks = int(self.num_of_trees / 2)

            row_chunk_size = int(x_train.shape[0] / num_of_row_chunks)
            # We add the target column (it is the last column of the whole dataset) to the left row chunks
            left_row_chunks = [
                pd.concat(
                    [
                        x_train.iloc[
                            row_chunk_size * idx : row_chunk_size * (idx + 1),
                            :column_split_idx,
                        ],
                        x_train.iloc[
                            row_chunk_size * idx : row_chunk_size * (idx + 1), -1
                        ],
                    ],
                    axis=1,
                )
                for idx in range(num_of_row_chunks - 1)
            ] + [
                pd.concat(
                    [
                        x_train.iloc[
                            row_chunk_size * (num_of_row_chunks - 1) :,
                            :column_split_idx,
                        ],
                        x_train.iloc[row_chunk_size * (num_of_row_chunks - 1) :, -1],
                    ],
                    axis=1,
                )
            ]
            right_row_chunks = [
                x_train.iloc[
                    row_chunk_size * idx : row_chunk_size * (idx + 1), column_split_idx:
                ]
                for idx in range(num_of_row_chunks - 1)
            ] + [
                x_train.iloc[
                    row_chunk_size * (num_of_row_chunks - 1) :, column_split_idx:
                ]
            ]

            return left_row_chunks + right_row_chunks

    def fit(self, x_train):

        x_train_chunks = self.train_data_split(x_train)

        for i in range(self.num_of_trees):
            # print(f"tree {i} features : {self.trees[i].features}")
            # print(f"data chunk {i} columns : {x_train_chunks[i].columns}")
            self.trees[i].fit(x_train_chunks[i])

    def predict(self, data):
        return pd.Series(
            [self.predict_one(row) for _, row in data.iterrows()], index=data.index
        )

    def predict_one(self, row):
        tree_predictions = [tree.predict_one(row) for tree in self.trees]
        counts = Counter(tree_predictions)
        # print(counts)

        # Get the most common element
        most_common = counts.most_common(1)

        most_common_value = most_common[0][0]
        # most_common_count = most_common[0][1]

        return most_common_value


if __name__ == "__main__":

    ###############################
    # DATA LOAD AND PREPROCESSING #
    ###############################

    print(
        " ###############################\n \
# DATA LOAD AND PREPROCESSING #\n \
###############################"
    )

    df_init = pd.read_excel("titanic.xlsx", sheet_name="titanic3", header=0)

    print(df_init.head())

    df = df_init[FEATURES + [TARGET]]

    print(df.isnull().sum())

    """df["sex"][df["sex"] == "male"] = 1
    df["sex"][df["sex"] == "female"] = 0"""

    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df["age"].fillna(df["age"].mean(), inplace=True)
    df["fare"].fillna(df["fare"].mean(), inplace=True)

    print(df.head(10))

    print(df.shape)
    print(df.isnull().sum())
    print(df.dtypes)

    os.makedirs(TREES_OUTPUT_DIR, exist_ok=True)

    #########################
    # TRAIN TEST DATA SPLIT #
    #########################

    print(
        " #########################\n \
# TRAIN TEST DATA SPLIT #\n \
#########################"
    )

    train_size = int(0.85 * df.shape[0])
    X_TRAIN = df[:train_size]
    X_TEST = df[train_size:]

    ################################
    # SIMPLE MODEL : DECISION TREE #
    ################################

    print(
        " ################################\n \
# SIMPLE MODEL : DECISION TREE #\n \
################################"
    )

    dt_model = CustomDecisionTree(TARGET, FEATURES, 0, 5)

    dt_model.fit(X_TRAIN)

    dot = visualize_tree(dt_model)
    dot.render(
        f"{TREES_OUTPUT_DIR}/tree", format="png", cleanup=False
    )  # Creates tree.png

    print(X_TEST.head(10))

    Y_Predict = dt_model.predict(X_TEST)
    print(Y_Predict.value_counts())

    display_cm(
        X_TEST[TARGET],
        Y_Predict,
        df[TARGET],
        "Confusion Matrix for Titanic decision tree",
        "Predicted Survival Label",
        "True Survival Label",
    )
    get_acc_score(X_TEST[TARGET], Y_Predict)

    ########################################
    # BAGGING MODEL : CUSTOM RANDOM FOREST #
    ########################################

    print(
        " ########################################\n \
# BAGGING MODEL : CUSTOM RANDOM FOREST #\n \
########################################"
    )

    rf_model = CustomRandomForest(150, TARGET, FEATURES, 0, 2)

    rf_model.fit(X_TRAIN)

    for i, tree in enumerate(rf_model.trees):
        dot = visualize_tree(tree)
        dot.render(
            f"{TREES_OUTPUT_DIR}/tree_{i}", format="png", cleanup=False
        )  # Creates tree.png
        # print(dt_model)

    Y_RF_Predict = rf_model.predict(X_TEST)
    print(Y_RF_Predict.value_counts())

    display_cm(
        X_TEST[TARGET],
        Y_RF_Predict,
        df[TARGET],
        "Confusion Matrix for Titanic Random Forest Model",
        "Predicted Survival Label",
        "True Survival Label",
    )
    get_acc_score(X_TEST[TARGET], Y_RF_Predict)
