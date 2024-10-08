import numpy as np
from typing import Self

"""
I chose to follow the suggested template for this obligatory assignment.
"""

def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """
    # Get the unique values and respective counts
    _, counts = np.unique(y, return_counts=True)

    # Get the acending percentage / proportions of respective counts from values
    proportions = counts / counts.sum()

    return proportions


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    proportions = count(y)
    gini_index = 1 - np.sum(proportions**2)
    return gini_index


def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    proportions = count(y)
    entropy = - np.sum(proportions * np.log2(proportions))
    return entropy


def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """
    return x <= value


def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    values, counts = np.unique(y, return_counts=True)
    index = np.argmax(counts) # Index of largest count
    return values[index] # Value with largest count


class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """

    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0.0,
        left: int | Self | None = None,
        right: int | Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        # Return True iff the node is a leaf node
        return self.value is not None


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        max_features: str | None = None,
        criterion: str = "entropy",
        random_state: int = 0
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)



    def fit(self, X: np.ndarray, y: np.ndarray, current_depth: int = 0):
        """
        Fit the decision tree to the training data.
        """
        # Initialize the random seed
        np.random.seed(self.random_state)

        # Fit on the root
        if current_depth == 0:
            self.root = self.fit(X, y, current_depth=1)

        ## Check base cases

        # Return a leaf node with given label, when all data points have the same label
        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        
        # Return leaf node with most common label, when all data points have identical feature values
        if np.all(X == X[0]):
            return Node(value=most_common(y))

        if self.max_depth is not None and current_depth >= self.max_depth: # Check if stopping condition is met
            return Node(value=most_common(y)) # Ensure a leaf node is returned, e.g not a None "node"
        

        # Get best feature and split, and use it to make the tree nodes
        best_feature, best_threshold, best_split_left, best_split_right = self.get_best_split(X, y)

        # Split data
        X_left, y_left = best_split_left
        X_right, y_right = best_split_right
        # If either split is empty, handle this case
        if len(X_left) == 0 or len(X_right) == 0:
            return Node(value=most_common(y))

        # Recursive call for left subtreee
        left_subtree = self.fit(X_left, y_left, current_depth + 1)

        # Recursive call for right subtree
        right_subtree = self.fit(X_right, y_right, current_depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def get_best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find the best feature and threshold to split the data to maximize information gain.
        """
        # Inits for the best values
        best_feature = None
        best_threshold = None
        best_information_gain = -float("inf")
        best_split_left = None
        best_split_right = None

        # Choose what features to consider for the split
        features_to_consider = None
        if self.max_features is None: # Use all features
            features_to_consider = np.arange(X.shape[1])
        elif self.max_features == "sqrt": # Use a random subset of sqrt of the features
            feature_subset_size = int(np.floor(np.sqrt(X.shape[1])))
            features_to_consider = self.rng.choice(np.arange(X.shape[1]), feature_subset_size, replace=False)
        elif self.max_features == "log2": # Use a random subset of log2 of the features
            feature_subset_size = int(np.floor(np.log2(X.shape[1])))
            features_to_consider = self.rng.choice(np.arange(X.shape[1]), feature_subset_size, replace=False)
            
        # Loop over all features
        for feature_index in features_to_consider:
            X_left, y_left, X_right, y_right = self.calculate_median_split(X, y, feature_index)

            information_gain = self.calculate_information_gain(y, y_left, y_right)

            if information_gain > best_information_gain:
                best_feature = feature_index
                best_threshold = np.median(X[:, best_feature])
                best_split_left = (X_left, y_left)
                best_split_right = (X_right, y_right)
                best_information_gain = information_gain
        
        return best_feature, best_threshold, best_split_left, best_split_right

    def calculate_information_gain(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Calculate the information gain on the current split
        """

        # Calculate weights
        weight_l = len(y_left) / len(y_parent)
        weight_r = len(y_right) / len(y_parent)

        if self.criterion == "entropy":
            # Use entropy -> (parent entropy - weighted average child entropy)
            information_gain = entropy(y_parent) - (weight_l * entropy(y_left) + weight_r * entropy(y_right))
        elif self.criterion == "gini":
            # Use Gini index
            information_gain = gini_index(y_parent) - (weight_l * gini_index(y_left) + weight_r * gini_index(y_right))

        return information_gain


    def calculate_median_split(self, X: np.ndarray, y: np.ndarray, feature_index: int) -> tuple:
        """
        Calculate the median split on the features
        """
        # Median feature value
        features = X[:, feature_index]
        median = np.median(features)

        # Split
        left_boolean_mask = split(features, median)
        right_boolean_mask = np.logical_not(left_boolean_mask)

        X_left, y_left = X[left_boolean_mask], y[left_boolean_mask]
        X_right, y_right = X[right_boolean_mask], y[right_boolean_mask]

        return X_left, y_left, X_right, y_right

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        predictions = np.array([self.predict_sample(sample, self.root) for sample in X])
        return predictions

    def predict_sample(self, sample, node):
        ## Check base case
        if node.is_leaf():
            return node.value
        
        # Recursive call
        if sample[node.feature] <= node.threshold:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)
    

    def print_tree(self, node: Node, depth: int = 0):
        """
        Print the decision tree.
        """
        if node is None:
            return

        indent = "  " * depth

        if node.is_leaf(): # Hit leaf, get value
            print(f"{indent}Leaf node: Predict {node.value}")
        else: # Continue down
            print(f"{indent}Feature Node: {node.feature}, HAS THRESHOLD {node.threshold:.2f}")
            print(f"{indent}Left subtree:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}Right subtree:")
            self.print_tree(node.right, depth + 1)


if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=4, criterion="entropy", max_features=None, random_state=seed)
    rf.fit(X_train, y_train)
    # rf.print_tree(rf.root)
    print()

    print("Testing:")
    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
