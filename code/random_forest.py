import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
        random_state = 0
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.trained_trees = []
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        for _ in range(self.n_estimators): # Create the forest
            # Initialize the random seed for each tree
            np.random.seed(self.random_state)

            # Sample random subset (indicies)
            random_data_subset = self.rng.choice(np.arange(X.shape[0]), X.shape[0], replace=True)

            X_subset = X[random_data_subset]
            y_subset = y[random_data_subset]

            # Create the tree
            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features, criterion=self.criterion)

            # Fit the tree
            tree.fit(X_subset, y_subset)

            self.trained_trees.append(tree)


    def predict(self, X: np.ndarray) -> np.ndarray:
        # Get the predictions of every tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trained_trees])
        
        # Take the majority vote across all trees
        majority_vote = []
        for i in range(tree_predictions.shape[1]): # Look at each sample's predictions
            votes = tree_predictions[:, i] # Collect predictions
            majority_vote.append(np.bincount(votes).argmax()) # Find the most common vote for this tree

        majority_vote = np.array(majority_vote)
        return majority_vote

if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
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

    rf = RandomForest(
        n_estimators=30, max_depth=None, criterion="entropy", max_features=None, random_state=seed
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
