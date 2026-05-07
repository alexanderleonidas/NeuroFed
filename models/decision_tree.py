from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import torch

# Create a wrapper class for sklearn's DecisionTreeClassifier
# that provides a PyTorch-compatible interface
class DecisionTreeWrapper():
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.model = DecisionTreeClassifier(max_depth=20, random_state=42)
        self.is_fitted = False

        params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 2, 4, 6, 8, 10],
            'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
            'splitter': ['best', 'random']
        }

        self.clf = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params,cv=5, n_jobs=5,verbose=1,)
    #
    # def forward(self, x):
    #     # If not fitted yet, return random outputs (will be properly fitted during training)
    #     if not self.is_fitted:
    #         raise RuntimeError('model is not fitted')
    #
    #     # Convert PyTorch tensor to a numpy array for sklearn
    #     if isinstance(x, torch.Tensor):
    #         x_np = x.cpu().numpy()
    #     else:
    #         x_np = x
    #
    #     # Get probability predictions and convert back to PyTorch tensor
    #     probs = self.model.predict_proba(x_np)[:, 1:2]  # Get positive class probability
    #     return torch.from_numpy(probs).to(x.device if isinstance(x, torch.Tensor) else 'cpu').float()

    def fit(self, x, y):
        # Method to fit the decision tree
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y

        self.clf.fit(x_np, y_np)
        self.is_fitted = True

    def predict(self, x):
        # Method to predict class labels
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        return self.clf.predict(x_np)

    def predict_proba(self, x):
        # Method to predict class probabilities
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        return self.clf.predict_proba(x_np)

    @staticmethod
    def get_accuracy(y_test, predictions):
        return accuracy_score(y_test, predictions)