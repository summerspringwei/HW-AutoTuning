"""MLP based regression model for predicting the speedup."""

import pickle
import os
import random
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import r_regression
from matplotlib import pyplot as plt

from prepare_dataset import preprocessing_features
from utils import get_logger, dataset_path

logger = get_logger(__name__)


def train_regression(X: np.ndarray, y: np.ndarray, train: bool = True):
    """Train a regression model.
     
    Train a regression model using the given features and speedup data. 
    If the model is already trained, load the model from the file.
    
    Parameters:
    ----------
    X : np.ndarray
        Features extracted from the compiler optimization passes.
    y : np.ndarray
        Speedup data.
    train : bool
        Whether to train the model or not.
    
    Returns:
    -------
    y_pred : np.ndarray
        Predicted speedup data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model_file_path = os.path.join(dataset_path, 'model.pkl')
    if os.path.exists(model_file_path) and not train:
        logger.info("Loading the regression model...")
        regr = pickle.load(open(model_file_path, 'rb'))
    else:
        logger.info("Training the regression model...")
        regr = MLPRegressor(random_state=1,
                            max_iter=380 * 2,
                            batch_size=128,
                            hidden_layer_sizes=256).fit(X_train, y_train)
        with open(model_file_path, 'wb') as f:
            pickle.dump(regr, f)

    # Validate the model
    y_pred = regr.predict(X_test)
    pred_err = np.abs(y_test - y_pred).mean()
    logger.info(f"Average prediction error: {pred_err}")
    logger.info(f"Sklearn average score: {regr.score(X_test, y_test)}")

    return y_pred, y_test


def plot_validation_results(y_pred: np.ndarray,
                            y_test: np.ndarray,
                            num_sampled: int = 100) -> None:
    """Plot the validation results.

    Plot the validation results by comparing the predicted speedup 
        with the ground truth speedup and show the distribution of prediction errors.
    
    Parameters:
    ----------
    y_pred : np.ndarray
        Predicted speedup data.
    y_test : np.ndarray
        Ground truth speedup data.
    num_sampled : int
        Number of samples to be randomly selected from the test data.
    """
    fig, ax = plt.subplots()
    # Randomly sample 100 points from prediction results
    y_pred_test = random.sample(list(zip(y_pred, y_test)), num_sampled)
    gap = (np.array([pred - test for (pred, test) in y_pred_test])).reshape(-1)
    # x_labels = [i for i in range(1, num_sampled + 1)]
    # ax.bar(x_labels, gap)
    ax.hist(gap, bins=10, alpha=0.75, color=(42 / 256, 157 / 256, 142 / 256))

    # Compute the pearson correlation coefficient between the prediction and the test
    y_pred = np.array(y_pred).reshape(len(y_pred), 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    correlation_coefficient = r_regression(y_pred, y_test)
    print(
        f"correlation_coefficient between predict and ground_truth: {np.average(correlation_coefficient)}"
    )
    plt.xlabel('Randomly Sampled Test Cases')
    plt.ylabel('(pred - test) Speedup Histogram')
    plt.title('cBench Speedup Prediction Error Distribution')
    plt.savefig("regression-cost-model-validation.png")
    plt.savefig("regression-cost-model-validation.svg")


def main():
    normalized_features, normalized_speedup = preprocessing_features()
    y_pred, y_test = train_regression(normalized_features,
                                      normalized_speedup,
                                      train=False)
    plot_validation_results(y_pred, y_test)


if __name__ == '__main__':
    main()
