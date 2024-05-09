import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

from utils import (create_cbench_cache_workload,
                   create_cbench_issue_width_workload, get_latency_list,
                   get_logger, all_benchmarks, dataset_path)

logger = get_logger("predict_gem5_sensitive")
""" Example of PMU data

example_perf_O0 = {
    'branch-misses': 11557400,
    'cache-misses': 156371,
    'cache-references': 1888149533,
    'cpu-cycles': 2978395204,
    'instructions': 3868967581,
    'cpu-clock': 357088,
    'L1-dcache-load-misses': 147130,
    'L1-dcache-loads': 1907766116,
    'L1-dcache-store-misses': 69366,
    'L1-dcache-stores': 1905722931,
    'branch-load-misses': 11624650,
    'branch-loads': 550346957
}

exapmple_perf_O3 = {
    'branch-misses': 1303412,
    'cache-misses': 127044,
    'cache-references': 119656312,
    'cpu-cycles': 538193598,
    'instructions': 1059175935,
    'cpu-clock': 934784,
    'L1-dcache-load-misses': 113539,
    'L1-dcache-loads': 84232969,
    'L1-dcache-store-misses': 81426,
    'L1-dcache-stores': 125580617,
    'branch-load-misses': 932571,
    'branch-loads': 185751104
}
"""


def preprocessing_pmu(pmd_data: Dict[str, int]) -> List[float]:
    """Preprocessing the pmu data, return a list of features

    Get raw pmu data and return the pmu features like cache miss rate, branch miss rate, cpi, etc.

    Parameters:
    ----------
    pmd_data : Dict[str, int]
        The pmu data.
    
    Return:
    -------
    pmu_feature : List[int]
        List of pmu features.
    """
    # Check the integrity of the pmu data
    perf_keys = [
        'branch-misses', 'cache-misses', 'cache-references', 'cpu-cycles',
        'instructions', 'cpu-clock', 'L1-dcache-load-misses',
        'L1-dcache-loads', 'L1-dcache-store-misses', 'L1-dcache-stores',
        'branch-load-misses', 'branch-loads'
    ]
    for k in perf_keys:
        if k not in pmd_data:
            return None

    cache_miss_rate = pmd_data['cache-misses'] / pmd_data['cache-references']
    branch_misses_rate = pmd_data['branch-misses'] / pmd_data['branch-loads']
    cpi = pmd_data['cpu-cycles'] / pmd_data['instructions']
    l1_dcache_load_miss_rate = pmd_data['L1-dcache-load-misses'] / pmd_data[
        'L1-dcache-loads']
    l1_dcache_load_miss_rate = pmd_data['L1-dcache-store-misses'] / pmd_data[
        'L1-dcache-stores']
    branch_load_miss_rate = pmd_data['branch-load-misses'] / pmd_data[
        'branch-loads']
    pmu_feature = [
        cache_miss_rate, branch_misses_rate, cpi, l1_dcache_load_miss_rate,
        l1_dcache_load_miss_rate, branch_load_miss_rate
    ]

    return pmu_feature


def load_pmu_O0_O3(benchmark: str) -> List[float]:
    """Load the pmu data for O0 and O3 optimization level and return the pmu features
    
    Parameters:
    ----------
    benchmark : str
        The benchmark name.
    
    Returns:
    -------
    pmu_O0_feature + pmu_O3_feature : List[float]
        List of pmu features for O0 and O3 optimization level.
    """
    file_path = os.path.join(dataset_path, f"{benchmark}.pkl")
    with open(file_path, "rb") as file:
        (stats_and_pmu_list, speedup_list, time_list, y_O3, pmu_O0,
         pmu_O3) = pickle.load(file)
    pmu_O0_feature = preprocessing_pmu(pmu_O0)
    pmu_O3_feature = preprocessing_pmu(pmu_O3)
    if pmu_O0_feature is None or pmu_O3_feature is None:
        return None
    return pmu_O0_feature + pmu_O3_feature


def preprocess_cbench_cache_sensitive(benchmark: str,
                                      dataset_name: str = "20"
                                      ) -> Optional[List[float]]:
    """Preprocess the cbench cache sensitive data and return the latency list
    
    Parameters:
    ----------
    benchmark : str
        The benchmark name in cBench.
    dataset_name : str
        The dataset name in cBench.
    
    Returns:
    -------
    latency_list : Optional[List[float]]
        The latency list.
    """
    latency_list = get_latency_list(
        create_cbench_cache_workload(benchmark, dataset_name))
    if latency_list is not None:
        return latency_list

    return None


def preprocess_cbench_issue_width_sensitive(benchmark,
                                            dataset_name: str = "20"
                                            ) -> Optional[List[float]]:
    """Preprocess the cbench issue width sensitive data and return the latency list
    
    Parameters:
    ----------
    benchmark : str
        The benchmark name in cBench.
    dataset_name : str
        The dataset name in cBench.
    
    Returns:
    -------
    latency_list : Optional[List[float]]
        The latency list.
    """
    latency_list = get_latency_list(
        create_cbench_issue_width_workload(benchmark, dataset_name))
    if latency_list is not None:
        return latency_list

    return None


def prepare_cbench_cache_dataset(
        threshold: float = 0.01,
        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare the cbench cache sensitive dataset for training"""
    features, labels = [], []
    for benchmark in all_benchmarks:
        latency_list = preprocess_cbench_cache_sensitive(benchmark)
        # latency_list = preprocess_cbench_issue_width_sensitive(benchmark)
        pmu_feature = load_pmu_O0_O3(benchmark)
        if latency_list is not None:
            logger.info(f"variance of latency list: {np.var(latency_list)}")
            features.append(pmu_feature)
            if np.var(latency_list) > threshold:
                labels.append(1)
            else:
                labels.append(0)
    if normalize:
        features = np.array(features)
        features = (features - features.mean(axis=0)) / features.std(axis=0)
    return features, labels


def xgboost_predict(X: np.ndarray, y: np.ndarray, test_size: float = 42):
    """Train and Predict the cache sensitive using XGBoost classifier
    
    Parameters:
    ----------
    X : np.ndarray
        The features.
    y : np.ndarray
        The labels.
    test_size : float
        The size of the test set.
    
    """
    # Split the data into training and testing sets
    X, y = sklearn.utils.shuffle(X, y, random_state=42)
    logger.info(f"Get {len(X)} training records")
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=classification_rep,
                                                        random_state=42)
    # Create an instance of XGBClassifier
    xgb_model = xgb.XGBClassifier(n_estimators=100,
                                  max_depth=3,
                                  learning_rate=0.1)

    # Fit the model on the training data
    xgb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = xgb_model.predict(X_test)

    logger.info(f"Test set shape: {np.array(X_test).shape}")
    logger.info(f"Groud truth: { y_test}")
    logger.info(f"Predict: {y_pred}")
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    logger.info(f"<<< Accuracy: {accuracy} >>>")
    logger.info(f"Confusion Matrix: {confusion}\n")
    logger.info(f"Classification Report: {classification_rep}\n")


def main():
    features, labels = prepare_cbench_cache_dataset()
    # for f, l in zip(features, labels):
    #     logger.info(f, l)
    xgboost_predict(features, labels)


if __name__ == "__main__":
    main()
