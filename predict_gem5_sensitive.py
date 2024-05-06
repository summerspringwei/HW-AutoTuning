import os
import pickle
import numpy as np
from typing import List

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb


from utils import (create_cbench_cache_workload,
                   create_cbench_issue_width_workload, all_benchmarks,
                   dataset_path, get_dirname_by_params, extract,
                   get_latency_list)

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


def preprocessing_pmu(pmd_data: dict[str, int])-> List[int]:
    # return cache-miss-rate, branch-misses-rate, cpi, l1-dcache-load-miss-rate, l1-dcache-store-rate, branch-load-misses-rate
    cache_miss_rate = pmd_data['cache-misses'] / pmd_data['cache-references']
    branch_misses_rate = pmd_data['branch-misses'] / pmd_data['branch-loads']
    cpi = pmd_data['cpu-cycles'] / pmd_data['instructions']
    l1_dcache_load_miss_rate = pmd_data['L1-dcache-load-misses'] / pmd_data['L1-dcache-loads']
    l1_dcache_load_miss_rate = pmd_data['L1-dcache-store-misses'] / pmd_data['L1-dcache-stores']
    branch_load_miss_rate = pmd_data['branch-load-misses'] / pmd_data['branch-loads']
    pmu_feature = [cache_miss_rate, branch_misses_rate, cpi, l1_dcache_load_miss_rate, l1_dcache_load_miss_rate, branch_load_miss_rate]

    return pmu_feature


# for benchmark in all_benchmarks:
    # latency_list = get_latency_list(create_cbench_issue_width_workload(benchmark, "20"))
    # if latency_list is not None:
    #     round_list = [round(latency, 3) for latency in latency_list]
    #     print(benchmark, round_list, round(np.var(latency_list[:3]), 3), np.var(latency_list[4:]))

    # print("*"*50)
    # latency_list = get_latency_list(
    #     create_cbench_cache_workload(benchmark, "20"))
    # if latency_list is not None:
    #     round_list = [round(latency, 3) for latency in latency_list]
    #     # print(benchmark, round_list, round(np.var(latency_list[:3]), 3), np.var(latency_list[4:]))
    #     print(benchmark, round_list, round(np.var(latency_list), 2))


def load_pmu_O0_O3(benchmark):
    file_path = os.path.join(dataset_path, f"{benchmark}.pkl")
    with open(file_path, "rb") as file:
        (stats_and_pmu_list, speedup_list, time_list, y_O3, pmu_O0,
        pmu_O3) = pickle.load(file)
    pmu_O0_feature = preprocessing_pmu(pmu_O0)
    pmu_O3_feature = preprocessing_pmu(pmu_O3)
    return pmu_O0_feature + pmu_O3_feature


def preprocess_cbench_cache_sensitive(benchmark):
    latency_list = get_latency_list(
        create_cbench_cache_workload(benchmark, "20"))
    if latency_list is not None:
        return latency_list
    
    return None


def preprocess_cbench_issue_width_sensitive(benchmark):
    latency_list = get_latency_list(create_cbench_issue_width_workload(benchmark, "20"))
    if latency_list is not None:
        return latency_list

    return None


def prepare_cbench_cache_dataset(threshold=0.01, normalize=True):
    features, labels = [], []
    for benchmark in all_benchmarks:
        latency_list = preprocess_cbench_cache_sensitive(benchmark)
        # latency_list = preprocess_cbench_issue_width_sensitive(benchmark)
        pmu_feature = load_pmu_O0_O3(benchmark)
        if latency_list is not None:
            print(np.var(latency_list))
            features.append(pmu_feature)
            if np.var(latency_list) > threshold:
                labels.append(1)
            else:
                labels.append(0)
    if normalize:
        features = np.array(features)
        features = (features - features.mean(axis=0)) / features.std(axis=0)
    return features, labels


def xgboost_predict(X, y):
    # Split the data into training and testing sets
    X, y = sklearn.utils.shuffle(X, y, random_state=42)
    print(len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of XGBClassifier
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)

    # Fit the model on the training data
    xgb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = xgb_model.predict(X_test)

    print(f"Test set shape: {np.array(X_test).shape}")
    print(f"Groud truth: { y_test}")
    print(f"Predict: {y_pred}")
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", classification_rep)


if __name__ == "__main__":
    features, labels = prepare_cbench_cache_dataset()
    for f, l in zip(features, labels):
        print(f, l)
    xgboost_predict(features, labels)
    # print(feature_label)
