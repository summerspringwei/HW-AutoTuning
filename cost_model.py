
from typing import List
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from program_record import ProgramRecord, preprocessing_features_keys
from utils import load_dataset, all_benchmarks, dataset_path


def load_all_dataset():
    all_program_record_list = []
    all_features_keys = set()
    all_perf_keys = set()
    for benchmark in all_benchmarks:
        program_record_list, features_keys, perf_keys = load_dataset(f'{dataset_path}/{benchmark}.pkl')
        all_program_record_list.extend(program_record_list)
        features_keys = preprocessing_features_keys(features_keys)
        all_features_keys.update(features_keys)
        all_perf_keys.update(perf_keys)
    # Convert set to list to make the features ordered
    all_features_keys = sorted(list(all_features_keys))
    all_perf_keys = sorted(list(all_perf_keys))
    return all_program_record_list, all_features_keys, all_perf_keys


def preprocessing_features():
    all_program_record_list, all_features_keys, all_perf_keys = load_all_dataset()
    all_features, all_speedup = [], []
    # 1. Get all features from all benchmarks
    print(all_features_keys)
    for program_record in all_program_record_list:
        all_features.append(program_record.get_features(all_features_keys))
        all_speedup.append([program_record.get_speedup(),]) # We need 2-D array
    # 2. Normalize features
    scaler = StandardScaler()
    print(np.array(all_features)[:5, :30])
    normalized_features = scaler.fit_transform(all_features)
    # 3. Remove zero variance features
    sel = VarianceThreshold()
    normalized_features = sel.fit_transform(normalized_features)
    print(normalized_features.shape)
    # normalized_speedup = scaler.fit_transform(all_speedup)
    normalized_speedup = np.array(all_speedup)
    return normalized_features, normalized_speedup


def train_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=380*2, batch_size=128, hidden_layer_sizes=256).fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    pickle.save(regr.get_params(True), 'model.pkl')
    print(np.abs(y_test-y_pred).mean())
    print(regr.score(X_test, y_test))


def main():
    normalized_features, normalized_speedup = preprocessing_features()
    print(normalized_features[:5, :30])
    print(normalized_speedup[:5, :30])
    train_regression(normalized_features, normalized_speedup)


if __name__ == '__main__':
    main()
