
from typing import List
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import r_regression

from program_record import ProgramRecord, preprocessing_features_keys
from utils import load_dataset, all_benchmarks, dataset_path


def load_all_dataset():
    all_program_record_list = []
    all_features_keys = set()
    all_perf_keys = set()
    for benchmark in all_benchmarks:
        file_path = f'{dataset_path}/{benchmark}.pkl'
        if not os.path.exists(file_path):
            continue
        program_record_list, features_keys, perf_keys = load_dataset(file_path)
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


def train_regression(X, y, train=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    if os.path.exists('model.pkl') and not train:
        regr= pickle.load(open('model.pkl', 'rb'))
    else:
        regr = MLPRegressor(random_state=1, max_iter=380*2, batch_size=128, hidden_layer_sizes=256).fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    with open('model.pkl','wb') as f:
        pickle.dump(regr, f)
    
    print(np.abs(y_test-y_pred).mean())
    print(regr.score(X_test, y_test))

    return y_pred, y_test


def draw_pred_results(y_pred, y_test, num_sampled=100):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    x_labels = [i for i in range(1, num_sampled+1)]
    # Randomly sample 100 points from prediction results
    import random
    
    y_pred_test = random.sample(list(zip(y_pred, y_test)), num_sampled)
    print("Abs:")
    gap = (np.array([pred-test for (pred, test) in y_pred_test])).reshape(-1)
    ax.bar(x_labels, gap)

    # Compute the pearson correlation coefficient between the prediction and the test
    y_pred = np.array(y_pred).reshape(len(y_pred), 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    correlation_coefficient = r_regression(y_pred, y_test)
    print(f"correlation_coefficient:{np.average(correlation_coefficient)}")
    plt.xlabel('Randomly Sampled Test Points')
    plt.ylabel('(pred - test) Speedup')
    plt.title('cBench Speedup Prediction Error Distribution')
    plt.savefig("pred-test.png")
    plt.savefig("pred-test.svg")


def cost_model_repression():
    normalized_features, normalized_speedup = preprocessing_features()
    print(normalized_features[:5, :30])
    print(normalized_speedup[:5, :30])
    y_pred, y_test = train_regression(normalized_features, normalized_speedup, train=False)
    draw_pred_results(y_pred, y_test)


# def dump_features_keys():
#     all_program_record_list, all_features_keys, all_perf_keys = load_all_dataset()
#     all_features, all_speedup = [], []
#     # 1. Get all features from all benchmarks
#     print(all_features_keys)
#     for program_record in all_program_record_list:
#         all_features.append(program_record.get_features(all_features_keys))
#         all_speedup.append([program_record.get_speedup(),]) # We need 2-D array
#     # save to numpy file
#     with open(os.path.join(dataset_path, 'cached_data.npy'), 'wb') as f:
#         pickle.dump([("cbench", np.array(all_features), np.array(all_speedup)),], f)



def dump_features_keys():
    record_benchmark_list = []
    all_program_record_list = []
    all_features_keys = set()
    all_perf_keys = set()
    for benchmark in all_benchmarks:
        file_path = f'{dataset_path}/{benchmark}.pkl'
        if not os.path.exists(file_path):
            continue
        program_record_list, features_keys, perf_keys = load_dataset(file_path)
        all_program_record_list.extend(program_record_list)
        record_benchmark_list.extend([benchmark,] * len(program_record_list))
        features_keys = preprocessing_features_keys(features_keys)
        all_features_keys.update(features_keys)
        all_perf_keys.update(perf_keys)
    
    # Convert set to list to make the features ordered
    all_features_keys = sorted(list(all_features_keys))
    all_perf_keys = sorted(list(all_perf_keys))

    all_features, all_speedup = [], []
    # 1. Get all features from all benchmarks
    print(all_features_keys)
    for program_record in all_program_record_list:
        all_features.append(program_record.get_features(all_features_keys))
        all_speedup.append(1.0 / program_record.get_speedup()) # Get cost rather than speed up
    # 2. Normalize features
    scaler = StandardScaler()
    print(np.array(all_features)[:5, :30])
    normalized_features = scaler.fit_transform(all_features)
    # 3. Remove zero variance features
    sel = VarianceThreshold()
    normalized_features = sel.fit_transform(normalized_features)
    print(normalized_features.shape)
    # normalized_speedup = scaler.fit_transform(all_speedup)
    normalized_speedup = np.array(all_speedup, dtype=np.float32)

    # Dump features and speedup to numpy file
    benchmark_features_dict = {benchmark:[] for benchmark in all_benchmarks}
    benchmark_labels_dict = {benchmark:[] for benchmark in all_benchmarks}
    for benchmark, feature, label in zip(record_benchmark_list, normalized_features, normalized_speedup):
        benchmark_features_dict[benchmark].append(np.expand_dims(np.array(feature, dtype=np.float32), axis=0))
        benchmark_labels_dict[benchmark].append(label)

    cached_data = []
    for benchmark, features in benchmark_features_dict.items():
        cached_data.append((benchmark, features, benchmark_labels_dict[benchmark]))
    with open(os.path.join(dataset_path, 'cached_data.pkl'), 'wb') as f:
        pickle.dump(cached_data, f)


def cost_model_rank():
    from mlp_model import State, SegmentSumMLPConfig, SegmentSumMLPTrainer, TrainerConfig
    model_config = SegmentSumMLPConfig(input_dim=198)
    state = State(model_config)
    state.load(dataset_path)
    trainer = SegmentSumMLPTrainer(state=state, )
    trainer.train_full()


if __name__ == '__main__':
    # main()
    dump_features_keys()
    cost_model_rank()