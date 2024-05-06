import os
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from program_record import preprocessing_features_keys
from utils import load_one_benchmark_dataset, get_logger, all_benchmarks, dataset_path
from mlp_model import State, SegmentSumMLPConfig, SegmentSumMLPTrainer

logger = get_logger(__name__)


def dump_training_data() -> None:
    """Dump All training data to pickle file.
    
    Load all benchmarks' data, get all features from all benchmarks, 
    normalize features, remove zero variance features, 
    and dump features and speedup to pickle file.

    The dumped data is a list of tuples, each tuple contains three elements:
    - benchmark name : str
    - features of the benchmark: List[np.ndarray] with the ndarray shape (1, 198)
    - costs of the benchmark: List[float]
    """
    # 1. Load all benchmarks' data
    record_benchmark_list, all_program_record_list = [], []
    all_features_keys, all_perf_keys = set(), set()
    for benchmark in all_benchmarks:
        file_path = f'{dataset_path}/{benchmark}.pkl'
        if not os.path.exists(file_path):
            continue
        program_record_list, features_keys, perf_keys = load_one_benchmark_dataset(
            file_path)
        all_program_record_list.extend(program_record_list)
        # Save corresponding benchmark name for each record
        record_benchmark_list.extend([
            benchmark,
        ] * len(program_record_list))
        features_keys = preprocessing_features_keys(features_keys)
        all_features_keys.update(features_keys)
        all_perf_keys.update(perf_keys)
    # Convert set to list to make the features ordered
    all_features_keys = sorted(list(all_features_keys))
    all_perf_keys = sorted(list(all_perf_keys))

    all_features, all_speedup = [], []
    # 2. Get all features from all benchmarks
    logger.debug(all_features_keys)
    for program_record in all_program_record_list:
        all_features.append(program_record.get_features(all_features_keys))
        all_speedup.append(
            1.0 /
            program_record.get_speedup())  # Get cost rather than speed up
    # 3. Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(all_features)
    # 4. Remove zero variance features
    sel = VarianceThreshold()
    normalized_features = sel.fit_transform(normalized_features)
    logger.info(f"Feature shape for training: {normalized_features.shape}")
    normalized_speedup = np.array(all_speedup, dtype=np.float32)

    # 5. Dump features and speedup to pickle file
    benchmark_features_dict = {benchmark: [] for benchmark in all_benchmarks}
    benchmark_labels_dict = {benchmark: [] for benchmark in all_benchmarks}
    for benchmark, feature, label in zip(record_benchmark_list,
                                         normalized_features,
                                         normalized_speedup):
        benchmark_features_dict[benchmark].append(
            np.expand_dims(np.array(feature, dtype=np.float32), axis=0))
        benchmark_labels_dict[benchmark].append(label)
    cached_data = []
    for benchmark, features in benchmark_features_dict.items():
        cached_data.append(
            (benchmark, features, benchmark_labels_dict[benchmark]))
    with open(os.path.join(dataset_path, 'cached_data.pkl'), 'wb') as f:
        pickle.dump(cached_data, f)


def train_ranking_cost_model():
    """Train the ranking cost model using the dumped training data."""
    model_config = SegmentSumMLPConfig(input_dim=198)
    state = State(model_config)
    state.load(dataset_path)
    trainer = SegmentSumMLPTrainer(state=state)
    trainer.train_full()


def main():
    dump_training_data()
    train_ranking_cost_model()


if __name__ == '__main__':
    main()
