import os
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from program_record import preprocessing_features_keys, ProgramRecord
from utils import load_one_benchmark_dataset, get_logger, all_benchmarks, dataset_path

logger = get_logger(__name__)


def load_all_dataset() -> Tuple[List[ProgramRecord], List[str], List[str]]:
    """Load all datasets from all benchmarks.

    Returns:
    -------
    all_program_record_list : List[ProgramRecord]
        List of program records.
    all_features_keys : List[str]
        List of feature keys.
    all_perf_keys : List[str]
        List of perf tool keys.
    """
    all_program_record_list = []
    all_features_keys, all_perf_keys = set(), set()
    for benchmark in all_benchmarks:
        file_path = f'{dataset_path}/{benchmark}.pkl'
        if not os.path.exists(file_path):
            logger.warning(f'{file_path} does not exist')
            continue
        program_record_list, features_keys, perf_keys = load_one_benchmark_dataset(file_path)
        all_program_record_list.extend(program_record_list)
        all_features_keys.update(preprocessing_features_keys(features_keys))
        all_perf_keys.update(perf_keys)
    # Convert set to list to make the features ordered
    all_features_keys = sorted(list(all_features_keys))
    all_perf_keys = sorted(list(all_perf_keys))
    return all_program_record_list, all_features_keys, all_perf_keys


def preprocessing_features() -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess all features and speedup data.
    
    Returns:
    -------
    normalized_features : np.ndarray
        Normalized features.
    all_speedup : np.ndarray
        Speedup data.
    """
    all_program_record_list, all_features_keys, _ = load_all_dataset()
    all_features, all_speedup = [], []
    # 1. Get all features from all benchmarks
    for program_record in all_program_record_list:
        all_features.append(program_record.get_features(all_features_keys))
        all_speedup.append([
            program_record.get_speedup(),
        ])  # We need 2-D array
    # 2. Normalize features
    scaler = StandardScaler()
    logger.debug(f"Example features: {np.array(all_features)[:5, :30]}")
    normalized_features = scaler.fit_transform(all_features)
    # 3. Remove zero variance features
    sel = VarianceThreshold()
    normalized_features = sel.fit_transform(normalized_features)
    logger.info(f"All dataset shape: {normalized_features.shape}")
    all_speedup = np.array(all_speedup)
    return normalized_features, all_speedup
