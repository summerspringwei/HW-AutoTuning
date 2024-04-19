"""Data structure for the program record."""

from typing import Set
from typing import List


''' Data examples for the ProgramRecord class
features = {
    'bitcnts.break-crit-edges.NumBroken': 3,
    'bitcnts.build-libcalls.NumNoCapture': 2,
    'bitcnts.build-libcalls.NumNoUndef': 5,
    ...
}

PMU = {
    'branch-misses': 5194894,
    'cache-misses': 118037,
    'cache-references': 174728391,
    'cpu-cycles': 922545757,
    'instructions': 1977440570,
    'cpu-clock': 938016,
    'L1-dcache-load-misses': 103202,
    'L1-dcache-loads': 113585127,
    'L1-dcache-store-misses': 61794,
    'L1-dcache-stores': 82486579,
    'branch-load-misses': 5679808,
    'branch-loads': 399485872
}

speedup over -O3: 0.2761904761904762
'''


def remove_module_name_from_features_keys(feature_name: str) -> str:
    """Remove the module name from the feature name.
    
    The example input name is bitcnts.break-crit-edges.NumBroken, 
    and the output name is break-crit-edges.NumBroken.

    Parameters:
    ----------
    feature_name : str
        The feature name.
    
    Returns:
    -------
    str
        The feature name without the module name.
    """
    return ".".join(feature_name.split(".")[1:])


def preprocessing_features_keys(features_keys: List[str]) -> List[str]:
    """Preprocess the features keys.
    
    Remove the module name from the feature keys.
    
    Parameters:
    ----------
    features_keys : List[str]
        The feature keys.
    
    Returns:
    -------
    List[str]
        The feature keys without the module name.
    """
    return [
        remove_module_name_from_features_keys(key) for key in features_keys
    ]


class ProgramRecord:
    features_dict = {}
    features_list = None
    perf_dict = {}
    perf_list = None
    speedup = 0
    real_time = 0

    def __init__(self, features, perf, speedup, real_time) -> None:
        self.features_dict = {
            remove_module_name_from_features_keys(key): value
            for key, value in features.items()
        }
        self.perf_dict = perf
        self.speedup = speedup
        self.real_time = real_time

    def get_features(self, features_keys: Set = None) -> List[int]:
        """Get the features list.
        
        Convert the features dictionary to a list.
        Note some features may not exist in the features dictionary, and we set them to 0.

        Parameters:
        ----------
        features_keys : Set
            The feature keys.
        
        Returns:
        -------
        List[int]
            The features list.
        """
        if self.features_list is None:
            self.features_list = [
                self.features_dict[key] if key in self.features_dict else 0
                for key in features_keys
            ]
        return self.features_list

    def get_perf(self):
        """Get the performance list."""
        return self.perf_list

    def get_last_level_cache_miss_rate(self) -> float:
        """Get the last level cache miss rate."""
        return self.perf_dict['cache-misses'] / self.perf_dict[
            'cache-references']

    def get_l1_cache_miss_rate(self)-> float:
        """Get the L1 cache miss rate."""
        return self.perf_dict['L1-dcache-load-misses'] / self.perf_dict[
            'L1-dcache-loads']

    def get_speedup(self) -> float:
        """Get the speedup."""
        return self.speedup
