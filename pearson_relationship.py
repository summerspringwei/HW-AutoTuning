import pickle
import numpy as np
from enum import Enum
from typing import List, Mapping, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import r_regression
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (18, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-small',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

from program_record import ProgramRecord
from utils import load_dataset, all_benchmarks, dataset_path

'''
From https://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf
for the absolute value of r
00-.19 “very weak”
.20-.39 “weak”
.40-.59 “moderate”
.60-.79 “strong”
.80-1.0 “very strong”
'''


class LabelType(Enum):
    SPEEDUP = 0
    CACHE_MISS = 1


def preprocessing_features(record_list: List[ProgramRecord], labels=LabelType.SPEEDUP):
    '''Proprocessing features and normalize them
    
        Return the label based on type
    '''
    # Normalize features
    features_set = set()
    for record in record_list:
        features_set.update(record.features_dict.keys())
    features_keys = sorted(list(features_set))
    features = [record.get_features(features_keys) for record in record_list]
    if labels is LabelType.CACHE_MISS :
        labels = [record.get_last_level_cache_miss_rate() for record in record_list]
    else: # By default, we return the speedup
        labels = [record.get_speedup() for record in record_list]
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    return normalized_features, labels
    

def compute_pearson_correlation(features, labels, feature_keys, threshold=0.2):
    correlation_coefficient = r_regression(features, labels)
    # print(f"correlation_coefficient:{correlation_coefficient}")
    # Get the most important features
    important_features = []
    features_list = [f for f in feature_keys]
    for i, coef in enumerate(correlation_coefficient):
        if abs(coef) > threshold:
            important_features.append((features_list[i], coef))
    # Sort features by importance
    important_features.sort(key=lambda x: abs(x[1]), reverse=True)
    return important_features



def compute_pearson_correlation_for_benchmark(benchmark, threshold=0.2, labels=LabelType.SPEEDUP):
    record_list, features_keys, perf_keys = load_dataset(f'{dataset_path}/{benchmark}.pkl')
    features, labels = preprocessing_features(record_list, labels=labels)
    important_features = compute_pearson_correlation(features, labels, features_keys, threshold)
    return important_features


def draw_heat_map_for_benchmark(benchmark_important_features: Mapping[str, List[Tuple[str, float]]], labels=LabelType.SPEEDUP):
    '''Draw the heat map for the important features

    '''
    # 0. Convert list of features to a map of features
    benchmark_important_features_dict = {}
    # 1. Get all benchmarks
    benchmark_list = list(benchmark_important_features.keys())
    features_set = set()
    for benchmark in benchmark_list:
        # strip file name from features
        benchmark_important_features_dict[benchmark] = {}
        for f, c in benchmark_important_features[benchmark]:
            nf = ".".join(f.split('.')[1:])
            benchmark_important_features_dict[benchmark][nf] = c
            features_set.add(nf)
    # 2. Create the matrix
    matrix = np.zeros((len(benchmark_list), len(features_set)))
    features_list = list(features_set)
    for i, benchmark in enumerate(benchmark_list):
        for j, feature in enumerate(features_list):
            matrix[i, j] = benchmark_important_features_dict[benchmark].get(feature, 0)
    # 3. Draw the heat map
    ax = sns.heatmap(matrix, annot=True, fmt=".1f", xticklabels=features_list, yticklabels=benchmark_list)
    label_str = 'Speedup' if labels is LabelType.SPEEDUP else 'Cache-Miss-Rate'
    ax.set_title(f'Pearson Correlations between features and {label_str} for cBench')
    plt.tight_layout()
    plt.savefig(f'heatmap-{label_str}.png')
    plt.savefig(f'heatmap-{label_str}.svg')


def main():
    benchmark_important_features = {}
    labels = LabelType.CACHE_MISS
    for benchmark in all_benchmarks:
        important_features = compute_pearson_correlation_for_benchmark(benchmark, labels=labels, threshold=0.2)
        benchmark_important_features[benchmark] = important_features
        print(f"Important features for {benchmark}:")
        for feature, coef in important_features:
            fmt = "%.2f" % coef
            print(f"{feature:<50}:{fmt}")
        print("="*20)
    draw_heat_map_for_benchmark(benchmark_important_features, labels=labels)


if __name__=="__main__":
    # compute_pearson_correlation_for_benchmark('automotive_bitcount')
    main()
