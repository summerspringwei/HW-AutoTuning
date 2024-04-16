


''' Data examples for the ProgramRecord class
features: {'bitcnts.break-crit-edges.NumBroken': 3, 'bitcnts.build-libcalls.NumNoCapture': 2, 'bitcnts.build-libcalls.NumNoUndef': 5, 'bitcnts.build-libcalls.NumNoUnwind': 1, 'bitcnts.early-cse.NumCSE': 26, 'bitcnts.early-cse.NumCSELoad': 2, 'bitcnts.early-cse.NumSimplify': 9, 'bitcnts.elim-avail-extern.NumRemovals': 1, 'bitcnts.function-attrs.NumMemoryAttr': 1, 'bitcnts.function-attrs.NumNoCapture': 1, 'bitcnts.function-attrs.NumNoRecurse': 1, 'bitcnts.function-attrs.NumNoSync': 1, 'bitcnts.function-attrs.NumReadOnlyArg': 1, 'bitcnts.globalopt.NumDeleted': 5, 'bitcnts.globalopt.NumFastCallFns': 1, 'bitcnts.globalopt.NumInternalFunc': 1, 'bitcnts.globalopt.NumUnnamed': 16, 'bitcnts.gvn.NumGVNInstr': 8, 'bitcnts.gvn.NumGVNSimpl': 8, 'bitcnts.indvars.NumLFTR': 7, 'bitcnts.instcombine.NumCombined': 60, 'bitcnts.instcombine.NumDeadInst': 2, 'bitcnts.instcombine.NumSimplified': 1, 'bitcnts.instcombine.NumWorklistIterations': 4, 'bitcnts.jump-threading.NumThreads': 1, 'bitcnts.lcssa.NumLCSSA': 64, 'bitcnts.loop-rotate.NumRotated': 3, 'bitcnts.loop-unroll.NumCompletelyUnrolled': 1, 'bitcnts.loop-unroll.NumUnrolled': 1, 'bitcnts.mem2reg.NumPHIInsert': 8, 'bitcnts.mem2reg.NumSingleStore': 4, 'bitcnts.reassociate.NumChanged': 15, 'bitcnts.simplifycfg.NumFoldBranchToCommonDest': 1, 'bitcnts.simplifycfg.NumSimpl': 45, 'bitcnts.sroa.NumAllocaPartitions': 11, 'bitcnts.sroa.NumDeleted': 55, 'bitcnts.sroa.NumPromoted': 11}
PMU: {'branch-misses': 5194894, 'cache-misses': 118037, 'cache-references': 174728391, 'cpu-cycles': 922545757, 'instructions': 1977440570, 'cpu-clock': 938016, 'L1-dcache-load-misses': 103202, 'L1-dcache-loads': 113585127, 'L1-dcache-store-misses': 61794, 'L1-dcache-stores': 82486579, 'branch-load-misses': 5679808, 'branch-loads': 399485872}
speedup over -O3: 0.2761904761904762
'''
from typing import Set
from typing import List

def remove_module_name_from_features_keys(feature_name: str):
    return ".".join(feature_name.split(".")[1:])


def preprocessing_features_keys(features_keys: List[str]):
    # Note the format of feature keys is modulename.pass.featurename, e.g., bitcnts.break-crit-edges.NumBroken
    # we need to strip the modulename, and keep the pass.featurename, e.g., break-crit-edges.NumBroken
    return [remove_module_name_from_features_keys(key) for key in features_keys]


class ProgramRecord:
    features_dict = {}
    features_list = None
    perf_dict = {}
    perf_list = None
    speedup = 0
    real_time = 0
    def __init__(self, features, perf, speedup, real_time) -> None:
        self.features_dict = {remove_module_name_from_features_keys(key): value for key, value in features.items()}
        self.perf_dict = perf
        self.speedup = speedup
        self.real_time = real_time
    
    def get_features(self, features_keys: Set = None):
        if self.features_list is None:
            self.features_list = [self.features_dict[key] if key in self.features_dict else 0 for key in features_keys]
        return self.features_list

    def get_perf(self):
        return self.perf_list
    
    def get_last_level_cache_miss_rate(self):
        return self.perf_dict['cache-misses'] / self.perf_dict['cache-references']
    
    def get_l1_cache_miss_rate(self):
        return self.perf_dict['L1-dcache-load-misses'] / self.perf_dict['L1-dcache-loads']
    
    def get_speedup(self):
        return self.speedup

    def get_perf(self, metric: str):
        if metric in self.perf_dict:
            return self.perf_dict[metric]
        else:
            raise ValueError(f'No such metric {metric} in the perf_dict')
