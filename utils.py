import os
import pickle
import logging
import sys
import pathlib
from logging import Logger
import numpy as np
from typing import List, Tuple, Set

from program_record import ProgramRecord

all_benchmarks = [
    'automotive_bitcount', 'automotive_qsort1', 'automotive_susan_c',
    'automotive_susan_e', 'automotive_susan_s', 'bzip2d', 'bzip2e',
    'consumer_jpeg_c', 'consumer_jpeg_d', 'consumer_lame', 'consumer_tiff2bw',
    'consumer_tiff2rgba', 'consumer_tiffdither', 'consumer_tiffmedian',
    'network_patricia', 'office_stringsearch1', 'security_blowfish_d',
    'security_blowfish_e', 'security_rijndael_d', 'security_rijndael_e',
    'security_sha', 'telecom_CRC32', 'telecom_adpcm_c', 'telecom_adpcm_d',
    'telecom_gsm'
]

this_dir_path = pathlib.Path(__file__).parent.resolve()
dataset_path = os.path.join(this_dir_path, 'data_perf_features')

cbench_dir = "/home/xiachunwei/Projects/ArmCBench/cBench_V1.1/"


def get_logger(name: str, log_level=logging.INFO) -> Logger:
    """Create or get a logger by its name. This is essentially a wrapper of python's native logger.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logger : Logger
        The logger instance.
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(levelname)s - %(asctime)s - %(filename)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger


logger = get_logger("utils")


def load_dataset(
        file_path: str) -> Tuple[List[ProgramRecord], Set[str], Set[str]]:
    """Load the dataset from the dataset file (pickle format).

    Load features, performance metrics, speedup, and real time from the dataset file.
    We need to return feature keys as not all the features are used for one benchmark.
    Parameters
    ----------
    file_path : str
        The path of the dataset file.
    Returns
    -------
    Tuple[List[ProgramRecord], Set[str], Set[str]]
        A tuple of three elements: the list of ProgramRecord instances, the set of feature keys, and the set of performance keys.
    """
    with open(file_path, 'rb') as file:
        (stats_and_pmu_list, speedup_list, time_list, y_O3, pmu_O0,
         pmu_O3) = pickle.load(file)
        # Get all the keys
        features_keys, perf_keys = set(), set()
        for [stats, pmu] in stats_and_pmu_list:
            features_keys.update(stats.keys())
            perf_keys.update(pmu.keys())
        record_list = [
            ProgramRecord(state_and_pmu[0], state_and_pmu[1], speedup,
                          time) for state_and_pmu, speedup, time in zip(
                              stats_and_pmu_list, speedup_list, time_list)
        ]
    return record_list, features_keys, perf_keys


def get_cbench_cmd_args(benchmark_name, dataset_name: str) -> str:
    """Get the command line arguments for the cBench benchmark.
    
    Parameters
    ----------
    benchmark_name : str
        The name of the benchmark. e.g., automotive_bitcount
    dataset_name : str
        The name of the dataset. select from 1 to 20.
    
    Returns
    -------
    str
        The command line arguments for the corresponding cBench benchmark.
    """
    if dataset_name not in [str(i) for i in range(1, 21)]:  # 1 to 20
        raise ValueError(f"Invalid dataset_name: {dataset_name}")

    benchmark_dataset_map = {
        "network_dijkstra":
        f"../../network_dijkstra_data/{dataset_name}.dat",  # OK
        "automotive_bitcount":
        "1125000",  # OK
        "automotive_qsort1":
        f"../../automotive_qsort_data/{dataset_name}.dat",  # OK
        "automotive_susan_c":
        f"../../automotive_susan_data/{dataset_name}.pgm output_large.corners.pgm -c",  # OK
        "security_sha":
        f"../../office_data/{dataset_name}.txt",
        "automotive_susan_e":
        f"../../automotive_susan_data/{dataset_name}.pgm output_large.edges.pgm -e",
        "automotive_susan_s":
        f"../../automotive_susan_data/{dataset_name}.pgm output_large.smoothing.pgm -s",
        "bzip2d":
        f"-d -k -f -c ../../bzip2_data/{dataset_name}.bz2",  # OK
        "bzip2e":
        f"-z -k -f -c ../../automotive_qsort_data/{dataset_name}.dat",
        "consumer_jpeg_c":
        f"-dct int -progressive -opt -outfile output_large_encode.jpeg ../../consumer_jpeg_data/{dataset_name}.ppm",
        "consumer_jpeg_d":
        f"-dct int -ppm -outfile output_large_decode.ppm ../../consumer_jpeg_data/{dataset_name}.jpg",
        "consumer_lame":
        f"../../consumer_data/{dataset_name}.wav output_large.mp3",
        "consumer_mad":
        f"--time=100000 --output=wave:output.wav ../../consumer_data/{dataset_name}.mp3",  # pass
        "consumer_tiff2bw":
        f"../../consumer_tiff_data/{dataset_name}.tif output_largebw.tif",  # OK
        "consumer_tiff2rgba":
        f"-c none ../../consumer_tiff_data/{dataset_name}.tif output_largergba.tif",
        "consumer_tiffdither":
        f"-c g4 ../../consumer_tiff_data/{dataset_name}.bw.tif output_largedither.tif",
        "consumer_tiffmedian":
        f"../../consumer_tiff_data/{dataset_name}.nocomp.tif output_largemedian.tif",
        "office_ghostscript":
        f"-sDEVICE=ppm -dNOPAUSE -q -sOutputFile=output.ppm -- ../../office_data/{dataset_name}.ps",  # pass
        "telecom_adpcm_c":
        f"< ../../telecom_data/{dataset_name}.pcm > output_large.adpcm",
        "network_patricia":
        f"../../network_patricia_data/{dataset_name}.udp",
        "security_blowfish_d":
        f"d ../../office_data/{dataset_name}.benc output_large.txt 1234567890abcdeffedcba0987654321",
        "security_blowfish_e":
        f"e ../../office_data/{dataset_name}.txt output_large.enc 1234567890abcdeffedcba0987654321",
        "security_rijndael_d":
        f"../../office_data/{dataset_name}.enc output_large.dec d 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321",
        "security_rijndael_e":
        f"../../office_data/{dataset_name}.txt output_large.enc e 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321",
        "telecom_adpcm_d":
        f"< ../../telecom_data/{dataset_name}.adpcm > output_large.pcm",
        "telecom_CRC32":
        f"../../telecom_data/{dataset_name}.pcm > output_large.txt",
        "telecom_gsm":
        f"-fps -c ../../telecom_gsm_data/{dataset_name}.au",
        "office_stringsearch1":
        f"../../office_data/{dataset_name}.txt ../../office_data/{dataset_name}.s.txt output.txt"
    }
    if benchmark_name not in benchmark_dataset_map.keys():
        raise ValueError(f"Invalid benchmark_name: {benchmark_name}")
    return benchmark_dataset_map[benchmark_name]


def create_cbench_cache_workload(benchmark_name, dataset_name, unit="kB"):
    args_list = []
    issue_width = 8
    l2_size, l1_d_size, l1_i_size = 256, 4, 6
    for i in range(7):
        if l2_size >= 1024 and unit == "MB":
            args_list.append(
                (f"{l2_size // 1024}MB", f"{l1_d_size}kB", f"{l1_i_size}kB", 8,
                 4, 6, benchmark_name, str(dataset_name), issue_width))
        else:
            args_list.append(
                (f"{l2_size}kB", f"{l1_d_size}kB", f"{l1_i_size}kB", 8, 4, 6,
                 benchmark_name, str(dataset_name), issue_width))
        l2_size, l1_d_size, l1_i_size = l2_size * 2, l1_d_size * 2, l1_i_size * 2

    return args_list


def create_cbench_cache_workload(benchmark_name, dataset_name) -> List[dict]:
    """Create a list of arguments for the cBench benchmark with different cache sizes for gem5.
    Parameters
    ----------
    benchmark_name : str
        The name of the benchmark.
    dataset_name : str
        The name of the dataset.
    
    Returns
    -------
    List[dict]
        A list of dictionaries, each dictionary contains the arguments for the cBench benchmark.
    """
    args_list = []
    issue_width = 8
    l2_size, l1_d_size, l1_i_size = 256, 4, 6
    for _ in range(7):
        args_list.append({
            "l2_size": l2_size,
            "l1_d_size": l1_d_size,
            "l1_i_size": l1_i_size,
            "l2_assoc": 8,
            "l1_d_assoc": 4,
            "l1_i_assoc": 6,
            "benchmark_name": benchmark_name,
            "dataset_name": str(dataset_name),
            "issue_width": issue_width
        })
        l2_size, l1_d_size, l1_i_size = l2_size * 2, l1_d_size * 2, l1_i_size * 2

    return args_list


def create_cbench_issue_width_workload(benchmark_name, dataset_name):
    args_list = []
    l2_size, l1_d_size, l1_i_size = 1024 * 4, 64, 96

    for i in range(7):
        issue_width = i + 1
        args_list.append(
            (f"{l2_size}kB", f"{l1_d_size}kB", f"{l1_i_size}kB", 8, 4, 6,
             benchmark_name, str(dataset_name), issue_width))

    return args_list


def create_cbench_issue_width_workload(benchmark_name,
                                       dataset_name) -> List[dict]:
    """Create a list of arguments for the cBench benchmark with different issue widths for gem5.
    
    Parameters
    ----------
    benchmark_name : str
        The name of the benchmark.
    dataset_name : str
        The name of the dataset.
    
    Returns
    -------
    List[dict]
        A list of dictionaries, each dictionary contains the arguments for the cBench benchmark.
    """
    args_list = []
    l2_size, l1_d_size, l1_i_size = 1024 * 4, 64, 96

    for i in range(7):
        issue_width = i + 1
        args_list.append({
            "l2_size": l2_size,
            "l1_d_size": l1_d_size,
            "l1_i_size": l1_i_size,
            "l2_assoc": 8,
            "l1_d_assoc": 4,
            "l1_i_assoc": 6,
            "benchmark_name": benchmark_name,
            "dataset_name": str(dataset_name),
            "issue_width": issue_width
        })

    return args_list


def get_dirname_by_params(l2_size: int = 1024,
                          l1d_size: int = 64,
                          l1i_size: int = 96,
                          l2_assoc: int = 8,
                          l1d_assoc: int = 4,
                          l1i_assoc: int = 6,
                          benchmark_name: str = "automotive_bitcount",
                          dataset_name: str = "1",
                          issue_width: int = None):
    """Get the output directory name by the parameters.
    Parameters
    ----------
    l2_size : int, optional
        The size of the L2 cache, by default 1024
    l1d_size : int, optional
        The size of the L1 data cache, by default 64
    l1i_size : int, optional
        The size of the L1 instruction cache, by default 96
    l2_assoc : int, optional
        The associativity of the L2 cache, by default 8
    l1d_assoc : int, optional
        The associativity of the L1 data cache, by default 4
    l1i_assoc : int, optional
        The associativity of the L1 instruction cache, by default 6
    benchmark_name : str, optional
        The name of the benchmark, by default "automotive_bitcount"
    dataset_name : str, optional
        The name of the dataset, by default "1"
    issue_width : int, optional
        The issue width of the processor, by default None
    Returns
    -------
    out_dir: str
        The path to the output directory.
    """
    if issue_width is not None:
        output_dir = f"{benchmark_name}_dataset_{dataset_name}_l2_{l2_size}_l1d_{l1d_size}_l1i_{l1i_size}_l2_assoc_{l2_assoc}_l1d_assoc_{l1d_assoc}_l1i_assoc_{l1i_assoc}_issue_width_{issue_width}_m5out"
    else:
        output_dir = f"{benchmark_name}_dataset_{dataset_name}_l2_{l2_size}_l1d_{l1d_size}_l1i_{l1i_size}_l2_assoc_{l2_assoc}_l1d_assoc_{l1d_assoc}_l1i_assoc_{l1i_assoc}_m5out"
    out_dir = os.path.join(cbench_dir, benchmark_name, "src_work", output_dir)
    return out_dir


def extract_sim_seconds_from_stats(output_dir: str) -> float:
    """Extract the simSeconds from the stats.txt file.
    Parameters
    ----------
    output_dir : str
        The path to the gem5 simulation output directory.
    Returns
    -------
    sim_seconds: float
        The simSeconds from the stats.txt file.
    """
    stats_file = os.path.join(output_dir, "stats.txt")
    if not (os.path.exists(stats_file)):
        raise FileNotFoundError(f"stats.txt not found in {output_dir}")
    with open(stats_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "simSeconds" in line:
                sim_seconds = line.split()[1]
                return float(sim_seconds)
        return 0


def get_latency_list(args_list: List[dict], baseline_index=3):
    """Get the normalized latency list based on the baseline.
    Parameters
    ----------
    args_list : List[Tuple]
        The list of arguments for the cBench benchmark.
    """
    # Check for the empty args_list
    if args_list is None or len(args_list) == 0:
        raise ValueError("Empty args_list")
    latency_list = []
    for args in args_list:
        latency_list.append(
            extract_sim_seconds_from_stats(get_dirname_by_params(**args)))
    # Check the latency list and filter out the invalid ones
    for latency in latency_list:
        if latency == 0:
            logger.warn(f"Invalide latency for args: {args_list}")
            return None
    # Get the baseline latency
    baseline_latency = latency_list[baseline_index] if baseline_index < len(
        latency_list) else latency_list[len(latency_list) // 2]
    logger.debug(f"baseline_latency: {baseline_latency}")
    # Normalize the latency based on the basedline
    normalized_latency_list = baseline_latency / np.array(
        [latency for latency in latency_list])

    return normalized_latency_list
