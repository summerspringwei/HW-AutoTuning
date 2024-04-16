import os
import numpy as np
import matplotlib.pyplot as plt

from utils import create_cbench_cache_workload, create_cbench_issue_width_workload, all_benchmarks

import matplotlib.pylab as pylab

params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (12, 8),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-small',
    'ytick.labelsize': 'medium'
}
pylab.rcParams.update(params)

cbench_dir = "/home/xiachunwei/Projects/ArmCBench/cBench_V1.1/"

def get_dirname_by_params(l2_size, l1d_size, l1i_size, l2_assoc, l1d_assoc, l1i_assoc,
            benchmark_name, dataset_name, issue_width=None):
    if issue_width is not None:
        output_dir = f"{benchmark_name}_dataset_{dataset_name}_l2_{l2_size}_l1d_{l1d_size}_l1i_{l1i_size}_l2_assoc_{l2_assoc}_l1d_assoc_{l1d_assoc}_l1i_assoc_{l1i_assoc}_issue_width_{issue_width}_m5out"
    else:
        output_dir = f"{benchmark_name}_dataset_{dataset_name}_l2_{l2_size}_l1d_{l1d_size}_l1i_{l1i_size}_l2_assoc_{l2_assoc}_l1d_assoc_{l1d_assoc}_l1i_assoc_{l1i_assoc}_m5out"
    return os.path.join(cbench_dir, benchmark_name, "src_work", output_dir)


def extract(output_dir):
    stats_file = os.path.join(output_dir, "stats.txt")
    if not (os.path.exists(stats_file)):
        return None
    with open(stats_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "simSeconds" in line:
                sim_seconds = line.split()[1]
                return sim_seconds
    return None


def convert_kb_to_mb(kb: str):
    val = int(kb[:-2])
    if val >= 1024:
        return str(val // 1024) + "MB"
    else:
        return kb

def get_latency_list(args_list):
    # Check for the empty args_list
    if args_list is None or len(args_list) == 0:
        return None
    latency_list = []
    for args in args_list:
        latency_list.append(extract(get_dirname_by_params(*args)))
    # Check the latency list and filter out the invalid ones
    for latency in latency_list:
        if latency is None:
            print(f"Invalide latency for args: {args_list}")
            return None
    print(latency_list)
    baseline_latency = float(latency_list[3])
    latency_list = baseline_latency / np.array(
        [float(latency) for latency in latency_list])
    print(latency_list)

    return latency_list

def draw_one_benchmark_cache(args_list, ax=None, marker='*'):
    latency_list = get_latency_list(args_list)
    if latency_list is None:
        return ax
    cache_size_list = [convert_kb_to_mb(args[0]) for args in args_list]
    ax.plot(cache_size_list,
            latency_list,
            label=args_list[0][-2],
            marker=marker,
            markersize=12)

    return ax


def draw_all_benchmark_cache():
    fig, ax = plt.subplots()
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("automotive_bitcount",
                                                   "1",
                                                   unit="MB"),
                            ax,
                            marker="o")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("network_dijkstra",
                                                   "19",
                                                   unit="MB"),
                            ax,
                            marker="*")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("bzip2d", "20", unit="MB"),
                            ax,
                            marker="x")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("automotive_qsort1",
                                                   "20",
                                                   unit="MB"),
                            ax,
                            marker="v")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_sha", "20"),
                            ax,
                            marker="^")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("automotive_susan_e", "20"),
                            ax,
                            marker="<")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("automotive_susan_s", "20"),
                            ax,
                            marker=">")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("bzip2e", "20"),
                            ax,
                            marker="1")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_jpeg_c", "20"),
                            ax,
                            marker="2")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_jpeg_d", "20"),
                            ax,
                            marker="3")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_lame", "20"),
                            ax,
                            marker="4")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_tiff2rgba", "20"),
                            ax,
                            marker="s")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_tiffdither",
                                                   "20"),
                            ax,
                            marker="p")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_tiffmedian",
                                                   "20"),
                            ax,
                            marker="P")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("network_patricia", "20"),
                            ax,
                            marker="h")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_blowfish_d",
                                                   "20"),
                            ax,
                            marker="H")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_blowfish_e",
                                                   "20"),
                            ax,
                            marker="D")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_rijndael_d",
                                                   "20"),
                            ax,
                            marker="X")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_rijndael_e",
                                                   "20"),
                            ax,
                            marker="d")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_jpeg_d", "20"),
                            ax,
                            marker="|")
    ax = draw_one_benchmark_cache(create_cbench_cache_workload("telecom_CRC32", "20"),
                            ax,
                            marker="_")
    # ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_mad", "20"), ax, marker="8")
    # ax = draw_one_benchmark_cache(create_cbench_cache_workload("telecom_gsm", "20"), ax, marker="1")
    # ax = draw_one_benchmark_cache(create_cbench_cache_workload("telecom_adpcm_d", "20")) # Too big, the gem5 simulation is too slow
    # ax = draw(create_cbench_cache_workload("consumer_tiff2bw", "20", unit="MB"), ax, marker="+")

    ax.legend(fontsize="10")
    plt.xlabel('L2 Cache Size')
    plt.ylabel('Speedup')
    plt.title('cBench Speedup vs L2 Cache Size, Simulated on gem5')
    plt.savefig("gem5-cache-size.png")
    plt.savefig("gem5-cache-size.svg")


def draw_one_benchmark_issue_width(args_list, ax=None, marker='*'):
    latency_list = get_latency_list(args_list)
    if latency_list is None:
        return ax
    issue_width_list = [args[-1] for args in args_list]
    ax.plot(issue_width_list,
            latency_list,
            label=args_list[0][-3],
            marker=marker,
            markersize=12)

    return ax


def draw_all_benchmark_issue_width():
    fig, ax = plt.subplots()
    markers_list = ["*", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "P", "h", "H", "D", "X", "d", "|", "_"]
    for benchmark, marker in zip(all_benchmarks, markers_list):
        ax = draw_one_benchmark_issue_width(create_cbench_issue_width_workload(benchmark, "20"),
                            ax, marker=marker)
    ax.legend(fontsize="10")
    plt.xlabel('Issue width')
    plt.ylabel('Speedup')
    plt.title('cBench Speedup vs issue width, Simulated on gem5')
    plt.savefig("gem5-issue-width.png")
    plt.savefig("gem5-issue-width.svg")


if __name__ == "__main__":
    draw_all_benchmark_cache()
    draw_all_benchmark_issue_width()
