import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    create_cbench_cache_workload,
    create_cbench_issue_width_workload,
    all_benchmarks,
    get_dirname_by_params,
    extract,
    get_latency_list)

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


def convert_kb_to_mb(kb: str):
    val = int(kb[:-2])
    if val >= 1024:
        return str(val // 1024) + "MB"
    else:
        return kb



def draw_one_benchmark_cache(args_list, ax=None, marker='*'):
    latency_list = get_latency_list(args_list)
    if latency_list is None:
        return ax
    print(args_list[0][-4], latency_list)
    cache_size_list = [convert_kb_to_mb(args[0]) for args in args_list]
    ax.plot(cache_size_list,
            latency_list,
            label=args_list[0][-3],
            marker=marker,
            markersize=12)

    return ax


# def draw_all_benchmark_cache():
#     fig, ax = plt.subplots()
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("automotive_bitcount",
#                                                    "1",
#                                                    unit="MB"),
#                             ax,
#                             marker="o")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("network_dijkstra",
#                                                    "19",
#                                                    unit="MB"),
#                             ax,
#                             marker="*")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("bzip2d", "20", unit="MB"),
#                             ax,
#                             marker="x")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("automotive_qsort1",
#                                                    "20",
#                                                    unit="MB"),
#                             ax,
#                             marker="v")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_sha", "20"),
#                             ax,
#                             marker="^")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("automotive_susan_e", "20"),
#                             ax,
#                             marker="<")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("automotive_susan_s", "20"),
#                             ax,
#                             marker=">")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("bzip2e", "20"),
#                             ax,
#                             marker="1")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_jpeg_c", "20"),
#                             ax,
#                             marker="2")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_jpeg_d", "20"),
#                             ax,
#                             marker="3")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_lame", "20"),
#                             ax,
#                             marker="4")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_tiff2rgba", "20"),
#                             ax,
#                             marker="s")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_tiffdither",
#                                                    "20"),
#                             ax,
#                             marker="p")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_tiffmedian",
#                                                    "20"),
#                             ax,
#                             marker="P")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("network_patricia", "20"),
#                             ax,
#                             marker="h")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_blowfish_d",
#                                                    "20"),
#                             ax,
#                             marker="H")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_blowfish_e",
#                                                    "20"),
#                             ax,
#                             marker="D")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_rijndael_d",
#                                                    "20"),
#                             ax,
#                             marker="X")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("security_rijndael_e",
#                                                    "20"),
#                             ax,
#                             marker="d")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_jpeg_d", "20"),
#                             ax,
#                             marker="|")
#     ax = draw_one_benchmark_cache(create_cbench_cache_workload("telecom_CRC32", "20"),
#                             ax,
#                             marker="_")
#     # ax = draw_one_benchmark_cache(create_cbench_cache_workload("consumer_mad", "20"), ax, marker="8")
#     # ax = draw_one_benchmark_cache(create_cbench_cache_workload("telecom_gsm", "20"), ax, marker="1")
#     # ax = draw_one_benchmark_cache(create_cbench_cache_workload("telecom_adpcm_d", "20")) # Too big, the gem5 simulation is too slow
#     # ax = draw(create_cbench_cache_workload("consumer_tiff2bw", "20", unit="MB"), ax, marker="+")

#     ax.legend(fontsize="10")
#     plt.xlabel('L2 Cache Size')
#     plt.ylabel('Speedup')
#     plt.title('cBench Speedup vs L2 Cache Size, Simulated on gem5')
#     plt.savefig("gem5-cache-size.png")
#     plt.savefig("gem5-cache-size.svg")


def draw_all_benchmark_cache():
    fig, ax = plt.subplots()
    markers_list = ["*", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "P", "h", "H", "D", "X", "d", "|", "_"]
    for benchmark, marker in zip(all_benchmarks, markers_list):
        ax = draw_one_benchmark_cache(create_cbench_cache_workload(benchmark, "20"),
                            ax, marker=marker)
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
