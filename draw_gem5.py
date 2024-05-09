from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from utils import (create_cbench_cache_workload,
                   create_cbench_issue_width_workload, get_latency_list,
                   get_logger, all_benchmarks)

# Set the style of the plot
params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (12, 8),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-small',
    'ytick.labelsize': 'medium'
}
pylab.rcParams.update(params)
logger = get_logger("draw_gem5")


def convert_kb_to_mb(kb: str) -> str:
    """Convert KB to MB if the value is larger than 1024KB.
    
    e.g. 2048KB -> 2MB

    Parameters
    ----------
    kb : str
        Value in KB.
    
    Returns
    -------
    str
        Value in MB if value is large than 1024kB.
    """
    val = int(kb[:-2])
    if val >= 1024:
        return str(val // 1024) + "MB"
    else:
        return kb


def draw_one_benchmark_cache(args_list: List[Dict],
                             ax: plt.Axes = None,
                             marker: str = '*') -> plt.Axes:
    """Draw the latency vs cache size plot for one benchmark.
    Parameters
    ----------
    args_list : List[Dict]
        List of arguments for the benchmark.
    ax : plt.Axes
        Axes object for the plot.
    marker : str
        Marker for the plot.
    
    Returns
    -------
    plt.Axes
        Axes object for the plot.
    """

    if args_list is None or len(args_list) == 0:
        return ax
    latency_list = get_latency_list(args_list)
    if latency_list is None:
        return ax
    cache_size_list = [convert_kb_to_mb(args["l2_size"]) for args in args_list]
    ax.plot(cache_size_list,
            latency_list,
            label=args_list[0]["benchmark_name"],
            marker=marker,
            markersize=12)

    return ax


def draw_all_benchmark_cache():
    """Draw the latency vs cache size plot for all benchmarks."""
    _, ax = plt.subplots()
    markers_list = [
        "*", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "P", "h",
        "H", "D", "X", "d", "|", "_"
    ]
    for benchmark, marker in zip(all_benchmarks, markers_list):
        ax = draw_one_benchmark_cache(create_cbench_cache_workload(
            benchmark, "20"),
                                      ax,
                                      marker=marker)
    ax.legend(fontsize="10")
    plt.xlabel('L2 Cache Size')
    plt.ylabel('Speedup')
    plt.title('cBench Speedup vs L2 Cache Size, Simulated on gem5')
    plt.savefig("figures/gem5-cache-size.png")
    plt.savefig("figures/gem5-cache-size.svg")
    logger.info("Plots for cache size are saved")


def draw_one_benchmark_issue_width(args_list: List[Dict],
                                   ax: plt.Axes = None,
                                   marker: str = '*'):
    """Draw the latency vs issue width plot for one benchmark."""
    latency_list = get_latency_list(args_list)
    if latency_list is None:
        return ax
    issue_width_list = [args["issue_width"] for args in args_list]
    ax.plot(issue_width_list,
            latency_list,
            label=args_list[0]["benchmark_name"],
            marker=marker,
            markersize=12)

    return ax


def draw_all_benchmark_issue_width():
    """Draw the latency vs issue width plot for all benchmarks."""
    _, ax = plt.subplots()
    markers_list = [
        "*", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "P", "h",
        "H", "D", "X", "d", "|", "_"
    ]
    for benchmark, marker in zip(all_benchmarks, markers_list):
        ax = draw_one_benchmark_issue_width(create_cbench_issue_width_workload(
            benchmark, "20"),
                                            ax,
                                            marker=marker)
    ax.legend(fontsize="10")
    plt.xlabel('Issue width')
    plt.ylabel('Speedup')
    plt.title('cBench Speedup vs issue width, Simulated on gem5')
    plt.savefig("figures/gem5-issue-width.png")
    plt.savefig("figures/gem5-issue-width.svg")
    logger.info("Plots for issue width are saved")


if __name__ == "__main__":
    draw_all_benchmark_cache()
    draw_all_benchmark_issue_width()
