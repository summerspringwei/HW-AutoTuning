import os
import subprocess
from multiprocessing import Pool

from utils import create_cbench_cache_workload, get_cbench_cmd_args, all_benchmarks, create_cbench_issue_width_workload

gem5_binary = "/home/xiachunwei/Software/gem5/build/ARM/gem5.fast"
scripts_path = "/home/xiachunwei/Software/gem5/configs/deprecated/example/se.py"
toolchain_path = "/home/xiachunwei/Software/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu/aarch64-none-linux-gnu/libc"
cbench_home = "/home/xiachunwei/Projects/ArmCBench/cBench_V1.1"
cbench_log_dir = "/home/xiachunwei/Software/gem5/cbench_logs/"


def run_gem5_simulation(l2_size="2MB",
                        l1d_size="32kB",
                        l1i_size="32kB",
                        l2_assoc=8,
                        l1d_assoc=2,
                        l1i_assoc=2,
                        benchmark_name="network_dijkstra",
                        dataset_name="19",
                        issue_width=8):
    
    print(f"Simulation for {benchmark_name} with dataset {dataset_name} start.")
    output_dir = f"{benchmark_name}_dataset_{dataset_name}_l2_{l2_size}_l1d_{l1d_size}_l1i_{l1i_size}_l2_assoc_{l2_assoc}_l1d_assoc_{l1d_assoc}_l1i_assoc_{l1i_assoc}_issue_width_{issue_width}_m5out"
    fstdout = open(f"{cbench_log_dir}/{output_dir}.stdout", "w")
    fstderr = open(f"{cbench_log_dir}/{output_dir}.stderr", "w")
    option_list = get_cbench_cmd_args(benchmark_name, dataset_name)
    # print(option_list)
    subprocess.run(
        [
            gem5_binary,
            f"--outdir={output_dir}",
            scripts_path,
            "--caches",
            "--l2cache",
            "--arm-iset=aarch64",
            "--cacheline_size=64",
            f"--cpu-type=ArmO3CPU",
            f"--l2_size={l2_size}",
            f"--l2_assoc={l2_assoc}",
            f"--l1d_size={l1d_size}",
            f"--l1d_assoc={l1d_assoc}",
            f"--l1i_size={l1i_size}",
            f"--l1i_assoc={l1i_assoc}",
            "--mem-size=4GB",
            f"--redirects=/lib={toolchain_path}",
            f"--redirects=/lib64={toolchain_path}/lib64",
            f"--interp-dir={toolchain_path}",
            f"--issuewidth={issue_width}",
            "--cmd", "./a.out",
            f"-o={option_list}"
        ], 
        cwd=os.path.join(cbench_home, benchmark_name, "src_work"),
        stdout=fstdout, stderr=fstderr)
    fstdout.flush()
    fstdout.close()
    fstderr.flush()
    fstderr.close()
    print(f"Simulation for {benchmark_name} with dataset {dataset_name} finished.")


def main():
    args_list = []

    for benchmark in all_benchmarks:
        args_list.extend(create_cbench_issue_width_workload(benchmark, "20"))
    # args_list.extend(create_cbench_cache_workload("security_sha", "20"))
    # args_list.extend(create_cbench_cache_workload("automotive_susan_e", "20"))
    # args_list.extend(create_cbench_cache_workload("automotive_susan_s", "20"))
    # args_list.extend(create_cbench_cache_workload("bzip2e", "20"))
    # args_list.extend(create_cbench_cache_workload("consumer_jpeg_c", "20"))
    # args_list.extend(create_cbench_cache_workload("consumer_jpeg_d", "20"))
    # args_list.extend(create_cbench_cache_workload("consumer_lame", "20"))
    # args_list.extend(create_cbench_cache_workload("consumer_mad", "20"))
    # args_list.extend(create_cbench_cache_workload("consumer_tiff2rgba", "20"))
    # args_list.extend(create_cbench_cache_workload("consumer_tiffdither", "20"))
    # args_list.extend(create_cbench_cache_workload("consumer_tiffmedian", "20"))
    # args_list.extend(create_cbench_cache_workload("network_patricia", "20"))
    # args_list.extend(create_cbench_cache_workload("security_blowfish_d", "20"))
    # args_list.extend(create_cbench_cache_workload("security_blowfish_e", "20"))
    # args_list.extend(create_cbench_cache_workload("security_rijndael_d", "20"))
    # args_list.extend(create_cbench_cache_workload("security_rijndael_e", "20"))
    # args_list.extend(create_cbench_cache_workload("consumer_jpeg_d", "20"))
    # args_list.extend(create_cbench_cache_workload("telecom_adpcm_d", "20"))
    # args_list.extend(create_cbench_cache_workload("telecom_CRC32", "20"))
    # args_list.extend(create_cbench_cache_workload("telecom_gsm", "20"))
    # args_list.append(["2MB", "32kB", "32kB", 8, 2, 2, "automotive_qsort1", "1", 2])

    n_threads = len(args_list)
    with Pool(n_threads) as p:
        p.starmap(run_gem5_simulation, args_list)


if __name__=="__main__":
    main()
