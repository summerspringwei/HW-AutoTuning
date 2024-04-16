
import pickle
from program_record import ProgramRecord


def load_dataset(file_path: str):
    with open(file_path, 'rb') as file:
        (stats_and_pmu_list, speedup_list, time_list, y_O3) = pickle.load(file)
        # Get all the keys
        features_keys = set()
        perf_keys = set()
        for [stats, pmu] in stats_and_pmu_list:
            features_keys.update(stats.keys())
            perf_keys.update(pmu.keys())
        # print(f"features_keys:{features_keys}, len:{len(features_keys)}")
        # print(f"perf_keys:{perf_keys}, len:{len(perf_keys)}")
        record_list = [ProgramRecord(state_and_pmu[0], state_and_pmu[1], speedup, time) for state_and_pmu, speedup, time in zip(stats_and_pmu_list, speedup_list, time_list)]
    return record_list, features_keys, perf_keys


all_benchmarks = [
        'automotive_bitcount', 
        'automotive_qsort1', 'automotive_susan_c',
        'automotive_susan_e', 'automotive_susan_s', 'bzip2d', 'bzip2e',
        'consumer_jpeg_c', 'consumer_jpeg_d', 'consumer_lame',
        'consumer_tiff2bw', 'consumer_tiff2rgba', 'consumer_tiffdither',
        'consumer_tiffmedian', 'network_patricia',
        'office_stringsearch1', 'security_blowfish_d', 'security_blowfish_e',
        'security_rijndael_d', 'security_rijndael_e', 'security_sha',
        'telecom_CRC32', 'telecom_adpcm_c', 'telecom_adpcm_d', 'telecom_gsm'
    ]

dataset_path = 'data_perf_features'

def get_cbench_cmd_args(benchmark_name, dataset_name: str) -> str:
    benchmark_dataset_map = {
        "network_dijkstra": f"../../network_dijkstra_data/{dataset_name}.dat", # OK
        "automotive_bitcount": "1125000", # OK
        "automotive_qsort1": f"../../automotive_qsort_data/{dataset_name}.dat", # OK
        "automotive_susan_c": f"../../automotive_susan_data/{dataset_name}.pgm output_large.corners.pgm -c", # OK
        "security_sha": f"../../office_data/{dataset_name}.txt",
        "automotive_susan_e": f"../../automotive_susan_data/{dataset_name}.pgm output_large.edges.pgm -e",
        "automotive_susan_s": f"../../automotive_susan_data/{dataset_name}.pgm output_large.smoothing.pgm -s",
        "bzip2d": f"-d -k -f -c ../../bzip2_data/{dataset_name}.bz2", # OK
        "bzip2e": f"-z -k -f -c ../../automotive_qsort_data/{dataset_name}.dat",
        "consumer_jpeg_c": f"-dct int -progressive -opt -outfile output_large_encode.jpeg ../../consumer_jpeg_data/{dataset_name}.ppm",
        "consumer_jpeg_d": f"-dct int -ppm -outfile output_large_decode.ppm ../../consumer_jpeg_data/{dataset_name}.jpg",
        "consumer_lame": f"../../consumer_data/{dataset_name}.wav output_large.mp3",
        "consumer_mad": f"--time=100000 --output=wave:output.wav ../../consumer_data/{dataset_name}.mp3", # pass
        "consumer_tiff2bw": f"../../consumer_tiff_data/{dataset_name}.tif output_largebw.tif", # OK
        "consumer_tiff2rgba": f"-c none ../../consumer_tiff_data/{dataset_name}.tif output_largergba.tif",
        "consumer_tiffdither": f"-c g4 ../../consumer_tiff_data/{dataset_name}.bw.tif output_largedither.tif",
        "consumer_tiffmedian": f"../../consumer_tiff_data/{dataset_name}.nocomp.tif output_largemedian.tif",
        "office_ghostscript": f"-sDEVICE=ppm -dNOPAUSE -q -sOutputFile=output.ppm -- ../../office_data/{dataset_name}.ps", # pass
        "telecom_adpcm_c": f"< ../../telecom_data/{dataset_name}.pcm > output_large.adpcm",
        "network_patricia": f"../../network_patricia_data/{dataset_name}.udp",
        "security_blowfish_d": f"d ../../office_data/{dataset_name}.benc output_large.txt 1234567890abcdeffedcba0987654321",
        "security_blowfish_e": f"e ../../office_data/{dataset_name}.txt output_large.enc 1234567890abcdeffedcba0987654321",
        "security_rijndael_d": f"../../office_data/{dataset_name}.enc output_large.dec d 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321",
        "security_rijndael_e": f"../../office_data/{dataset_name}.txt output_large.enc e 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321",
        "telecom_adpcm_d": f"< ../../telecom_data/{dataset_name}.adpcm > output_large.pcm",
        "telecom_CRC32": f"../../telecom_data/{dataset_name}.pcm > output_large.txt",
        "telecom_gsm": f"-fps -c ../../telecom_gsm_data/{dataset_name}.au",
    }
    if benchmark_name in benchmark_dataset_map:
        return benchmark_dataset_map[benchmark_name]
    else:
        assert False, f"benchmark_name: {benchmark_name} not found in benchmark_dataset_map"


def create_cbench_cache_workload(benchmark_name, dataset_name, unit="kB"):
    args_list = []
    l2_size, l1_d_size, l1_i_size = 256, 4, 6
    for i in range(7):
        if l2_size >= 1024 and unit == "MB":
                args_list.append(
                    (f"{l2_size // 1024}MB", f"{l1_d_size}kB", f"{l1_i_size}kB", 8, 4, 6, benchmark_name, str(dataset_name))
                )
        else:
            args_list.append(
                (f"{l2_size}kB", f"{l1_d_size}kB", f"{l1_i_size}kB", 8, 4, 6, benchmark_name, str(dataset_name)) 
            )
        l2_size, l1_d_size, l1_i_size = l2_size * 2, l1_d_size * 2, l1_i_size * 2
    
    return args_list


def create_cbench_issue_width_workload(benchmark_name, dataset_name):
    args_list = []
    l2_size, l1_d_size, l1_i_size = 1024*4, 64, 96
    
    for i in range(7):
        issue_width = i + 1
        args_list.append(
            (f"{l2_size}kB", f"{l1_d_size}kB", f"{l1_i_size}kB", 8, 4, 6, benchmark_name, str(dataset_name), issue_width)
        )
        
    
    return args_list