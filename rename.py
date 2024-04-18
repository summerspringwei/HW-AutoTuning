
from utils import all_benchmarks, cbench_dir
import re
import os

with open("tmp.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        old_line = line
        if line.endswith("assoc_6_m5out"):
            line = line.replace("assoc_6_m5out", "assoc_6_issue_width_8_m5out")
        if line.find("MB"):
            match = re.search(r'(\d+)MB', line)
            if match:
                cache_size = int(match.group(1))
                line = line.replace(f"{cache_size}MB", f"{cache_size*1024}kB")
        if old_line != line:
            print(line)
            if not os.path.exists(os.path.join(cbench_dir, line)):
                os.rename(os.path.join(cbench_dir, old_line), os.path.join(cbench_dir, line))
            
