# Common commands in compilation process
generate unoptimized IR
```
clang {cflags} -emit-llvm -c -O3 -Xclang -disable-llvm-optzns -o {IR}
```
use opt to optimize IR
```
opt {seq} {IR} -o {IR_opt}
```
from optimized IR to obj (use -filetype=asm to generate asm)
```
llc -O3 -filetype=obj {IR_opt} -o {obj}
```
from obj to exe
```
clang {obj} -o {exe}
```
