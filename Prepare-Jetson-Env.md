
## Using CLI to setup proxy on DSS server
We use [brook](https://github.com/txthinking/brook.git) as the proxy provider.
Download and install brook on DSS server, then start the brook server and client:
```shell
nohup brook server -l :9999 -p password &
nohup brook client -s 127.0.0.1:9999 -p password --http 0.0.0.0:1081 &
```
Note we need set the client ip address to `0.0.0.0` to allow access from other machines.
Test on DSS to see whether we can access internet through the brook proxy:

```
curl -x 127.0.0.1:1081 www.google.com
```
Then we setup  proxy for `apt` on the jetson TX2 board:
```
sudo vim /etc/apt/apt.conf.d/proxy.conf
```
and add the following lines to the `proxy.conf`:
```shell
Acquire::http::Proxy "http://169.254.25.35:1081/";
Acquire::https::Proxy "http://169.254.25.35:1081/";
```
Note that  `169.254.25.35` is the ip address of the ethernet which connects with jetson boards.
Then we can use apt to install software on jetson TX2 board.


## Install `perf` from source
We tried to install perf by apt but their isn't perf for jetson TX2's linux version (We can get the linux version with cmd `uname -r`).
We need to build from linux source code.
```shell
wget https://mirrors.edge.kernel.org/pub/linux/kernel/v4.x/linux-4.4.38.tar.gz
tar -xvf linux-4.4.38.tar.gz
cd linux-4.4.38/tools/perf
make -j4
```
If everthing is OK, then we can see `perf` in the folder.
copy it the the system binary folder:

```shell
cp ./perf /usr/bin/
```

Then we can test whether `perf` works and list all the software and hardware events:

```shell
perf list
```
For more perf usage, go to [this](https://www.brendangregg.com/perf.html) website.


## TODO collect profiling data for cBench
