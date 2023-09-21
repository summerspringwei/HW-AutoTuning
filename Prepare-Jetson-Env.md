
## Using CLI to setup proxy on DSS server
We use [brook](https://github.com/txthinking/brook.git) as the proxy provider.
Download and install brook on DSS server, then start the brook server and client:
```shell
nohup brook server -l :9999 -p password &
nohup brook client -s 127.0.0.1:9999 -p password --http 0.0.0.0:1081 &
```
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
Then we can use apt to install software on jetson TX2 board.