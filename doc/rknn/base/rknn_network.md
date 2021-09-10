# 通过网线连接rknn 


### 板端ip设置
```
[root@M1126:/]# ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN group default qlen 10
    link/can
3: eth0: <BROADCAST,MULTICAST,DYNAMIC,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether a2:98:b0:72:18:90 brd ff:ff:ff:ff:ff:ff
```
可以看到ip地址为空

通过`ifconfig eth0 192.168.1.136 netmask 255.255.255.0` 配置ip地址
```
[root@M1126:/]# ifconfig eth0 192.168.1.136 netmask 255.255.255.0
[root@M1126:/]# ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN group default qlen 10
    link/can
3: eth0: <BROADCAST,MULTICAST,DYNAMIC,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether a2:98:b0:72:18:90 brd ff:ff:ff:ff:ff:ff
    inet 192.168.1.136/24 brd 192.168.1.255 scope global eth0
       valid_lft forever preferred_lft forever
```

### pc端ip设置
开发板上默认的ip地址是 `192.168.1.136`, 这里需要把pc的IP地址修改的一致，确认都在 `192.168.1.1/24` 网关下。
**注意**： pc上有多块网卡，有多块虚拟的网络适配器，需要正确找到以太网对应那个网络适配器，只有这个的网卡和网线相连。
![img](./imgs/1.png)


改成 ip地址是 `192.168.1.101`, gateway `192.168.1.1`  netmask `255.255.255.0`
![img](./imgs/2.png)

### ping测试

在pc机上执行 即可
```
F:\tmp>ping 192.168.1.136

正在 Ping 192.168.1.136 具有 32 字节的数据:
来自 192.168.1.136 的回复: 字节=32 时间<1ms TTL=64
来自 192.168.1.136 的回复: 字节=32 时间<1ms TTL=64
来自 192.168.1.136 的回复: 字节=32 时间<1ms TTL=64
来自 192.168.1.136 的回复: 字节=32 时间<1ms TTL=64

192.168.1.136 的 Ping 统计信息:
    数据包: 已发送 = 4，已接收 = 4，丢失 = 0 (0% 丢失)，
往返行程的估计时间(以毫秒为单位):
    最短 = 0ms，最长 = 0ms，平均 = 0ms
```


或者在 开发板上执行`ping 192.168.1.101`


### 配置启动项

执行重启之后，发现 ip设置又失效了，所以需要配置启动项。

``` bash
vim /etc/init.d/S999start_network
chmod +x /etc/init.d/S999start_network

```