# rknn misc


## 设备连接

- 通过usb线实现adb 连接
  - [x] linux 可用
  - [x] windows 可用，NPU设备不可用
- 通过usb线实现ntb 连接
  - [x] windows 可用，NPU设备可用
- [x] 通过串口线实现连接
- [x] 通过网口线实现连接
- [ ] 通过wifi实现连接


## client/pc 

### ADB

### NTB
NTB(Non-Transparent Bridge), windows 系统下使用NTB协议连接设备。
对应的NTB客户端是：
- windows: `Lib\site-packages\rknn\3rdparty\platform-tools\ntp\windows-x86_64\npu_transfer_proxy.exe`
- linux: `lib/python3.6/site-packages/rknn/3rdparty/platform-tools/ntp/linux-x86_64/npu_transfer_proxy`

### network

`ifconfig eth0 192.168.1.136 netmask 255.255.255.0`



## device
### /usr/bin/start_usb.sh

start_usb.sh 可以使设备切换成adb模式或ntb模式。


``` bash
#!/bin/sh

configfs_init()
{
    PID=$1
    CONFIG_STRING=$2
    mkdir -p /dev/usb-ffs -m 0770
    mkdir -p /dev/usb-ffs/$CONFIG_STRING -m 0770
    mount -t configfs none /sys/kernel/config
    mkdir -p /sys/kernel/config/usb_gadget/rockchip  -m 0770
    echo 0x2207 > /sys/kernel/config/usb_gadget/rockchip/idVendor
    echo $PID > /sys/kernel/config/usb_gadget/rockchip/idProduct
    mkdir -p /sys/kernel/config/usb_gadget/rockchip/strings/0x409   -m 0770
    echo "0123456789ABCDEF" > /sys/kernel/config/usb_gadget/rockchip/strings/0x409/serialnumber
    echo "rockchip"  > /sys/kernel/config/usb_gadget/rockchip/strings/0x409/manufacturer
    echo "rk3xxx"  > /sys/kernel/config/usb_gadget/rockchip/strings/0x409/product
    mkdir -p /sys/kernel/config/usb_gadget/rockchip/configs/b.1  -m 0770
    mkdir -p /sys/kernel/config/usb_gadget/rockchip/configs/b.1/strings/0x409  -m 0770
    echo 500 > /sys/kernel/config/usb_gadget/rockchip/configs/b.1/MaxPower
    echo \"$CONFIG_STRING\" > /sys/kernel/config/usb_gadget/rockchip/configs/b.1/strings/0x409/configuration
}

function_init()
{
    CONFIG_STRING=$1
    mkdir -p /sys/kernel/config/usb_gadget/rockchip/functions/ffs.$CONFIG_STRING
    rm -f /sys/kernel/config/usb_gadget/rockchip/configs/b.1/ffs.*
    ln -s /sys/kernel/config/usb_gadget/rockchip/functions/ffs.$CONFIG_STRING /sys/kernel/config/usb_gadget/rockchip/configs/b.1/ffs.$CONFIG_STRING
}

case "$1" in
adb)
    killall adbd start_rknn.sh rknn_server > /dev/null 2>&1

    echo "none" > /sys/kernel/config/usb_gadget/rockchip/UDC

    umount /sys/kernel/config
    umount /dev/usb-ffs/ntb > /dev/null 2>&1
    rm -rf /dev/usb-ffs/ntb

    configfs_init 0x0006 adb
    function_init adb

    # START_APP_BEFORE_UDC
    mkdir -p /dev/usb-ffs/adb
    mount -o uid=2000,gid=2000 -t functionfs adb /dev/usb-ffs/adb
    export service_adb_tcp_port=5555
    adbd&
    sleep 1

    UDC=`ls /sys/class/udc/| awk '{print $1}'`
    echo $UDC > /sys/kernel/config/usb_gadget/rockchip/UDC
    # START_APP_AFTER_UDC

    start_rknn.sh &

    ;;
ntb)
    killall adbd start_rknn.sh rknn_server > /dev/null 2>&1

    echo "none" > /sys/kernel/config/usb_gadget/rockchip/UDC

    umount /sys/kernel/config
    umount /dev/usb-ffs/adb > /dev/null 2>&1
    rm -rf /dev/usb-ffs/adb

    configfs_init 0x1808 ntb
    function_init ntb

    # START_APP_BEFORE_UDC
    mkdir -p /dev/usb-ffs/ntb
    mount -o uid=2000,gid=2000 -t functionfs ntb /dev/usb-ffs/ntb

    start_rknn.sh &

    ;;
*)
    echo "Usage: $0 {adb|ntb}"
    exit 1
esac

exit 0
```


#### demo
一行命令通过`adb shell`把设备切换到ntb模式：`adb shell nohup start_usb.sh ntb`

主要通过调用`start_rknn.sh`，`start_rknn.sh` 主要调用 `rknn_server`

### /usr/bin/start_rknn.sh

该程序是以无限循环的方式调用 `rknn_server` , 因为 `rknn_server`如果没连上会自动退出。

``` bash
#!/bin/sh

export RKNN_SERVER_PLUGINS='/usr/lib/npu/rknn/plugins/'

while true
do
  sleep 1
  rknn_server #>/dev/null 2>&1
done
```

### /usr/bin/rknn_server


```
[root@M1808]# rknn_server
start rknn server, version:1.4.0 (b4a8096 build: 2020-09-14 11:15:57)
1 NNPluginManager loadPlugins(41): No plugin find!
I NPUTransfer: Starting NPU Transfer Server, Transfer version 2.0.0 (8f9ebbc@2020-04-03T09:12:43)
E NPUTransfer: Transfer interface open failed!, USB_HOST: 0, name = /dev/usb-ffs/ntb
E NPUTransfer: Please open transfer first!
3 MsgThread recvTask(42): transfer may error, restart server!

[root@M1808 smswitch]# export RKNN_SERVER_PLUGINS='/usr/lib/npu/rknn/plugins/'
[root@M1808 smswitch]# rknn_server
start rknn server, version:1.4.0 (b4a8096 build: 2020-09-14 11:15:57)
I NPUTransfer: Starting NPU Transfer Server, Transfer version 2.0.0 (8f9ebbc@2020-04-03T09:12:43)
E NPUTransfer: Transfer interface open failed!, USB_HOST: 0, name = /dev/usb-ffs/ntb
E NPUTransfer: Please open transfer first!
36 MsgThread recvTask(42): transfer may error, restart server!
```


``` 
$ file rknn_server

rknn_server: ELF 64-bit LSB executable, ARM aarch64, version 1 (SYSV), dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, for GNU/Linux 3.7.0, BuildID[sha1]=c72079407ace7030c84c40a1ea9fa01dd71795ad, stripped
(base)
```


### /usr/bin/restart_rknn.sh

``` bash
#!/bin/sh

killall start_rknn.sh > /dev/null 2>&1
killall rknn_server > /dev/null 2>&1
start_rknn.sh &

```

## misc
```
[root@M1808 smswitch]# ls /usr/bin/ |grep -i rknn
restart_rknn.sh*
rknn_server*
start_rknn.sh*
```