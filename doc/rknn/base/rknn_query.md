# misc


## query

### query linux version
```
[root@M1808 smswitch]# uname -a
Linux M1808 4.4.194 #1 SMP PREEMPT Thu Jul 9 14:43:13 CST 2020 aarch64 GNU/Linux
```

### query rknn_server version

``` 
[root@M1808 smswitch]# strings /usr/bin/rknn_server | grep build
1.4.0 (b4a8096 build: 2020-09-14 11:15:57)
.note.gnu.build-id
```
### query rknn_runtime version
```
[root@M1808 smswitch]# ls /usr/lib/ |grep -i rknn
librknn_api.so*
librknn_runtime.so
```

```
[root@M1808 smswitch]#  strings /usr/lib/librknn_runtime.so | grep build
librknn_runtime version 1.4.0 (4c92df0 build: 2020-09-11 20:27:18 base: 1112)
E [%s:%d]build tensor io fail
W [%s:%d]tensor ref input num > max_io %u, stop build
W [%s:%d]tensor ref output num > max_io %u, stop build
```

### query galcore version
```
[root@M1808 smswitch]# dmesg | grep -i galcore
[    9.336152] galcore: loading out-of-tree module taints kernel.
[    9.338977] galcore: npu init.
[    9.341009] galcore: start npu probe.
[    9.342806] Galcore version 6.4.0.227915
[    9.342811] Galcore options:
[    9.345647] Galcore Info: ContiguousBase=0x37400000 ContiguousSize=0x400000
[    9.349963] Galcore Info: MMU mapped external shared SRAM[0] CPU base=0xfec10000 GPU virtual address=0xfec10000 size=0x1f0000
[    9.349999] Galcore Info: MMU mapped core 0 SRAM[0] hardware virtual address=0x400000 size=0x80000
[    9.350016] Galcore Info: MMU mapped core 0 SRAM[1] hardware virtual address=0x480000 size=0x80000
[    9.353111] galcore ffbc0000.npu: Init npu devfreq
[    9.353287] galcore ffbc0000.npu: leakage=23
[    9.353556] galcore ffbc0000.npu: pvtm-volt-sel=2
[    9.354080] galcore ffbc0000.npu: avs=0
[    9.355514] galcore ffbc0000.npu: l=0 h=2147483647 hyst=5000 l_limit=0 h_limit=0
```

## misc

```
[root@M1808 smswitch]# ls /usr/bin/ |grep -i npu
input-event-daemon*
libinput*
libinput-debug-events*
libinput-list-devices*
npu_uvc_device.sh*
rk_npu_uvc_device*
ts_uinput*


[root@M1808 smswitch]# ls /usr/bin/ |grep -i usb
lsusb*
lsusb.py*
start_usb.sh*
usbdevice*
usbhid-dump*

```