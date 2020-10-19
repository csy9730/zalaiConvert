# zalaiConvert



- [ ] add: xxx
- [ ] add: 添加反馈 : parse export success!
- [ ] convertWrap2::解析 fps
- [ ] 调整rknn.log
- [ ] add: `python3 -m zalaiConvert.convertWrap -c foo.json`
- [x] add：在哪里内置adb程序？
add adb to whl or add to qml?
adb程序很小，直接内置wheel进去。

```

                          Performance
========================================================================
Total Time(us): 106762
FPS: 9.37
========================================================================

======================================================
            Memory Profile Info Dump
======================================================
System memory:
    maximum allocation : 39.08 MiB
    total allocation   : 197.72 MiB
NPU memory:
    maximum allocation : 141.80 MiB
    total allocation   : 143.33 MiB

Total memory:
    maximum allocation : 180.88 MiB
    total allocation   : 341.04 MiB

INFO: When evaluating memory usage, we need consider
the size of model, current model size is: 58.87 MiB
```


## misc


- [x] add: git
- [x] add: requirements.txt
- [x] add: setup.py
- [ ] add: logger
- [ ] add: listDevice
- [x] add: `adb devices` 
- [x] `adb shell`
- [x] `series ssh shell` 切换成adb模式
` python -c "from rknn.api import RKNN;print(RKNN().list_devices())" `

- [x] add: zalai::stdout
- [ ] rm : tmp.t
`from rknn.api.rknn_base import RKNNBase;RKNNBase().list_devices()`


`start_usb.sh  ntb`

```
which start_usb.sh
/usr/bin/start_usb.sh

adb shell nohup start_usb.sh ntb
[root@M1808 ~]# vi /usr/bin/start_usb.sh
[root@M1808 ~]# ls /dev/usb-ffs/
```



## arch

进程调用=》flask服务器=》json协议
process call => flask => return json 