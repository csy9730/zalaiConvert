# zalaiConvert

- [ ] add: xxx
- [ ] add：在哪里内置adb程序？


python3 -m zalaiConvert.convertWrap -c foo.json



- [x] add: git
- [x] add: requirements.txt
- [x] add: setup.py
- [ ] add: logger
- [ ] add: listDevice
- [ ] add: `adb devices` 
- [ ] `adb shell`
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