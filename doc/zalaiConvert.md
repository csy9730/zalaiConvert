# zalaiConvert

## connection
### zalaiConvert list

#### adb/ntb
`start_usb.sh  ntb`


```
which start_usb.sh
/usr/bin/start_usb.sh
```

```
adb shell nohup start_usb.sh ntb
[root@M1808 ~]# vi /usr/bin/start_usb.sh
[root@M1808 ~]# ls /dev/usb-ffs/
```

```
python -m rknn.bin.list_devices

*************************
all device(s) with ntb mode:
4486dcfc35a505c0
*************************
```

### zalaiConvert killserver
#### NTB protocl
NTB(Non-Transparent Bridge)

`zal_rk130\Lib\site-packages\rknn\3rdparty\platform-tools\ntp\windows-x86_64\npu_transfer_proxy.exe`

`taskkill /im npu_transfer_proxy.exe /f`


## base
### env snippet
``` bat
@set PY=E:\ProgramData\Anaconda3\envs\zal_rk140
@set PATH=%PATH%;%PY%\Library\bin;%PY%\Library/bin;%PY%\Lib\site-packages\~knn\api\lib\hardware\Windows_x64;%PY%\Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64;

python -m rknn.bin.visualization
```