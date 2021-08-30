# todo

## todo

- [ ] add: 
- [ ] add: 添加反馈 : parse export success!
- [ ] convertWrap2::解析 fps
- [ ] 调整rknn.log
- [ ] tune: adb detect rknn devices
- [x] add: `python3 -m zalaiConvert.convertWrap -c foo.json`
- [x] add：在哪里内置adb程序？


**Q**: add adb to whl or add to qml?

**A**: adb程序很小，直接内置wheel进去。


## todo2

- [x] add: git
- [x] add: requirements.txt
- [x] add: setup.py
- [ ] add: logger
- [ ] add: listDevice
- [x] add: `adb devices` 
- [x] `adb shell`
- [x] `series ssh shell` 切换成adb模式
` python -c "from rknn.api import RKNN;print(RKNN().list_devices())" `
- [ ] bug: `RKNN().list_devices()` 无法发现rknn
- [ ] 
- [x] add: zalai::stdout
- [ ] rm : tmp.t
`from rknn.api.rknn_base import RKNNBase;RKNNBase().list_devices()`


## 20210814 
- [x] add: onnx2rknn:: --input-size-list
- [ ] add: flask + rknn, web api
- [ ] merge: yolov3 yolov4
- [ ] add: cameraViewer
- [ ] 


```
python -c "import sys;import os;os.environ['PATH'] = ';'.join([r'H:\project\mylib\py_misc_project', os.environ['PATH']]);os.system('cmd')"          
```