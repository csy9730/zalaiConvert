# readme

zalaiConvert 是 基于RKNN_TOOLKIT的封装的应用脚本，工具集。

## install
```
pip install git+https://github.com/csy9730/zalaiConvert.git  --user
pip install git+git://github.com/csy9730/zalaiConvert.git  --user
```

## demo
### ntb device found

windows系统下，连接adb设备，并切换成ntb设备。
```
python -m zalaiConvert list

```

### convert network to rknn
``` bash
python -m zalaiConvert.convert.onnx2rknn yolov4_mirror_best.weights --darknet-cfg yolov4.cfg -o abc.rknn --framework darknet --dataset dataset.txt --normalize-params 0 0 0 255

python -m zalaiConvert.convert.onnx2rknn yolov4_mirror_best.weights --darknet-cfg yolov4.cfg -o abc_q.rknn --framework darknet --dataset dataset.txt --normalize-params 0 0 0 255 --do-quantization
```

### rknn farward
```
python -m zalaiConvert.farward.rknn_yolo_farward abc_q.rknn -i test.jpg -o out_q.jpg --network yolov4.cfg --name-file class.txt --device rk1808
```