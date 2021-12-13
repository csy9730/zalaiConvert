# readme

zalaiConvert 是基于RKNN_TOOLKIT的封装的应用脚本，工具集。


## install

安装方法：

``` bash
pip install git+https://github.com/csy9730/zalaiConvert.git  --user
pip install git+git://github.com/csy9730/zalaiConvert.git  --user
```

## demo
### ntb device uti

#### found ntb device

windows系统下，连接adb设备，并切换成ntb设备。
```
python -m zalaiConvert list

```

### convert network to rknn
#### darknet/yolov4 转换成rknn
``` bash
python -m zalaiConvert.convert.onnx2rknn yolov4_mirror_best.weights --darknet-cfg yolov4.cfg -o abc.rknn --framework darknet --dataset dataset.txt --normalize-params 0 0 0 255

python -m zalaiConvert.convert.onnx2rknn yolov4_mirror_best.weights --darknet-cfg yolov4.cfg -o abc_q.rknn --framework darknet --dataset dataset.txt --normalize-params 0 0 0 255 --do-quantization
```

### rknn forward

#### 图像分类
基于imagenet 1000类别的mobilenet的分类网络。
``` bash
python -m zalaiConvert.farward.rknn_classify_farward mobile.rknn -i test.jpg -o out_q.jpg  --name-file class.txt 
```

#### 目标检测
基于yolov3 的目标检测
```
python -m zalaiConvert.farward.rknn_yolo_farward abc_q.rknn -i test.jpg -o out_q.jpg --network yolov4.cfg --name-file class.txt --device rk1808
```
