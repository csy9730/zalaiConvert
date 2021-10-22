# caffe layer error


## question

```
(rk161) H:\tmp\openpose>python -m zalaiConvert.convert.onnx2rknn 1/pose_iter_440000.caffemodel --darknet-cfg 1/pose_deploy_linevec.prototxt -o abc.rknn --framework caffe --dataset 1.txt --normalize-params 0 0 0 255
--> config model
done
--> Loading model
E Deprecated caffe input usage, please change it to input layer.
E Catch exception when loading caffe model: 1/pose_deploy_linevec.prototxt!
E Traceback (most recent call last):
E   File "rknn\api\rknn_base.py", line 261, in rknn.api.rknn_base.RKNNBase.load_caffe
E   File "rknn\base\RKNNlib\RK_nn.py", line 72, in rknn.base.RKNNlib.RK_nn.RKnn.load_caffe
E   File "rknn\base\RKNNlib\converter\caffeloader.py", line 1099, in rknn.base.RKNNlib.converter.caffeloader.CaffeLoader.load
E   File "rknn\base\RKNNlib\converter\caffeloader.py", line 852, in rknn.base.RKNNlib.converter.caffeloader.CaffeLoader.parse_net_param
E   File "rknn\api\rknn_log.py", line 312, in rknn.api.rknn_log.RKNNLog.e
E ValueError: Deprecated caffe input usage, please change it to input layer.
Load model failed!
```
### 原因

该模型是openpose的posenet，该 prototxt格式是旧的
### 方法
需要修改`pose_deploy_linevec.prototxt`文件。

开头几行的内容如下：
```
input: "image"
input_dim: 1
input_dim: 3
input_dim: 1 # This value will be defined at runtime
input_dim: 1 # This value will be defined at runtime

layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "image"
  top: "conv1_1"
  ...
```

把第一个input layer改成：
``` 
layer {
  name: "data"
  type: "Input"
  top: "image"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 224
      dim: 224
    }
  }
}

layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "image"
  top: "conv1_1"
  ...
```

## misc
[https://t.rock-chips.com/forum.php?mod=viewthread&tid=93&extra=&page=2](https://t.rock-chips.com/forum.php?mod=viewthread&tid=93&extra=&page=2)
