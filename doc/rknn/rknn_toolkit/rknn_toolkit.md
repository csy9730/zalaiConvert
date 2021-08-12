# rknntoolkit

## install



github仓库网址：[rknn-toolkit](https://github.com/rockchip-linux/rknn-toolkit)

github仓库附件下载网址：[rknn-toolkit/release](https://github.com/rockchip-linux/rknn-toolkit/releases) 

2021/8/8的rknn最新版本是1.6.1

``` bash
conda create -n rk161 python=3.6.12
conda activate rk161

pip3 install rknn_toolkit-1.6.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision==0.6.1 torch==1.5.1
pip3 install tensorflow==1.11.0
pip3 install tqdm

```

## eval/run
### windows 

Windows 平台并不提供 NPU 模拟器，所以在 Windows 平台上必须接 Rockchip NPU 设备
才可以使用推理 性能评估 内存评估 等功能 。

> RKNN Toolkit does not support exporting precompiled rknn models from Windows and MacOS via the build() interface. The precompiled model needs to be exported via the interface export_rknn_precompile_model(). This interface requires a connection to the RK1808 device. (Note that RK3399pro does not support this feature)

> Windows 和 MacOS 操作系统不支持直接通过 build() 预编译模型。
> Windows 和 MacOS 操作系统上，需要连接 RK1808设备，使用 export_rknn_precompile_model() 才能导出预编译模型。注意，RK3399pro不支持该特征。



### eval memory

#### demo
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

### 精度分析功能



#### rknn.accuracy_analysis

```
(256, 256, 3)
--> Loading model
Load done
Init runtime environment
--> Begin _analysis model accuracy
print('--> Begin _analysis model accuracy')
        perf_ana = rknn.accuracy_analysis(inputs="1.txt")
E Catch exception when snapshot: AttributeError("'Namespace' object has no attribute 'can_snapshot'",)
E Traceback (most recent call last):
E   File "rknn/api/rknn_base.py", line 774, in rknn.api.rknn_base.RKNNBase.accuracy_analysis
E AttributeError: 'Namespace' object has no attribute 'can_snapshot'
-1
done
time:  98.67426705360413
pred [array([[[[-3.96728516e-04, -3.09228897e-04,  6.60896301e-04, ...,
          -1.82056427e-03, -1.09481812e-03, -1.15108490e-03],
         [-9.49859619e-04, -6.86168671e-04,  1.34563446e-03, ...,
          -1.38378143e-03, -6.60896301e-04, -6.57081604e-04],
```


#### rknn.accuracy_analysis 2
```
perf_ana = rknn.accuracy_analysis(inputs="1.txt")

--> Building model
W The target_platform is not set in config, using default target platform rk1808.
W The channel_mean_value filed will not be used in the future!
done
--> Export RKNN model
--> Begin _analysis model accuracy
E snapshot must call after quantize build when the original model is a float model.
W ----------------Warning(2)----------------
E Catch exception when snapshot: ValueError('snapshot must call after quantize build when the original model is a float model.',)
E Traceback (most recent call last):
E   File "rknn/api/rknn_base.py", line 1074, in rknn.api.rknn_base.RKNNBase.snapshot
E   File "rknn/api/rknn_log.py", line 312, in rknn.api.rknn_log.RKNNLog.e
E ValueError: snapshot must call after quantize build when the original model is a float model.
-1
```

### 用时分析

## misc

模拟器是完美实现，代价是运行速度极慢，随便跑个模型都要20多秒。
设备运行就不同了，运行速度极快，但是结果可能和模拟器一致，也可能和模拟器结果完全不同。

rockchip每次更新驱动/运行时版本，就是使模拟器和设备结果一致。
