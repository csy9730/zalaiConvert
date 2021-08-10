# RKNN error

## rknn


### RKNN init failed. error code: RKNN_ERR_DEVICE_UNAVAILABLE
[无法开启NPU，推理报错](http://t.rock-chips.com/forum.php?mod=viewthread&tid=175)

```
0123456789ABCDEF
*************************
None
E NPUTransfer: Cannot connect to proxy: cannot connect to 127.0.0.1:11808: Unknown error
E RKNNAPI: rknn_init,  driver open fail!  ret = -4!
E Catch exception when init runtime!
E Traceback (most recent call last):
E   File "rknn\api\rknn_base.py", line 1154, in rknn.api.rknn_base.RKNNBase.init_runtime
E   File "rknn\api\rknn_runtime.py", line 356, in rknn.api.rknn_runtime.RKNNRuntime.build_graph
E Exception: RKNN init failed. error code: RKNN_ERR_DEVICE_UNAVAILABLE
E Current device id is: None
E Devices connected:
E ['0123456789ABCDEF']
Init runtime environment failed
```

设备断电，重启设备，usb设备重新插拔。

### RKNN_ERR_MODEL_INVALID

```
E Exception: RKNN init failed. error code: RKNN_ERR_MODEL_INVALID
```
### Init runtime environment failed!

```
(rk_1_4) ➜  2 ./runs.sh
--> config model
done
--> Loading model
/home/zal/anaconda3/envs/rk_1_4/lib/python3.6/site-packages/onnx_tf/common/__init__.py:87: UserWarning: FrontendHandler.get_outputs_names is deprecated. It will be removed in future release.. Use node.outputs instead.
  warnings.warn(message)
done
--> Building model
W The target_platform is not set in config, using default target platform rk1808.
W The channel_mean_value filed will not be used in the future!
done
E Using device with adb mode to init runtime, but npu_transfer_proxy is running, it may cause conflict. Please terminate npu_transfer_proxy first.
E Catch exception when init runtime!
E Traceback (most recent call last):
E   File "rknn/api/rknn_base.py", line 1128, in rknn.api.rknn_base.RKNNBase.init_runtime
E   File "rknn/api/rknn_runtime.py", line 168, in rknn.api.rknn_runtime.RKNNRuntime.__init__
E   File "rknn/api/rknn_platform_utils.py", line 296, in rknn.api.rknn_platform_utils.start_ntp_or_adb
E Exception: Init runtime environment failed!
E Current device id is: None
E Devices connected:
E ['4486dcfc35a505c0']
Init runtime environment failed
(rk_1_4) ➜  2 adb devices
```
问题分析：
npu_transfer_proxy 是pc端windows/linux的npu工具，rk板子也没这玩意。

``` bash
# 查看 pid
ps -ef |grep npu_transfer_proxy

# 杀死pid
kill -9 xxx 
```

## onnx


### shape error
input shape is different when onnx convert to rknn #66
[https://github.com/rockchip-linux/rknn-toolkit/issues/66](https://github.com/rockchip-linux/rknn-toolkit/issues/66)

shape显示的顺序反了。实际没问题。

### rknn onnx Upsample

```
D Save log info to: C:\Windows\Temp\0\RKNN_toolkit.log

--> Config model


--> Loading onnx model

No module named 'decorator'
D Save log info to: C:\Windows\Temp\0\RKNN_toolkit.log

--> Config model


--> Loading onnx model

E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\onnx_tf\common\__init__.py:87: UserWarning: FrontendHandler.get_outputs_names is deprecated. It will be removed in future release.. Use node.outputs instead.
  warnings.warn(message)
I Current ONNX Model use ir_version 4 opset_version 9
D import clients finished
I build output layer attach_Conv_3458:out0
I Try match Conv_3458:out0
I Match r_conv [['Conv_3458', 'Initializer_head.3.weight', 'Initializer_head.3.bias']] [['Conv', 'Constant_0', 'Constant_1']] to
[['convolution']]
I Try match Relu_3457:out0
I Match r_relu [['Relu_3457']] [['Relu']] to [['relu']]
I Try match BatchNormalization_3456:out0
I Match r_bn_v6 [['BatchNormalization_3456', 'Initializer_head.1.weight', 'Initializer_head.1.bias', 'Initializer_head.1.running_mean', 'Initializer_head.1.running_var']] [['BatchNormalization', 'Constant_0', 'Constant_1', 'Constant_2', 'Constant_3']] to [['batchnormalize']]
I Try match Conv_3455:out0
I Match r_conv [['Conv_3455', 'Initializer_head.0.weight', 'Initializer_head.0.bias']] [['Conv', 'Constant_0', 'Constant_1']] to
[['convolution']]
I Try match Concat_3454:out0
I Match concat_4 [['Concat_3454']] [['Concat']] to [['concat']]
I Try match Relu_3321:out0
I Match r_relu [['Relu_3321']] [['Relu']] to [['relu']]
I Try match Upsample_3431:out0
W Not match tensor Upsample_3431:out0
E Try match Upsample_3431:out0 failed, catch exception!
W ----------------Warning(1)----------------
E Catch exception when loading onnx model: H:\Project\Github\hrnet_facial_landmark\abc2.onnx!
E Traceback (most recent call last):
E   File "rknn\base\RKNNlib\converter\convert_onnx.py", line 1130, in rknn.base.RKNNlib.converter.convert_onnx.convert_onnx.match_paragraph_and_param
E   File "rknn\base\RKNNlib\converter\convert_onnx.py", line 1039, in rknn.base.RKNNlib.converter.convert_onnx.convert_onnx._onnx_push_ready_tensor
E TypeError: 'NoneType' object is not iterable
E During handling of the above exception, another exception occurred:
E Traceback (most recent call last):
E   File "rknn\api\rknn_base.py", line 559, in rknn.api.rknn_base.RKNNBase.load_onnx
E   File "rknn\base\RKNNlib\converter\convert_onnx.py", line 1136, in rknn.base.RKNNlib.converter.convert_onnx.convert_onnx.match_paragraph_and_param
E   File "rknn\api\rknn_log.py", line 312, in rknn.api.rknn_log.RKNNLog.e
E ValueError: Try match Upsample_3431:out0 failed, catch exception!
--> Load onnx model failed!

***********************close window 0***********************

***********************close server***********************

```



## tensorflow
### OutOfRangeError FIFOQueue 

报错：OutOfRangeError: FIFOQueue '_1_batch/fifo_queue' is closed and has insufficient elements 解决办法

木头VS星星 2018-08-09 16:51:02  11569  收藏
分类专栏： 基础知识 文章标签： Tensorflow OutofRangeError Python
版权
  Tensorflow在跑图像数据时报错，报错信息如下：

OutOfRangeError (see above for traceback): FIFOQueue '_1_batch/fifo_queue' is closed and has insufficient elements (requested 8, current size 1)
	 [[Node: batch = QueueDequeueManyV2[component_types=[DT_FLOAT, DT_UINT8], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/device:CPU:0"](batch/fifo_queue, batch/n)]]
  信息显示读取数据队列出现问题，刚开始一直以为是  tf.train.shuffle_batch参数出现问题，于是不断调参数，但均无效。

```
Exception in thread Thread-2:
Traceback (most recent call last):
  File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\threading.py", line 916, in _bootstrap_inner
    self.run()
  File "rknn\base\acuitylib\provider\queue_provider.py", line 98, in rknn.base.acuitylib.provider.queue_provider.QueueProvider.run
  File "rknn\base\acuitylib\provider\queue_provider.py", line 102, in rknn.base.acuitylib.provider.queue_provider.QueueProvider.run
  File "rknn\base\acuitylib\provider\text_provider.py", line 65, in rknn.base.acuitylib.provider.text_provider.TextProvider.get_batch
  File "rknn\base\acuitylib\provider\base_provider.py", line 163, in rknn.base.acuitylib.provider.base_provider.BaseProvider._tensor_list_to_batch
  File "rknn\base\acuitylib\provider\base_provider.py", line 105, in rknn.base.acuitylib.provider.base_provider.BaseProvider._reshape_batches
  File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\numpy\core\fromnumeric.py", line 292, in reshape
    return _wrapfunc(a, 'reshape', newshape, order=order)
  File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\numpy\core\fromnumeric.py", line 56, in _wrapfunc
    return getattr(obj, method)(*args, **kwds)
ValueError: cannot reshape array of size 262144 into shape (1,256,256,3)

E Traceback (most recent call last):
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\client\session.py", line 1292, in _do_call
E     return fn(*args)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\client\session.py", line 1277, in _run_fn
E     options, feed_dict, fetch_list, target_list, run_metadata)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\client\session.py", line 1367, in _call_tf_sessionrun
E     run_metadata)
E tensorflow.python.framework.errors_impl.OutOfRangeError: FIFOQueue '_0_fifo_queue' is closed and has insufficient elements (requested 1, current size 0)
E        [[{{node fifo_queue_Dequeue}} = QueueDequeueV2[component_types=[DT_FLOAT], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/device:CPU:0"](fifo_queue)]]
E During handling of the above exception, another exception occurred:
E Traceback (most recent call last):
E   File "rknn\api\rknn_base.py", line 950, in rknn.api.rknn_base.RKNNBase.hybrid_quantization_step1
E   File "rknn\api\rknn_base.py", line 1797, in rknn.api.rknn_base.RKNNBase._quantize2
E   File "rknn\base\RKNNlib\app\medusa\quantization.py", line 105, in rknn.base.RKNNlib.app.medusa.quantization.Quantization.run
E   File "rknn\base\RKNNlib\app\medusa\quantization.py", line 44, in rknn.base.RKNNlib.app.medusa.quantization.Quantization._run_quantization
E   File "rknn\base\RKNNlib\app\medusa\workspace.py", line 145, in rknn.base.RKNNlib.app.medusa.workspace.Workspace.run
E   File "rknn\base\RKNNlib\app\medusa\workspace.py", line 126, in rknn.base.RKNNlib.app.medusa.workspace.Workspace._run_iteration
E   File "rknn\base\RKNNlib\RKNN_session.py", line 30, in rknn.base.RKNNlib.RKNN_session.RKNNSession.run
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\client\session.py", line 887, in run
E     run_metadata_ptr)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\client\session.py", line 1110, in _run
E     feed_dict_tensor, options, run_metadata)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\client\session.py", line 1286, in _do_run
E     run_metadata)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\client\session.py", line 1308, in _do_call
E     raise type(e)(node_def, op, message)
E tensorflow.python.framework.errors_impl.OutOfRangeError: FIFOQueue '_0_fifo_queue' is closed and has insufficient elements (requested 1, current size 0)
E        [[{{node fifo_queue_Dequeue}} = QueueDequeueV2[component_types=[DT_FLOAT], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/device:CPU:0"](fifo_queue)]]
E Caused by op 'fifo_queue_Dequeue', defined at:
E   File "<string>", line 1, in <module>
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\multiprocessing\spawn.py", line 105, in spawn_main
E     exitcode = _main(fd)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\multiprocessing\spawn.py", line 118, in _main
E     return self._bootstrap()
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\multiprocessing\process.py", line 258, in _bootstrap
E     self.run()
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\multiprocessing\process.py", line 93, in run
E     self._target(*self._args, **self._kwargs)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\rknn\visualization\server\rknn_func.py", line 261, in start_quantization_convert
E     ret = rknn.hybrid_quantization_step1(dataset=dataset)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\rknn\api\rknn.py", line 277, in hybrid_quantization_step1
E     ret = self.rknn_base.hybrid_quantization_step1(dataset=dataset)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\ops\data_flow_ops.py", line 433, in dequeue
E     self._queue_ref, self._dtypes, name=name)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\ops\gen_data_flow_ops.py", line 4097, in queue_dequeue_v2
E     timeout_ms=timeout_ms, name=name)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 787, in _apply_op_helper
E     op_def=op_def)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\util\deprecation.py", line 488, in new_func
E     return func(*args, **kwargs)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\framework\ops.py", line 3272, in create_op
E     op_def=op_def)
E   File "E:\ProgramData\Anaconda3\envs\zal_rk140\lib\site-packages\tensorflow\python\framework\ops.py", line 1768, in __init__
E     self._traceback = tf_stack.extract_stack()
E OutOfRangeError (see above for traceback): FIFOQueue '_0_fifo_queue' is closed and has insufficient elements (requested 1, current size
0)
E        [[{{node fifo_queue_Dequeue}} = QueueDequeueV2[component_types=[DT_FLOAT], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/device:CPU:0"](fifo_queue)]]
--> Model process failed!
```


```python

tf.train.shuffle_batch(
      tensors,
      batch_size = 16,
      capacity = 512,  # 这个参数多次调整，无效
      min_after_dequeue = 128,  # 这个参数多次调整，无效
      keep_input=True,
      num_threads=num_threads,  # 这个参数调整多次，无效
      seed=seed,
      enqueue_many=enqueue_many,
      shapes=shapes,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)
```

  最终解决方法：原始数据问题，存在一张异常照片，删除即可。
  最后在检查数据的时候，发现原始数据中存在一张异常照片，删除即恢复正常。

  所以，当遇到类似问题时，请先检查原始数据，可能并不是程序本身的错误。


### tensorflow.math.tanh 
```
G:\wsl\Project\M808_arith\project\rknnWrap\yolov4_test>bash2 runs.sh
--> config model
done
--> Loading model
done
--> Building model
W The target_platform is not set in config, using default target platform rk1808.
W The channel_mean_value filed will not be used in the future!
E Catch exception when building RKNN model!
E Traceback (most recent call last):
E   File "rknn\api\rknn_base.py", line 895, in rknn.api.rknn_base.RKNNBase.build
E   File "rknn\api\rknn_base.py", line 1797, in rknn.api.rknn_base.RKNNBase._quantize2
E   File "rknn\base\RKNNlib\app\medusa\quantization.py", line 105, in rknn.base.RKNNlib.app.medusa.quantization.Quantization.run
E   File "rknn\base\RKNNlib\app\medusa\quantization.py", line 44, in rknn.base.RKNNlib.app.medusa.quantization.Quantization._run_quantization
E   File "rknn\base\RKNNlib\app\medusa\workspace.py", line 139, in rknn.base.RKNNlib.app.medusa.workspace.Workspace.run
E   File "rknn\base\RKNNlib\app\medusa\workspace.py", line 109, in rknn.base.RKNNlib.app.medusa.workspace.Workspace._setup_graph
E   File "rknn\base\RKNNlib\app\medusa\workspace.py", line 110, in rknn.base.RKNNlib.app.medusa.workspace.Workspace._setup_graph
E   File "rknn\base\RKNNlib\RKNNnetbuilder.py", line 274, in rknn.base.RKNNlib.RKNNnetbuilder.RKNNNetBuilder.build
E   File "rknn\base\RKNNlib\RKNNnetbuilder.py", line 278, in rknn.base.RKNNlib.RKNNnetbuilder.RKNNNetBuilder.build
E   File "rknn\base\RKNNlib\RKNNnetbuilder.py", line 305, in rknn.base.RKNNlib.RKNNnetbuilder.RKNNNetBuilder.build_layer
E   File "rknn\base\RKNNlib\RKNNnetbuilder.py", line 305, in rknn.base.RKNNlib.RKNNnetbuilder.RKNNNetBuilder.build_layer
E   File "rknn\base\RKNNlib\RKNNnetbuilder.py", line 305, in rknn.base.RKNNlib.RKNNnetbuilder.RKNNNetBuilder.build_layer
E   [Previous line repeated 52 more times]
E   File "rknn\base\RKNNlib\RKNNnetbuilder.py", line 331, in rknn.base.RKNNlib.RKNNnetbuilder.RKNNNetBuilder.build_layer
E   File "rknn\base\RKNNlib\RKNNnetbuilder.py", line 336, in rknn.base.RKNNlib.RKNNnetbuilder.RKNNNetBuilder.build_layer
E   File "rknn\base\RKNNlib\layer\RKNNlayer.py", line 287, in rknn.base.RKNNlib.layer.RKNNlayer.RKNNLayer.compute_tensor
E   File "rknn\base\RKNNlib\layer\mish.py", line 28, in rknn.base.RKNNlib.layer.mish.MishRelu.compute_out_tensor
E AttributeError: module 'tensorflow.math' has no attribute 'tanh'
Build model failed!
```

查看`lib\site-packages\tensorflow\math\__init__.py`, 可以发现确实没有`tanh`。

查看`tensorflow\__init__.py`，可以发现
```
from tensorflow.python import tan
from tensorflow.python import tanh
```

修改`tensorflow\math\__init__.py`, 添加 上面的语句。
