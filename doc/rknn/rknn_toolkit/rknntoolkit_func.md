# rknntoolkit

```
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'accuracy_analysis', 'build', 'config', 'direct_build', 'eval_memory', 'eval_perf', 'export_rknn', 'export_rknn_precompile_model', 'export_rknn_sync_model', 'fetch_rknn_model_config', 'get_sdk_version', 'hybrid_quantization_step1', 'hybrid_quantization_step2', 'inference', 'init_runtime', 'list_devices', 'list_support_target_platform', 'load_caffe', 'load_darknet', 'load_firmware', 'load_keras', 'load_mxnet', 'load_onnx', 'load_pytorch', 'load_rknn', 'load_tensorflow', 'load_tflite', 'register_op', 'release', 'rknn_base', 'rknn_log', 'target', 'verbose']
```



### function

'list_devices', 
'list_support_target_platform', 



#### main

- build
- config
- inference
- eval_memory
- eval_perf
- 'export_rknn
- export_rknn_precompile_model

##### runtime

- get_sdk_version
- init_runtime

##### load

- load_caffe
- load_darknet
- oad_firmware
- load_keras
- load_mxnet
- load_onnx
- load_pytorch
- load_rknn
- load_tensorflow
- load_tflite

##### release

release

##### misc

- accuracy_analysis
- direct_build
- export_rknn_sync_model
- hybrid_quantization_step1
- hybrid_quantization_step2
- register_op



##### fetch_rknn_model_config

```
>>> rknn.fetch_rknn_model_config(model)
OrderedDict([('target_platform', ['RK1808']), ('network_platform', 'caffe'), ('ori_network_platform', 'caffe'), ('input_fmt', 0), ('input_transpose', 1), ('name', 'Pfldc469Bestpth'), ('version', '1.3.2'), ('ovxlib_version', '1.1.12'), ('node_num', 1), ('norm_tensor_num', 2), ('const_tensor_num', 0), ('virtual_tensor_num', 1), ('optimization_level', 3), ('mean_value', [0.0, 0.0, 0.0, 256.0]), ('mean_value_chns', [4]), ('reorder', [0, 1, 2]), ('input_num', 1), ('output_num', 1), ('pre_compile', 1), ('pre_compile_version', 1), ('case_type', 2), ('nodes', [OrderedDict([('lid', 'nbg_0'), ('name', 'nbg'), ('op', 'VSI_NN_OP_NBG'), ('uid', 0), ('input_num', 1), ('output_num', 1), ('nn', OrderedDict([('nbg', OrderedDict([('type', 'VSI_NN_NBG_FILE'), ('url', 'data_file_name')]))]))])]), ('norm_tensor', [OrderedDict([('tensor_id', 0), ('size', [8, 1]), ('dim_num', 2), ('dtype', OrderedDict([('scale', 0.003892257), ('zero_point', 2), ('qnt_type', 'VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC'), ('vx_type', 'VSI_NN_TYPE_UINT8')]))]), OrderedDict([('tensor_id', 1), ('size', [112, 112, 3, 1]), ('dim_num', 4), ('dtype', OrderedDict([('scale', 0.00390625), ('zero_point', 0), ('qnt_type', 'VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC'), ('vx_type', 'VSI_NN_TYPE_UINT8')]))])]), ('const_tensor', []), ('virtual_tensor', []), ('connection', [OrderedDict([('node_id', 0), ('left', 'input'), ('left_tensor_id', 0), ('right_tensor', OrderedDict([('type', 'norm_tensor'), ('tensor_id', 1)]))]), OrderedDict([('node_id', 0), ('left', 'output'), ('left_tensor_id', 0), ('right_tensor', OrderedDict([('type', 'norm_tensor'), ('tensor_id', 0)]))])]), ('graph', [OrderedDict([('left', 'output'), ('left_tensor_id', 0), ('right', 'norm_tensor'), ('right_tensor_id', 0)]), OrderedDict([('left', 'input'), ('left_tensor_id', 0), ('right', 'norm_tensor'), ('right_tensor_id', 1)])])])
```

### attribute

#### normal attribute

- target
- verbose

#### class attribute

rknn_log

rknn.rknn_base
<rknn.api.rknn_base.RKNNBase object at 0x7f6670166fd0>



## misc

['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 


