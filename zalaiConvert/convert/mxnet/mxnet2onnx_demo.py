import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

def main():
    path='http://data.mxnet.io/models/imagenet/'
    sym = 'resnet-18-symbol.json'
    params = 'resnet-18-0000.params'
    # sym = 'resnet-50-symbol.json'
    # params = 'resnet-50-0000.params'
    [
        mx.test_utils.download(path+'resnet/18-layers/%s' % params),
        mx.test_utils.download(path+'resnet/18-layers/%s' % sym),
        mx.test_utils.download(path+'synset.txt')
    ]

    # 下载的输入符号和参数文件

    # 标准Imagenet输入- 3通道，224*224
    input_shape = (1,3,224,224)

    # 输出文件的路径
    onnx_file = './mxnet_exported_resnet50.onnx'


    # 调用导出模型API。它返回转换后的onnx模型的路径
    converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)

if __name__ == "__main__":
    main()

"""

(zal_pytorch140) E:\nnCollect\zalaiConvert\tmp>python e:\nnCollect\zalaiConvert\zalaiConvert\convert\mxnet\1.py
[19:29:24] C:\Jenkins\workspace\mxnet-tag\mxnet\src\nnvm\legacy_json_util.cc:209: Loading symbol saved by previous version v0.8.0. Attempting to upgrade...
[19:29:24] C:\Jenkins\workspace\mxnet-tag\mxnet\src\nnvm\legacy_json_util.cc:217: Symbol successfully upgraded!
Traceback (most recent call last):
  File "e:\nnCollect\zalaiConvert\zalaiConvert\convert\mxnet\1.py", line 30, in <module>
    main()
  File "e:\nnCollect\zalaiConvert\zalaiConvert\convert\mxnet\1.py", line 27, in main
    converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch140\lib\site-packages\mxnet\contrib\onnx\mx2onnx\export_model.py", line 79, in export_model
    verbose=verbose)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch140\lib\site-packages\mxnet\contrib\onnx\mx2onnx\export_onnx.py", line 308, in create_onnx_graph_proto
    checker.check_graph(graph)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch140\lib\site-packages\onnx\checker.py", line 58, in checker
    proto.SerializeToString(), ctx)
onnx.onnx_cpp2py_export.checker.ValidationError


(zal_pytorch140) E:\nnCollect\zalaiConvert\tmp>python e:\nnCollect\zalaiConvert\zalaiConvert\convert\mxnet\1.py
[19:34:32] C:\Jenkins\workspace\mxnet-tag\mxnet\src\nnvm\legacy_json_util.cc:209: Loading symbol saved by previous version v0.8.0. Attempting to upgrade...
[19:34:32] C:\Jenkins\workspace\mxnet-tag\mxnet\src\nnvm\legacy_json_util.cc:217: Symbol successfully upgraded!
Traceback (most recent call last):
  File "e:\nnCollect\zalaiConvert\zalaiConvert\convert\mxnet\1.py", line 30, in <module>
    main()
  File "e:\nnCollect\zalaiConvert\zalaiConvert\convert\mxnet\1.py", line 27, in main
    converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch140\lib\site-packages\mxnet\contrib\onnx\mx2onnx\export_model.py", line 79, in export_model
    verbose=verbose)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch140\lib\site-packages\mxnet\contrib\onnx\mx2onnx\export_onnx.py", line 308, in create_onnx_graph_proto
    checker.check_graph(graph)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch140\lib\site-packages\onnx\checker.py", line 52, in checker
    proto.SerializeToString(), ctx)
onnx.onnx_cpp2py_export.checker.ValidationError: Unrecognized attribute: spatial for operator BatchNormalization

==> Context: Bad node spec: input: "data" input: "bn_data_gamma" input: "bn_data_beta" input: "bn_data_moving_mean" input: "bn_data_moving_var" output: "bn_data" name: "bn_data" op_type: "BatchNormalization" attribute { name: "epsilon" f: 2e-05 type: FLOAT } attribute { name: "momentum" f: 0.9 type: FLOAT } attribute { name: "spatial" i: 0 type: INT }

"""