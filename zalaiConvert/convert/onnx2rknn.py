import os
import sys
import json
import numpy as np
import cv2

from rknn.api import RKNN


if os.name == "nt":
    PY = os.path.dirname(os.path.abspath(sys.executable))
    os.environ['PATH'] = ";".join([
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
        os.path.join(PY, r"Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64"),
        os.path.join(PY, r"Lib\site-packages\~knn\api\lib\hardware\Windows_x64"),
        os.path.join(PY, r"Library/bin"),
        os.environ.get('PATH')
    ])

def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rknn converter')
    parser.add_argument('model', help='model file path')
    parser.add_argument('--output', '-o', default='out.rknn', help='save output image name')
    parser.add_argument('--config', '-c')

    parser.add_argument('--device', choices=['rk1808', 'rv1126'], help='device: rk1808, rv1126')
    parser.add_argument('--do-quantization', action='store_true', help='model file path')
    parser.add_argument('--pre-compile', action='store_true', help='model file path')

    parser.add_argument('--framework', help='model framework')
    parser.add_argument('--darknet-cfg')

    parser.add_argument('--dataset', help='a txt file contain image paths')

    parser.add_argument('--verbose', action='store_true', help='verbose information')
    parser.add_argument('--rknn-logfile') # "rknn.log"

    parser.add_argument('--rgb-reorder', action='store_true')
    parser.add_argument('--normalize-params', nargs='*')
    parser.add_argument('--quantized-algorithm', default='normal')

    parser.add_argument('--epochs', type=int, default=-1)

    parser.add_argument('--use-accanalyze', action='store_true')
    parser.add_argument('--use-farward', action='store_true')


    return parser.parse_args(cmds)

def model2Rknn(model, output, dataset, framework='onnx', **kwargs):
    if framework == 'onnx':
        return onnxmodel2Rknn(model, output, dataset, framework='onnx', **kwargs)
    elif framework == 'darknet':
        return onnxmodel2Rknn(model, output, dataset, framework='darknet', **kwargs)
    elif framework == 'pytorch':
        return onnxmodel2Rknn(model, output, dataset, framework='pytorch', **kwargs)
    elif framework == 'tensorflow':
        return onnxmodel2Rknn(model, output, dataset, framework='tensorflow', **kwargs)
    elif framework in ['caffee', 'tflite', 'mxnet']:
        return onnxmodel2Rknn(model, output, dataset, framework=framework, **kwargs)


def onnxmodel2Rknn(model, output, dataset, do_quantization=False, pre_compile=False, \
    verbose=None, normalize_params=None, device=None, epochs=-1, 
    log_file=None, framework='onnx', quantized_algorithm='normal',
    **kwargs):
    if normalize_params is None:
        normalize_params = ['0', '0', '0', '1']
    # Create RKNN object
    rknn = RKNN(verbose=verbose, verbose_file=log_file)
    
    # pre-process config
    print('--> config model')    
    rknn.config(channel_mean_value=' '.join(normalize_params), reorder_channel='0 1 2',\
        epochs=epochs, quantized_algorithm=quantized_algorithm)    
    # , target_platform=[target]
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    if framework == "pytorch":
        ret = rknn.load_pytorch(model=model,input_size_list=[[3, 512, 512]])
    elif framework == "onnx":
        ret = rknn.load_onnx(model=model)
    elif framework == "tensorflow":
        rknn.load_tensorflow(tf_pb = model,
            inputs = input_node,
            outputs = output_node,
            input_size_list=[[image_size, image_size, image_channel]])
    elif framework == "tflite":
        ret = rknn.load_tflite(model=model)
    elif framework == "darknet":
        ret = rknn.load_darknet(model=kwargs["darknet_cfg"], weight=model)
    elif framework == "caffe":
        ret = rknn.load_caffe(model=kwargs["darknet_cfg"], proto="caffe", blobs=model)
    elif framework == "mxnet":
        ret = rknn.load_mxnet(model=kwargs["darknet_cfg"], params=model, input_size_list=[[image_channel, image_size, image_size]])
    else:
        print("not a framework", framework)
        return -1

    if ret != 0:
        print('Load model failed!')
        return ret
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset, pre_compile=pre_compile)
    if ret != 0:
        print('Build model failed!')
        return ret
    print('done')

    if device:
        ret = rknn.init_runtime(target=device, eval_mem=False, rknn2precompile=pre_compile)
    if ret != 0:
        print('Init runtime environment failed')
        return ret

    # Export rknn model
    print('--> Export RKNN model')
    if os.name == "nt" and pre_compile:
        ret = rknn.export_rknn_precompile_model(output)
    else:
        ret = rknn.export_rknn(output)
    if ret != 0:
        print('Export mobilenet_v1.rknn failed!')
        return ret

    if kwargs.get("use_evalmemory"):
        memory_detail = rknn.eval_memory()
        print(memory_detail)

    if kwargs.get("use_accanalyze"):
        print('--> Begin analysis model accuracy')
        perf_ana = rknn.accuracy_analysis(inputs=dataset, target=device)
        print(perf_ana)



    print('done')
    rknn.release()
    return 0

def farward(img_path, device=None, **kwargs):
    rknn = RKNN()

    im = cv2.imread(img_path)
    im = cv2.resize(im, (image_size, image_size))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ret = rknn.load_rknn(output_path)
    ret = rknn.init_runtime(target=device, eval_mem=True)
    if ret!=0:
        print('Init runtime environment failed')
        exit(ret)
    outputs = rknn.inference(inputs=[im])
    print(type(outputs), outputs[0])

    def softmax(x):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis = 1, keepdims = True)
        s = x_exp / x_sum    
        return s
    
    outputs = softmax(outputs[0])
    print(outputs)

    rknn.eval_perf(inputs=[im])

    rknn.release()

def main(cmds=None):
    args = parse_args(cmds)
    args.framework = args.framework or 'onnx'
    opt = vars(args)

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as fp:
            dct = json.load(fp)
            opt.update(dct)

    model = opt.pop("model");
    output = opt.pop("output");
    dataset = opt.pop("dataset")
    model2Rknn(model, output, dataset, **opt)

    if args.use_farward:
        img_path = "./test.jpg"
        farward(img_path, **opt)
    
if __name__ == '__main__':
    main()