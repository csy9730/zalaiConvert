# coding: utf-8
import os
import sys
import cv2
import numpy as np
from rknn.api import RKNN

def rknn_precompile_get(rknn_in, rknn_out):
    
    print("---rknnInference---")
    mat = np.zeros((416,416,3), np.uint8)

    # 创建 RKNN 对象
    rknn = RKNN()

    ret = rknn.load_rknn(rknn_in)
    ret = rknn.init_runtime(target='rk1808', eval_mem=True, rknn2precompile=True)
    if ret!=0:
        print('Init runtime environment failed')
        exit(ret)
    outputs = rknn.inference(inputs=[mat])

    rknn.eval_perf(inputs=[mat])

    memory_detail = rknn.eval_memory()
    
    ret = rknn.export_rknn_precompile_model(rknn_out)

    rknn.release()

def darknet_rknn(cfg_in, weight_in, dataset_in, rknn_out): 
     # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2', batch_size=1, epochs=500)
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_darknet(model=cfg_in, weight=weight_in)
    if ret != 0:
        print('Load failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=dataset_in, pre_compile=False)
    if ret != 0:
        print('Build failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    
    rknn_temp_path = os.path.abspath(os.path.dirname(__file__)) + "\\tmp.t"
    ret = rknn.export_rknn(rknn_temp_path)
    if ret != 0:
        print('Export failed!')
        exit(ret)
    print('done')
    
    rknn_precompile_get(rknn_temp_path, rknn_out)
    
    os.remove(rknn_temp_path)

    rknn.release()
   
def test():
    print("test ok. \n")

if __name__ == "__main__":
    cfg_in     = "E:\\ZALAI\\rknn_convert\\yolov3_darknet\\yolov3_test.cfg"
    weight_in  = "E:\\ZALAI\\rknn_convert\\yolov3_darknet\\yolov3_test.weights"
    dataset_in = "E:\\ZALAI\\rknn_convert\\yolov3_darknet\\dataset.txt"
    rknn_out   = "E:\\ZALAI\\rknn_convert\\yolov3_darknet\\rknn\\yolov3_test.rknn"
    
    darknet_rknn(cfg_in, weight_in, dataset_in, rknn_out)
    
