# coding: utf-8
import os
import sys
import json
import argparse
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))

# 瑞星微平台
rk_source_platform_sets = ["caffe", "darknet", "mxnet", "onnx", "pytorch", "tensorflow", "tensorflow_lite"]
rk_target_platform_sets = ["rk3399pro", "rk1808", "rv1109", "rv1126"]

# 其他平台
zlg_source_platform_sets = []
zlg_target_platform_sets = []


def model_convert(cfg):
    import zalaiConvert.rknn_convert_utils as rknn_c
    source_platform = cfg.source_platform
    target_platform = cfg.target_platform
    model_name = cfg.model_name
    dct = {
        "target": target_platform, 
        "cfg_in": cfg.model_file,
        "weight_in": cfg.model_weight_file, 
        "dataset_in": cfg.dataset_in,
        "rknn_out": cfg.model_out_path,
        "log_file": None
        # "rknn.log"
    }

    if source_platform in rk_source_platform_sets and target_platform in rk_target_platform_sets:        
        rknn_c.__dict__[source_platform].__dict__[model_name].convert_rknn(**dct)
    else:
        print("exit.")
