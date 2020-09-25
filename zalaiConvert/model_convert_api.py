# coding: utf-8
import os
import sys
import json
import argparse
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))
import zalaiConvert.rknn_convert_utils as rknn_c

model_parser = argparse.ArgumentParser(description='rknn convert api.')

model_parser.add_argument('--config', dest='config', action='store', required=True,
                    help='json file for rknn model conversion.')

model_args = model_parser.parse_args()

# 瑞星微平台
rk_source_platform_sets = ["caffe", "darknet", "mxnet", "onnx", "pytorch", "tensorflow", "tensorflow_lite"]
rk_target_platform_sets = ["rk3399pro", "rk1808", "rv1109", "rv1126"]

# 其他平台
zlg_source_platform_sets = []
zlg_target_platform_sets = []

def model_convert():
    args_dic = vars(model_args)
    json_config = args_dic["config"]
    
    if not json_config is None:
            with open(json_config, 'r') as f:
                params_json = json.loads(f.read())
    
    source_platform = params_json["source_platform"]
    target_platform = params_json["target_platform"]
    
    if source_platform in rk_source_platform_sets and target_platform in rk_target_platform_sets:
        print("model conversion begins.")
        model_name        = params_json["model_name"]
        model_file        = params_json["model_file"]
        model_weight_file = params_json["model_weight_file"]
        dataset_file      = params_json["dataset_file"]
        model_out_path = params_json["model_out_path"]
        
        rknn_c.__dict__[source_platform].__dict__[model_name].convert_rknn(target_platform, model_file, model_weight_file, dataset_file, model_out_path)
    else:
        print("exit.")
    
    return

if __name__ == "__main__":
    print("model_convert_api\n")
    
    model_convert()
    