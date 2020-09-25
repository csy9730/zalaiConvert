import sys
import json
import os.path as osp

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '../..'))

dct = {
	"source_platform": "darknet",
	"target_platform": "rk1808",
	"model_name": "yolov3",
	"model_file": "data\\yolov3_darknet\\yolov3_test.cfg",
	"model_weight_file": "data\\yolov3_darknet\\yolov3_test.weights",
	"dataset_file": "data\\yolov3_darknet\\dataset.txt",
	"model_out_path": ".\\yolov3_test2.rknn"
}

jfile = 'tmp.rk.json'
with open(jfile, 'w') as fp:
    json.dump(dct, fp)
sys.argv = ['foo', '--config', jfile]

from zalaiConvert.model_convert_api import model_convert
model_convert()
