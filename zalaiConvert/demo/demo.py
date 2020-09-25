import sys
import json
import os.path as osp

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '../..'))

dct = {
	"source_platform": "darknet",
	"target_platform": "rk1808",
	"model_name": "yolov3",
	"model_file": osp.abspath("data\\yolov3_darknet\\yolov3_test.cfg"),
	"model_weight_file": osp.abspath("data\\yolov3_darknet\\yolov3_test.weights"),
	"dataset_file":osp.abspath("data\\yolov3_darknet\\dataset.txt"),
	"model_out_path": osp.abspath(".\\yolov3_test2.rknn")
}

jfile = 'tmp.rk.json'
with open(jfile, 'w') as fp:
    json.dump(dct, fp)
sys.argv = ['foo', '--config', jfile]

import time
print(time.time())
print(time.asctime())
from zalaiConvert.model_convert_api import model_convert
model_convert()
