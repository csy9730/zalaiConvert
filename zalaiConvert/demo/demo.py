import sys
import json
import os
import os.path as osp

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '../..'))

from zalaiConvert.convertWrap import main

if __name__ == "__main__":
    output_dir = "tmp_demo"
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    dct = {
        "source_platform": "darknet",
        "target_platform": "rk1808",
        "model_name": "yolov3",
        "model_file": osp.abspath("data\\yolov3_darknet\\yolov3_test.cfg"),
        "model_weight_file": osp.abspath("data\\yolov3_darknet\\yolov3_test.weights"),
        "dataset_file":osp.abspath("data\\yolov3_darknet\\dataset.txt"),
        "model_out_path": osp.abspath("tmp_demo\\yolov3_test2.rknn")
    }

    with open(dct["dataset_file"],'w') as fp:
        fp.write(osp.abspath("data\\yolov3_darknet\\yolov3_test.jpg"))
        
    jfile = 'tmp_demo/tmp.rk.json'
    with open(jfile, 'w') as fp:
        json.dump(dct, fp)

    sys.argv = ['foo', '--config', jfile]

    import time
    print(time.time())
    print(time.asctime())
    main()
    print(time.asctime())