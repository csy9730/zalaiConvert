import sys
import json
import os
import os.path as osp

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '../..'))

from zalaiConvert.farward.rknn_yolo_farward import main


if __name__ == "__main__":
    output_dir = "tmp_demo/switch"
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    dct = {
        "source_platform": "darknet",
        "target_platform": "rk1808",
        "model_name": "yolov3",
        "model_file": osp.abspath("data\\yolov3_darknet\\yolov3_test.cfg"),
        "model_weight_file": osp.abspath("data\\yolov3_darknet\\yolov3_test.weights"),
        "dataset_in": 'E:\\nnCollect\\zalaiyolo\\data\\samples\\export_data\\data_list.txt',# osp.abspath("data\\yolov3_darknet\\dataset.txt"),
        "model_out_path": osp.abspath("tmp_demo\\yolov3_test3.rknn")
    }

    with open(dct["dataset_in"],'w') as fp:
        fp.write(osp.abspath("data\\yolov3_darknet\\yolov3_test.jpg"))
        
    jfile = 'tmp_demo/tmp.rk.json'
    with open(jfile, 'w') as fp:
        json.dump(dct, fp)

    # sys.argv = ['foo', '--config', jfile]
    # main()
    cmd = ['-i', dct["model_weight_file"],
     '-o', dct["model_out_path"],
     '-nf', dct["model_file"],
     '-tp', "rk1808", '-df', dct["dataset_in"]]
     # , '--config', jfile
    print(cmd)
    main(cmd) 