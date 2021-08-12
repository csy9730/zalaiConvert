import sys
import os.path as osp
import subprocess
import time
import argparse
import json


from zalaiConvert.utils.common import dumpsStatusDec, TrainProc, dumpsDict

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
# 环境变量添加adb, rknn_api.dll

from zalaiConvert.utils.cli import CliRuner, namespaceMergeDict


class CliExporter(CliRuner):
    @dumpsStatusDec(TrainProc.ON_ARGPARSE_START, TrainProc.ON_ARGPARSE_END)
    def parse_args(self, cmd=None):   
        parser = argparse.ArgumentParser(description='rknn convert api.')

        parser.add_argument('--config', '-c', help='json file for rknn model conversion.')
        parser.add_argument('--input', '-i')
        parser.add_argument('--output', '-o')
        parser.add_argument('--network-file', '-nf')
        parser.add_argument('--dataset-in', '-df')
        parser.add_argument('--target-platform', '-tp')
        
        args = parser.parse_args(cmd)
        return args

    def mergeCfg(self, cfg):
        json_config = cfg.config

        args = {
            "model_out_path": cfg.output, 
            "model_weight_file": cfg.input,
            "model_file": cfg.network_file
        }

        namespaceMergeDict(cfg, args) 
        if json_config:
            with open(json_config, 'r') as f:
                params_json = json.load(f)
                namespaceMergeDict(cfg, params_json)
        namespaceMergeDict(cfg, self.defaultDict)
        return cfg

    @property
    def defaultDict(self):
        return {
            "source_platform": "darknet",
            "target_platform": "rk1808",
            "model_name": "yolov3"
        }

    def handle(self, cfg):
        from zalaiConvert.device_utils import checkToNtb
        from zalaiConvert.model_convert_api import model_convert
        # print(cfg)
        # exit(0)
        ret = checkToNtb()
        if ret == 0:
            model_convert(cfg)


def main(cmd=None):
    CliExporter().run(cmd)


if __name__ == "__main__":
    main()