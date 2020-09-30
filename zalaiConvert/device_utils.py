import sys
import os.path as osp
import subprocess
import time
import argparse

sys.path.append(osp.dirname(osp.abspath(__file__)))


def checkRknnDevice(args):
    from rknn.api import RKNN
    from model_convert_api import model_convert
    _, ntbs = RKNN().list_devices()
    if not ntbs:
        ret = subprocess.run("adb devices", stdout=subprocess.PIPE)
        print(ret.stdout)
        if ret.returncode != 0: 
            print("adb devices error;")
            return 5
        if b"0123456789ABCDEF" in ret.stdout:
            return 0
        else:
            print("please reset rknn device and connect rknn device")
            return 1
    
    return [], ntbs

def parse_args():
    model_parser = argparse.ArgumentParser(description='rknn convert api.')

    model_parser.add_argument('--config', dest='config', action='store', required=True,
                        help='json file for rknn model conversion.')

    model_args = model_parser.parse_args()
    return model_args

def main():
    args = parse_args()
    checkConvertWrap(args)


if __name__ == "__main__":
    main()