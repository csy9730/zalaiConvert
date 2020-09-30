import sys
import os.path as osp
import subprocess
import time
import argparse

sys.path.append(osp.dirname(osp.abspath(__file__)))
# 环境变量添加adb, rknn_api.dll


def checkConvertWrap(args):
    from rknn.api import RKNN
    from model_convert_api import model_convert
    adbs, ntbs = RKNN().list_devices()
    if not ntbs:
        ret = subprocess.run("adb devices", stdout=subprocess.PIPE)
        print(ret.stdout)
        if ret.returncode != 0: 
            print("adb devices error;")
            return 5
        if b"0123456789ABCDEF" in ret.stdout:
            ret = subprocess.run("adb shell nohup start_usb.sh ntb")  
            if ret.returncode != 0:
                print("adb shell error")
                return 4
            time.sleep(5)
            _, ntbs2 = RKNN().list_devices()
            if not ntbs2:
                print("no ntb devices;please install ntb driver")
                return 2
            else:
                model_convert(args)
                return 0
        else:
            print("please reset rknn device and connect rknn device")
            # print("no ntb devices;please install ntb driver")
            return 1
    else:
        model_convert(args)
        return 0

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