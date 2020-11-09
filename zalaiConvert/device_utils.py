import sys
import os
import os.path as osp
import subprocess
import time
import argparse
import json

from zalai.common.constant import TrainProc
from zalai.common.stdout_status import fStdoutDict, fStdoutStatusDecorator


sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from zalaiConvert.utils.constant import NtbDevice, NtbDeviceInfo


def checkRknnDevice(args):
    from rknn.api import RKNN
    _, ntbs = RKNN().list_devices()
    if not ntbs:
        ret = subprocess.run("adb devices", stdout=subprocess.PIPE)
        if ret.returncode != 0: 
            return NtbDevice.ADB_ERROR_CODE
        if b"0123456789ABCDEF" in ret.stdout:
            return NtbDevice.ADB_FOUND_CODE
        else:
            return NtbDevice.DEVICE_NOT_FOUND_CODE
    return NtbDevice.NTB_FOUND_CODE


def checkToNtb(args=None):
    """
        检测adb和ntb，并把adb切换成ntb设备，返回ntb设备检测错误码
    """
    

    os.environ['path'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin") + ";" + os.environ.get('path') 
    os.environ['path'] = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "Library", "bin") + ";" + os.environ.get('path') 

    from rknn.api import RKNN
    adbs, ntbs = RKNN().list_devices()
    if not ntbs:
        ret = subprocess.run("adb devices", stdout=subprocess.PIPE)
        # print(ret.stdout)
        if ret.returncode != 0: 
            return NtbDevice.ADB_ERROR_CODE
        if b"0123456789ABCDEF" in ret.stdout:
            ret = subprocess.run("adb shell nohup start_usb.sh ntb")  
            if ret.returncode != 0:
                return NtbDevice.NTB_SWITCH_ERRORR_CODE
            time.sleep(5)
            _, ntbs2 = RKNN().list_devices()
            if not ntbs2:
                return NtbDevice.NTB_NOT_FOUND_CODE
            else:
                return NtbDevice.NTB_FOUND_CODE
        else:
            return NtbDevice.DEVICE_NOT_FOUND_CODE
    else:
        return NtbDevice.NTB_FOUND_CODE


def parse_args(cmd=None):
    model_parser = argparse.ArgumentParser(description='rknn convert api.')
    # model_parser.add_argument('--config', dest='config', action='store', required=True)

    model_args = model_parser.parse_args(cmd)
    return model_args


@fStdoutStatusDecorator(TrainProc.ON_START, TrainProc.ON_EXIT)
def main(cmd=None):
    args = parse_args(cmd)
    ret = checkToNtb(args)
    error = NtbDeviceInfo[ret]
    fStdoutDict({"errCode": ret, "error": error})


if __name__ == "__main__":
    main()