import sys
import os
import os.path as osp
import subprocess
import time
import argparse
import json


from zalaiConvert.utils.common import dumpsStatusDec, TrainProc, dumpsDict

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from zalaiConvert.utils.constant import NtbDevice, NtbDeviceInfo


if os.name == "nt":
    PY = os.path.dirname(os.path.abspath(sys.executable))
    os.environ['path'] = ";".join([
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
        os.path.join(PY, r"Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64"),
        os.path.join(PY, r"Lib\site-packages\~knn\api\lib\hardware\Windows_x64"),
        os.path.join(PY, r"Library/bin"),
        os.environ.get('path')
    ])

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


def parse_adb(ret):
    sp = ret.split(b'\n')
    devs = []
    if len(sp) >= 2:
        for s in sp[1:]:
            if s:
                devs.append(s)
    return devs


def parseAdbStr(ss):
    import re
    pat = re.compile(r'([0-9a-fA-F]{8,}|null)\s+(?:device|unauthorized)')
    fd = pat.findall(str(ss, encoding='utf-8'))
    return fd


def checkToNtb(args=None):
    """
        检测adb和ntb，并把adb切换成ntb设备，返回ntb设备检测错误码
    """
    
    from rknn.api import RKNN
    adbs, ntbs = RKNN().list_devices()
    if not ntbs:
        ret = subprocess.run(["adb", "devices"], stdout=subprocess.PIPE)
        # print(ret.stdout)
        if ret.returncode != 0: 
            return NtbDevice.ADB_ERROR_CODE
        # print(ret.stdout)
        devs = parseAdbStr(ret.stdout)
        # print("devs", devs)
        for dev in devs:
            if dev:
            # if b"0123456789ABCDEF" in ret.stdout:
                ret = subprocess.run("adb shell nohup start_usb.sh ntb")  
                if ret.returncode != 0:
                    return NtbDevice.NTB_SWITCH_ERRORR_CODE
                time.sleep(5)
                _, ntbs2 = RKNN().list_devices()
                if not ntbs2:
                    return NtbDevice.NTB_NOT_FOUND_CODE
                else:
                    return NtbDevice.NTB_FOUND_CODE        
        return NtbDevice.DEVICE_NOT_FOUND_CODE
    else:
        return NtbDevice.NTB_FOUND_CODE


def parse_args(cmd=None):
    model_parser = argparse.ArgumentParser(description='rknn convert api.')
    # model_parser.add_argument('--config', dest='config', action='store', required=True)

    model_args = model_parser.parse_args(cmd)
    return model_args


@dumpsStatusDec(TrainProc.ON_START, TrainProc.ON_EXIT)
def main(cmd=None):
    args = parse_args(cmd)
    ret = checkToNtb(args)
    error = NtbDeviceInfo[ret]
    dumpsDict({"errCode": ret, "error": error})


def killserver(cmds=None):
    if os.name == "nt":
        os.system("taskkill /im adb.exe /f")
        os.system("taskkill /im npu_transfer_proxy.exe /f")


def startserver(cmds=None):
    npu = os.path.join(PY, 
        r"Lib\site-packages\rknn\3rdparty\platform-tools\ntp\windows-x86_64\npu_transfer_proxy.exe")
    print(npu)
    os.system("adb.exe start-server")
    subprocess.Popen([npu])
    # os.system(npu)


if __name__ == "__main__":
    main()