
class NtbDevice:
    NTB_FOUND_CODE = 0  
    NTB_FOUND = "ntb found"

    DEVICE_NOT_FOUND_CODE = 5
    DEVICE_NOT_FOUNDE = "please reset rknn device and connect rknn device"

    ADB_ERROR_CODE = 4
    ADB_ERROR = "adb devices error"

    ADB_FOUND_CODE = 3
    ADB_FOUND = "adb devices found"

    NTB_SWITCH_ERRORR_CODE = 2
    NTB_SWITCH_ERRORR = "adb shell error"

    NTB_NOT_FOUND_CODE = 1
    NTB_NOT_FOUND = "no ntb devices;please install ntb driver"


NtbDeviceInfo = [
    "ntb found",
    "no ntb devices;please install ntb driver",
    "adb devices found",
    "adb shell error",
    "adb devices error",
    "please reset rknn device and connect rknn device"
]