from ctypes import CDLL
import os
import sys

if(os.name == "nt"):
    os.environ['path'] = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), r"Lib\site-packages\~knn\api\lib\hardware\Windows_x64") + ";" + os.environ.get('path') 
    os.environ['path'] = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), r"Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64") + ";" + os.environ.get('path') 

print(os.environ.get('path'))

path = r"G:\rknn_convert\zal_rk130\Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64\librknn_api.dll"
path = r"G:\rknn_convert\zal_rk130\Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64\librknn_api.dll"
path2 = r"G:\rknn_convert\zal_rk130\Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64\libwinpthread-1.dll"

libc = CDLL(path)
lst = dir(libc)
print(lst)
# msg = "Hello, world!\n"
# libc.printf("Testing: %s", msg)