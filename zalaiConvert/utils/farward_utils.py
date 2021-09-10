import os
import sys
import time
from functools import wraps
import cv2
import numpy as np
from .rknn_utils import *

def activateEnv(pth=None):
    if pth is None:
        pth = sys.executable
    base = os.path.dirname(os.path.abspath(pth))
    if os.name == "nt":
        lst = [
            os.path.join(base, r"Library\mingw-w64\bin"),
            os.path.join(base, r"Library\usr\bin"),
            os.path.join(base, r"Library\bin"),
            os.path.join(base, r"Scripts"),
            os.path.join(base, r"bin"),
            os.path.join(base, r"Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64"),
            os.path.join(base, r"Lib\site-packages\~knn\api\lib\hardware\Windows_x64"),
            os.path.join(base, r"Lib\site-packages\torch\lib"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../bin"),
            base,
            os.environ.get('PATH')
        ]
    else:
        lst = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../bin"),
            base,
            os.environ.get('PATH')
        ]
    os.environ['PATH'] = ';'.join(lst)


def timeit(func): 
    @wraps(func)
    def wrapper(*dargs, **kwargs):
        tic = time.time()
        retval = func(*dargs, **kwargs)
        toc = time.time()
        # time.process_time()
        print('%s() used: %fs' % (func.__name__, toc - tic)) 
        return retval 
    return wrapper


def genImageDatasetList(data_dir, output=None):
    lst = os.listdir(data_dir)
    lst2 = []
    for s in lst:
        if os.path.splitext(s)[-1].lower() in ['.jpg', '.bmp', '.png']:
            lst2.append(os.path.abspath(os.path.join(data_dir, s)))
    if output:
        with open(output, "w") as fp:
            for s in lst2:
                fp.write(s + '\n')
    else:
        return lst2

def getImagePaddingKeepWhRatio(in_shape, out_shape):
    """
        1. 计算目标宽高比和自身宽高比,
        2. 尽量使宽高比一致
        3. 计算填充量
        4. 宽高填充量必然有一个为0
    """
    wh_ratio = in_shape[0] / in_shape[1]
    tg_wh_ratio = out_shape[0] / out_shape[1]
    paddings = [0, 0]
    if wh_ratio < tg_wh_ratio:
        paddings[0] = int(in_shape[1] * (tg_wh_ratio - wh_ratio))
    if wh_ratio >= tg_wh_ratio:
        paddings[1] = int(in_shape[0] / tg_wh_ratio - in_shape[0] / wh_ratio)
    return paddings

def getImagePaddingKeepWhRatio2(in_shape, out_shape):
    wh_ratio = in_shape[0] / in_shape[1]
    tg_wh_ratio = out_shape[0] / out_shape[1]
    
    paddings = [0, 0]
    if in_shape[0] * out_shape[1] < out_shape[0] *in_shape[1]:
        paddings[0] = int(in_shape[1] * out_shape[0] / out_shape[1]) - in_shape[0]
    else:
        paddings[1] = int(in_shape[0] / out_shape[0] * out_shape[1]) - in_shape[1]
    return paddings


def imagePadding(img, out_shape):
    """
        图片，添加padding，执行resize。

        img: np.ndarray
        out_shape: tuple(int,int)

        1.  计算目标宽高比和自身宽高比,计算填充量
        2.  填充宽/高，
        3.  通过resize放缩为目标尺寸
    """
    in_shape = img.shape
    paddings = getImagePaddingKeepWhRatio(in_shape, out_shape)
    pad2 = (paddings[0] //2, paddings[1] // 2)

    if len(in_shape) == 2:
        b_img = np.zeros((in_shape[0] + paddings[0],
                          in_shape[1] + paddings[1])).astype(np.uint8)
        b_img[pad2[0]: in_shape[0] + pad2[0], pad2[1]: in_shape[1] + pad2[1]] = img
    else:
        b_img = np.zeros((in_shape[0] + paddings[0], in_shape[1] + paddings[1],
                          in_shape[2])).astype(np.uint8)
        b_img[pad2[0]:in_shape[0] + pad2[0], pad2[1]: in_shape[1] + pad2[1], :] = img

    rz_img = cv2.resize(b_img, out_shape[::-1])
    # ratio = wanted_size[1] / tt_image.shape[0], wanted_size[0] / tt_image.shape[1]
    return rz_img, paddings


def loadClassname(name_file):
    name_list = []
    with open(name_file, 'r') as F:
        content = F.readlines()
        for i in range(len(content)):
            c = content[i].rstrip('\r').rstrip('\n')
            if c:
                name_list.append(c)
    return name_list


def draw_pts(img, kpts):
    img2 = img.copy()
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        cv2.circle(img2, (x, y), radius=2, thickness=-1, color=(0, 0, 255))
    return img2


class RknnPredictor(object):
    def __init__(self, rknn):
        self.rknn = rknn
        self.width, self.height = 256, 256

    def preprocess(self, img, with_normalize=None, hwc_chw=None, **kwargs):
        if img.shape[0:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        # img = imagePadding(img, (256,256))[0]
        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32)
        if hwc_chw:
            input_image = input_image.transpose([2, 0, 1])

        return [input_image]

    def postProcess(self, kpts):
        return kpts

    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs[0]

    @timeit    
    def predict(self, img, args):
        input_tensor = self.preprocess(img)
        score_map = self.farward(input_tensor)
        kpts = self.postProcess(score_map)
        return kpts

    # def draw(self, img, preds):
    #     return draw_pts(img, preds)

    # def __del__(self):
    #     self.rknn.release()

def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rknn predict & show key points')
    parser.add_argument('model', help='model file path')
    parser.add_argument('--input', '-i', help='image file path')
    parser.add_argument('--output', '-o', help='save output image name')
    parser.add_argument('--config')


    parser.add_argument('--network', '--network-cfg', '-nw')
    parser.add_argument('--name-file', help='class name file')

    parser.add_argument('--use-padding', action='store_true', help='model file path')
    parser.add_argument('--input-chw', action='store_true', help='model file path')
    parser.add_argument('--with-normalize', action='store_true', help='rknn with normalize')
    parser.add_argument('--hwc-chw', action='store_true', help='image preprocess: from HWC to CHW')

    # parser.add_argument('--target', choices=['rk1808', 'rv1126'], help='target device: rk1808, rk1126')
    parser.add_argument('--device', choices=['rk1808', 'rv1126'], help='device: rk1808, rv1126')
    parser.add_argument('--device-id')

    parser.add_argument('--task', choices=['segment', 'detect', 'classify', 'keypoint'], default='keypoint', help='device: rk1808, rk1126')
    parser.add_argument('--run-perf', action='store_true', help='eval perf')
    
    parser.add_argument('--verbose', action='store_true', help='verbose information')
    parser.add_argument('--save-npy', action='store_true')
    parser.add_argument('--save-img', action='store_true', help='save image')
    parser.add_argument('--show-img', action='store_true', help='show image')
    parser.add_argument('--mix-scale', type=float, help='segment task params: mix scale')
    
    parser.add_argument('--use-transfer', '-t', action='store_true')
    parser.add_argument('--dataset', '-ds')
    parser.add_argument('--weight', '-w')
    
    return parser.parse_args(cmds)


def predictWrap(source, model, args=None):
    from zalaiConvert.utils.cameraViewer import CameraViewer    
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    W, H = model.width, model.height

    for i, img in enumerate(imgs):
        pred = model.predict(img, args)
        img2 = model.draw(img, pred)

        # if args.save_npy:
        #     np.save('out_{0}.npy'.format(i=i), pred[0])
        if args.save_img:
            cv2.imwrite(args.output.format(i=i), img2.astype(np.uint8))

        if cmv.use_camera or args.show_img:
            cv2.imshow(cmv.title.format(i=i), img2)
            k = cv2.waitKey(cmv.waitTime)
            if k == 27:
                break
