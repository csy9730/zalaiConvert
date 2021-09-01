import os
import sys
import time
from functools import wraps
import cv2
import numpy as np
from rknn.api import RKNN

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


def parse_model_cfg(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif key in ['from', 'layers', 'mask']:  # return array
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'max_delta']

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return mdefs


def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH=0.5):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores, NMS_THRESH=0.5):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def draw_pts(img, kpts):
    img2 = img.copy()
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        cv2.circle(img2, (x, y), radius=2, thickness=-1, color=(0, 0, 255))
    return img2


def rknn_query_model(model):
    rknn = RKNN() 
    mcfg = rknn.fetch_rknn_model_config(model)
    if mcfg:
        # print(mcfg["target_platform"], "version=", mcfg["version"])
        print("pre_compile=", mcfg["pre_compile"])
    return mcfg


def get_io_shape(mcfg):
    mt = mcfg["norm_tensor"]
    mg = mcfg["graph"]

    in_shape = []
    out_shape = []
    for i, g in enumerate(mg):
        sz = mt[i]['size']
        sz.reverse()
        if g['left']=='output':
            out_shape.append(sz)
        else:
            in_shape.append(sz)
    return in_shape, out_shape


def getRknn(model, device=None, rknn2precompile=None, verbose=None, device_id=None, **kwargs):
    rknn = RKNN(verbose=verbose)  
    assert os.path.isfile(model)
    print('--> Loading model')  
    ret = rknn.load_rknn(model)
    if ret != 0:
        print('load_rknn failed')
        return None
    print('Load done')

    print('--> Init runtime environment')
    ret = rknn.init_runtime(target=device, device_id=device_id, eval_mem=False, rknn2precompile=rknn2precompile)
    if ret != 0:
        print('Init runtime environment failed')
        return None
    print('Init runtime done')

    rknn.model_path = model
    return rknn


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