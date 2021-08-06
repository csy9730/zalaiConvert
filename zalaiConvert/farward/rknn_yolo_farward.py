import os
import sys
import time
import numpy as np
import cv2
from rknn.api import RKNN

# import torch

GRID0 = 13
GRID1 = 26
GRID2 = 52

SPAN = 3
MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.5

def activateEnv(pth=None):
    if pth is None:
        pth = sys.executable
    base = os.path.dirname(os.path.abspath(pth))
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
    os.environ['PATH'] = ';'.join(lst)

activateEnv()


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
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh']

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return mdefs


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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):
    """
        mask: [int]
        anchors: [[int, int],]

        box: [left,top, width, height], normalize
        box_confidence:                 , normalize
        box_class_probs                 , normalize
    """
    print(input.shape, "shape")
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
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

def nms_boxes(boxes, scores):
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


def yolov3_post_process(input_data, anchors=None):
    # yolov3
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    if anchors is None:
        anchors = [[10, 17],  [19, 34],  [31, 62],  [34, 163],  [58, 9],  [63, 49], [107, 152], [142, 307], [282, 367]]
    candbox = anchors # [[anchors[i], anchors[i+1]] for i in range(0, 18, 2)] 
    print(candbox)
    # yolov3-tiny
    # masks = [[3, 4, 5], [0, 1, 2]]
    # candbox = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, candbox)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def guessSource(source=None):
    import glob
    from pathlib import Path
    img_formats = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')  # acceptable image suffixes
    
    img_paths = []
    video_id = None
    if source:
        p = str(Path(source).absolute())  # os-agnostic absolute path
        if p.endswith('.txt') and os.path.isfile(p):
            with open(sources, 'r') as f:
                files = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        elif '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        print("files", files)
        img_paths = [x for x in files if x.split('.')[-1].lower() in img_formats]
    else:
        video_id = 0
    return img_paths, video_id


def generate_camera(cap):
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # print(img.shape)
        yield img

def genImgs(img_paths):
    for i in img_paths:
        img = cv2.imread(i, 1)
        if img is None:
            print(i, "not found")
        else:
            yield img


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


def draw(image, boxes, scores, classes, class_list):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        print('class: {}, score: {}'.format(class_list[cl], score))
        print('box normal coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        left = max(0, np.floor(x + 0.5).astype(int))
        top = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)
        cv2.putText(image, '{0} {1:.1f}'.format(class_list[cl], score),
                    (left, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1)



def rknn_query_model(model):
    rknn = RKNN() 
    mcfg = rknn.fetch_rknn_model_config(model)
    print(mcfg["target_platform"], "version=", mcfg["version"])
    print("pre_compile=", mcfg["pre_compile"])

    return mcfg

def get_io_shape(mcfg):
    mt = mcfg["norm_tensor"]
    mg = mcfg["graph"]

    in_shape = []
    out_shape = []
    for i, g in enumerate(mg):
        if g['left']=='output':
            out_shape.append(mt[i]['size'])
        else:
            in_shape.append(mt[i]['size'])
    return in_shape, out_shape

def getRknn(model, device=None, rknn2precompile=None, verbose=None, **kwargs):
    rknn = RKNN(verbose=verbose)
    assert os.path.exists(model)
    print('--> Loading model')  
    ret = rknn.load_rknn(model)
    if ret != 0:
        print('load_rknn failed')
        return None
    print('Load done')

    print('Init runtime environment')
    ret = rknn.init_runtime(target=device, eval_mem=False, rknn2precompile=rknn2precompile) # 
    if ret != 0:
        print('Init runtime environment failed')
        return None
    print('Init runtime done')

    return rknn

def rknnPredict(inputs, rknn):
    outputs = rknn.inference(inputs=inputs)
    return outputs


def preprocess(img, with_normalize=None, hwc_chw=None, **kwargs):
    # print(img.shape)
    # height, width = img.shape[:2]
    # raw_img = img
    if img.shape[0:2] != (416, 416):
        img = cv2.resize(img, (416, 416))
    # img = imagePadding(img, (256,256))[0]
    input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_image = input_image.astype(np.float32)

    if not with_normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        input_image = (input_image/255.0 - mean) / std

    if hwc_chw:
        input_image = input_image.transpose([2, 0, 1])

    return input_image


COLOR_LIST = [[0,0,0],[0,0,128], [0,128,0], [128,0,0], [255,128,0], [128,0,255], [0, 255,128]]
# COLOR_LIST = np.array(COLOR_LIST)

def drawImagePoints(img, pred):
    for p in pred:
        cv2.circle(img, p, 3, (0,0,255))


def draw_predict(img, preds):
    for i in range(0, 68):
        cv2.circle(img, (int(preds[i][0]), int(preds[i][1])), 2, (0,255,0), -1)


def showImage(img, text="untitled"):
    cv2.imshow(text, img)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        raise StopIteration
    cv2.waitKey()


def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rknn predict & show key points')
    parser.add_argument('model', help='model file path')
    parser.add_argument('--input', '-i', help='image file path')
    parser.add_argument('--output', '-o', help='save output image name')
    parser.add_argument('--config')

    parser.add_argument('--network', '--network-cfg', '-nw')
    parser.add_argument('--use-padding', action='store_true', help='model file path')
    parser.add_argument('--save-img', action='store_true', help='model file path')
    parser.add_argument('--show-img', action='store_true', help='model file path')
    parser.add_argument('--input-chw', action='store_true', help='model file path')
    parser.add_argument('--device', help='device: rk1808, rk1126')
    parser.add_argument('--task', choices=['segment', 'detect', 'classify', 'keypoint'], default='keypoint', help='device: rk1808, rk1126')
    parser.add_argument('--mix-scale', type=float, help='segment task params: mix scale')
    parser.add_argument('--verbose', action='store_true', help='verbose information')
    parser.add_argument('--with-normalize', action='store_true', help='rknn with normalize')
    parser.add_argument('--hwc-chw', action='store_true', help='image preprocess: from HWC to CHW')
    parser.add_argument('--run-perf', action='store_true', help='eval perf')

    parser.add_argument('--save-npy', action='store_true')
    parser.add_argument('--use-transfer', '-t', action='store_true')
    parser.add_argument('--dataset', '-ds')
    parser.add_argument('--weight', '-w')
    
    return parser.parse_args(cmds)


def predictWrap(source, model_path, cfg=None, args=None):
    rknn = getRknn(model_path, device=args.device)
    if rknn is None:
        exit(-1)

    if cfg:
        pmc = parse_model_cfg(cfg)
        yolos = [s for s in pmc if s['type']=='yolo']
        NUM_CLS = yolos[0]["classes"]
        anchors = yolos[0]["anchors"]
    else:
        anchors = None
        NUM_CLS = 2 # len(class_list)

    class_list = tuple([str(i+1) for i in range(NUM_CLS)])
    

    img_paths, video_id = guessSource(source)
    if img_paths:
        imgs = genImgs(img_paths)
        use_camera = False
        print("use images")
        title = "img_{i}"
    else:
        cap = cv2.VideoCapture(video_id)
        imgs = generate_camera(cap)
        use_camera = True
        print("use camera")
        title = "camera"

    waitTime = 30 if use_camera else 0
    W, H = 416, 416
    LISTSIZE = NUM_CLS + 5
    for i, img in enumerate(imgs):
        if img.shape[0:2] != (W, H):
            img = cv2.resize(img, (W, H))

        img2 = preprocess(img, with_normalize=args.with_normalize, hwc_chw=args.hwc_chw)
        # print(img.shape)
        t0 = time.time()
        pred = rknnPredict([img2], rknn)
        print("time: ", time.time() - t0)
        # print("pred", pred, pred[0].shape)
        # print("shape", pred[0].shape)

        input_data = [
            np.transpose(pred[0].reshape(SPAN, LISTSIZE, GRID0, GRID0), (2, 3, 0, 1)),
            np.transpose(pred[1].reshape(SPAN, LISTSIZE, GRID1, GRID1), (2, 3, 0, 1)),
            np.transpose(pred[2].reshape(SPAN, LISTSIZE, GRID2, GRID2), (2, 3, 0, 1))
        ]

        boxes, classes, scores = yolov3_post_process(input_data, anchors)
        if boxes is not None:
            draw(img, boxes, scores, classes, class_list)
    
        # preds = postProcess(pred[0])
    
        # if args.save_npy:
        #     np.save('out_{0}.npy'.format(i=i), pred[0])
        
        # draw_predict(img, preds)

        # print(preds, preds.shape)
        if args.save_img:
            cv2.imwrite(args.output.format(i=i), img.astype(np.uint8))

        if args.show_img:
            cv2.imshow(title.format(i=i), img)
            k = cv2.waitKey(waitTime)
            if k == 27:
                break

    if use_camera:
        cap.release()

    cv2.destroyAllWindows()
    rknn.release()


def main(cmds=None):
    args = parse_args(cmds)
    model_path = args.model 
    img_path = args.input

    if args.output:
        args.save_img = True
    elif args.output is None and args.save_img:
        args.output = 'out.jpg'

    predictWrap(img_path, model_path, args.network, args)
    print("__________________exit__________________")

if __name__ == "__main__":
    # cmds += ['--use-padding', '--input-chw', '--device', 'rk1808', '--save-img', '--task', 'segment']
    main()
