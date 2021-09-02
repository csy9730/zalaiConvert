import os
import sys
import time
import numpy as np
import cv2
from rknn.api import RKNN

from zalaiConvert.farward.cameraViewer import CameraViewer  
from zalaiConvert.farward.farward_utils import activateEnv, loadClassname, parse_model_cfg, \
    filter_boxes, nms_boxes, timeit, parse_args, rknn_query_model, get_io_shape
from zalaiConvert.farward.farward_utils import getRknn, RknnPredictor

MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.5

activateEnv()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors, width=416):
    """
        input: float[GRID0, GRID0, SPAN, LISTSIZE], SPAN=2 or 3, LISTSIZE= NUM_CLASS+5
            input[LISTSIZE] = [x,y,w,h, box_conf, *cls_conf[...]]
        mask: [int]             mask is index of anchors
        anchors: [[int, int],]

        box: [left,top, width, height],   normalize
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
    box_wh /= (width, width)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def process2(input, mask, anchors, width=416):
    """
        input: float[GRID0, GRID0, SPAN, LISTSIZE], SPAN=2 or 3, LISTSIZE= NUM_CLASS+5
            input[LISTSIZE] = [x,y,w,h, box_conf, *cls_conf[...]]
        mask: [int]             mask is index of anchors
        anchors: [[int, int],]
        OBJ_THRESH：threhold
        
        Returns:
        box: [left,top, width, height],   normalize
        box_confidence:                 , normalize
        box_class_probs                 , normalize
    """
    
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    obj_thresh = -np.log(1/OBJ_THRESH - 1)
    pos = np.where(box_confidence > obj_thresh)
    input = input[pos]
    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    for idx, val in enumerate(pos[2]):
        box_wh[idx] = box_wh[idx] * anchors[pos[2][idx]]
    pos0 = np.array(pos[0])[:, np.newaxis]
    pos1 = np.array(pos[1])[:, np.newaxis]
    grid = np.concatenate((pos1, pos0), axis=1)
    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (width, width)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def yolov3_post_process(input_data, anchors=None, masks=None):
    # yolov3
    if masks is None:
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


def draw_box(image, boxes, scores, classes, class_list):
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


class RknnPredictor(object):
    def __init__(self, rknn):
        self.rknn = rknn

        self.anchors = None
        self.NUM_CLS = None
        self.masks = None
        self.width, self.height = 416, 416
        self.GRID = [13, 26, 52]
        self.SPAN = 3
        
        self._cfg_path = None

    def guess_cfg(self):
        self.mcfg = rknn_query_model(self.rknn.model_path)
        self.in_shape, self.out_shape = get_io_shape(self.mcfg)
        
        self.set_NUMCLS(self.out_shape[0][1] // self.SPAN - 5)
        print(self.in_shape, self.out_shape)

        self.GRID = [a[2] for a in self.out_shape]

    def set_NUMCLS(self, NUM_CLS):
        if self.NUM_CLS:
            assert self.NUM_CLS == NUM_CLS
        else:
            self.NUM_CLS = NUM_CLS

    @property
    def LISTSIZE(self):
        return self.NUM_CLS + 5

    def loadCfg(self, cfg_path=None):
        if cfg_path:
            pmc = parse_model_cfg(cfg_path)
            yolos = [s for s in pmc if s['type']=='yolo']
            self.set_NUMCLS(yolos[0]["classes"])
            self.anchors = yolos[0]["anchors"]
            self.SPAN = len(yolos[0]["mask"])
            self.masks = [y["mask"] for y in yolos]
            print(self.masks,self.SPAN)
            self._cfg_path = cfg_path
            self.guess_cfg()
            
    def loadGenClass(self, name_file=None):
        if name_file:
            class_list = loadClassname(name_file)
            self.set_NUMCLS(len(class_list))
        else:
            class_list = tuple([str(i+1) for i in range(self.NUM_CLS)])
        self.class_list = class_list

    def preprocess(self, img, with_normalize=None, hwc_chw=None, **kwargs):
        if img.shape[0:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        # img = imagePadding(img, (256,256))[0]
        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32)
        if hwc_chw:
            input_image = input_image.transpose([2, 0, 1])

        return [input_image]

    def postProcess(cls, preds):
        input_data = [
            np.transpose(preds[i].reshape(cls.SPAN, cls.LISTSIZE, g, g), (2, 3, 0, 1))
            for i, g in enumerate(cls.GRID)
        ]
        boxes, classes, scores = yolov3_post_process(input_data, cls.anchors, cls.masks)
        return boxes, classes, scores

    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs

    @timeit  
    def predict(self, img, args):
        input_tensor = self.preprocess(img)
        pred = self.farward(input_tensor)
        preds = self.postProcess(pred)
        return preds


def predictWrap(source, model, args=None):    
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    W, H = model.width, model.height

    for i, img in enumerate(imgs):
        # if img.shape[0:2] != (W, H):
        #     img = cv2.resize(img, (W, H))
    
        t0 = time.time()
        boxes, classes, scores = model.predict(img, args)
        print("time: ", time.time() - t0)
        if boxes is not None:
            draw_box(img, boxes, scores, classes, model.class_list)

        # if args.save_npy:
        #     np.save('out_{0}.npy'.format(i=i), pred[0])

        if args.save_img:
            cv2.imwrite(args.output.format(i=i), img.astype(np.uint8))

        if cmv.use_camera or args.show_img:
            cv2.imshow(cmv.format(i=i), img)
            k = cv2.waitKey(vmv.waitTime)
            if k == 27:
                break


    print("predict finished")


def main(cmds=None):
    args = parse_args(cmds)

    if args.output:
        args.save_img = True
    elif args.output is None and args.save_img:
        args.output = 'out.jpg'

    mcfg = rknn_query_model(args.model)
    print(get_io_shape(mcfg))
    # exit(0)
    rknn = getRknn(args.model, device=args.device, device_id=args.device_id)
    if rknn is None:
        exit(-1)
    model = RknnPredictor(rknn)
    model.loadCfg(args.network)
    model.loadGenClass(args.name_file)

    predictWrap(args.input, model, args)
    print("__________________exit__________________")

if __name__ == "__main__":
    # cmds += ['--use-padding', '--input-chw', '--device', 'rk1808', '--save-img', '--task', 'segment']
    main()
