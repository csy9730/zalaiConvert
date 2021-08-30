import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
from rknn.api import RKNN

from zalaiConvert.farward.cameraViewer import CameraViewer  
from zalaiConvert.farward.farward_utils import activateEnv, loadClassname, parse_model_cfg, filter_boxes, nms_boxes
from zalaiConvert.farward.farward_utils import getRknn

MAX_BOXES = 500
OBJ_THRESH = 0.6
NMS_THRESH = 0.3


CLASSES = ("helmet","person","hat")


activateEnv()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors, width=416):
    """
        mask: [int]
        anchors: [[int, int],]

        box: [left,top, width, height], normalize
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

def yolov3_post_process(input_data, anchors=None, img_size=416):
    # # yolov3
    # masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
    #             [59, 119], [116, 90], [156, 198], [373, 326]]
    # yolov3-tiny
    if len(input_data) == 2:
        masks = [[3,4,5], [0,1,2]]
        if anchors is None:
            anchors = [[10,14], [23,27], [37,58], [81,82], [135,169], [344,319]]
    else:
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        if anchors is None:
            # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
            anchors = [[ 8, 13], [17, 32], [28, 51], [39, 74], [56, 93], [67, 140], [108, 146], [140, 229], [239, 286]]
            # anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors, img_size)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # # Scale boxes back to original image shape.
    # width, height = 416, 416 #shape[1], shape[0]
    # image_dims = [width, height, width, height]
    # boxes = boxes * image_dims

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
        print('\nclass: {}, score: {}'.format(class_list[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        #   print('class: {}, score: {}'.format(CLASSES[cl], score))
        #   print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(class_list[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        #   print('class: {0}, score: {1:.2f}'.format(CLASSES[cl], score))
        #   print('box coordinate x,y,w,h: {0}'.format(box))


class RknnPredictor(object):
    def __init__(self, rknn):
        self.rknn = rknn
        self.anchors = None
        self.NUM_CLS = 3 # 80  3
        self.width, self.height = 416, 416  
        
        self.GRID0 = 13 # 13
        self.GRID1 = self.GRID0 * 2
        self.GRID2 = self.GRID0 * 4
        self.SPAN = 3

    @property
    def LISTSIZE(self):
        return self.NUM_CLS + 5

    def loadCfg(self, cfg_path=None):
        if cfg_path:
            pmc = parse_model_cfg(cfg_path)
            yolos = [s for s in pmc if s['type']=='yolo']
            self.NUM_CLS = yolos[0]["classes"]
            self.anchors = yolos[0]["anchors"]
            self._cfg_path = cfg_path
    def loadGenClass(self, name_file=None):
        if name_file:
            class_list = loadClassname(name_file)
            assert len(class_list) == self.NUM_CLS
        else:
            class_list = tuple([str(i+1) for i in range(model.NUM_CLS)])
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
        if len(preds) == 3: # 'tiny' not in rknn_name
            print("yolov4")
            input_data = [        
                np.transpose(preds[0].reshape(cls.SPAN, cls.LISTSIZE, cls.GRID2, cls.GRID2), (2, 3, 0, 1)),
                np.transpose(preds[1].reshape(cls.SPAN, cls.LISTSIZE, cls.GRID1, cls.GRID1), (2, 3, 0, 1)),
                np.transpose(preds[2].reshape(cls.SPAN, cls.LISTSIZE, cls.GRID0, cls.GRID0), (2, 3, 0, 1))
            ]
        else:
            print("yolov4_tiny")
            input_data = [        
                np.transpose(preds[0].reshape(cls.SPAN, cls.LISTSIZE, cls.GRID1, cls.GRID1), (2, 3, 0, 1)),
                np.transpose(preds[1].reshape(cls.SPAN, cls.LISTSIZE, cls.GRID0, cls.GRID0), (2, 3, 0, 1))
            ]
        boxes, classes, scores = yolov3_post_process(input_data, cls.anchors, cls.width)
        return boxes, classes, scores

    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs
        
    def predict(self, img, args):
        input_tensor = self.preprocess(img)
        pred = self.farward(input_tensor)
        preds = self.postProcess(pred)
        return preds


def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rknn predict & show key points')
    parser.add_argument('--model', '-m')
    parser.add_argument('--input', '-i', help='image file path')
    parser.add_argument('--output', '-o', help='save output image name')
    parser.add_argument('--config')
    parser.add_argument('--network', '--network-cfg', '-nw')
    parser.add_argument('--name-file')

    parser.add_argument('--device', choices=['rk1808', 'rv1126'], help='device: rk1808, rv1126')
    parser.add_argument('--device-id')

    parser.add_argument('--save-img', action='store_true', help='save image')
    parser.add_argument('--show-img', action='store_true', help='show image')

    return parser.parse_args()


def predictWrap(source, model, output, args):
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    W, H = model.width, model.height
    for i, img in enumerate(imgs):
        if img.shape[0:2] != (W, H):
            img = cv2.resize(img, (W, H))

        t0 = time.time()
        boxes, classes, scores = model.predict(img, args)
        print("time: ", time.time() - t0)

        # print(boxes, classes, scores)
        if boxes is not None:
            draw_box(img, boxes, scores, classes, model.class_list)
        cv2.imwrite(output, img)
        if cmv.use_camera or args.show_img:
            cv2.imshow(cmv.format(i=i), img)
            k = cv2.waitKey(vmv.waitTime)
            if k == 27:
                break
    print("predict finished")

def main(cmds=None):
    args = parse_args(cmds)

    rknn = getRknn(args.model, device=args.device, device_id=args.device_id)
    if rknn is None:
        exit(-1)
    model = RknnPredictor(rknn)
    model.loadGenClass(args.name_file)
    model.loadCfg(args.network)


    predictWrap(args.input, model, args.output, args)
    print("__________________exit__________________")

if __name__ == '__main__':
    cmds = ['-i', './10.jpg', '-o', 'save_10_rk140_dev.png', '-m', 'test_NQ.rknn', '--device', "rk1808"]
    cmds = ['-i', './1015.jpg', '-o', 'save_1015.png', '-m', 'test_NQ.rknn']
    main()
