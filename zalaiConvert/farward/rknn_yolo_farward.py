import os
import sys
import time
import numpy as np
import cv2
from rknn.api import RKNN

from zalaiConvert.utils.cameraViewer import CameraViewer  
from zalaiConvert.utils.farward_utils import activateEnv, loadClassname, \
    timeit, parse_args, RknnPredictor
from zalaiConvert.utils.rknn_utils import getRknn, rknn_query_model, get_io_shape
from zalaiConvert.utils.detect_utils import nms_boxes, filter_boxes, parse_model_cfg, yolov3_post_process, draw_box

activateEnv()

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
        """
            SPAN, LISTSIZE, GRID0, GRID0,  => GRID0, GRID0, SPAN, LISTSIZE
        """
        input_data = [
            np.transpose(preds[i].reshape(cls.SPAN, cls.LISTSIZE, g, g), (2, 3, 0, 1))
            for i, g in enumerate(cls.GRID)
        ]
        boxes, classes, scores = yolov3_post_process(input_data, cls.anchors, cls.masks)
        return boxes, classes, scores

    @timeit  
    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs

    @timeit  
    def predict(self, img, args):
        input_tensor = self.preprocess(img)
        pred = self.farward(input_tensor)
        preds = self.postProcess(pred)
        return preds

    def draw(self, img, preds):
        boxes, classes, scores = preds
        if boxes is not None:
            return draw_box(img, boxes, scores, classes, self.class_list)
        return img

def predictWrap(source, model, args=None):    
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    W, H = model.width, model.height

    for i, img in enumerate(imgs):
        preds = model.predict(img, args)
        img2 = model.draw(img, preds)

        # if args.save_npy:
        #     np.save('out_{0}.npy'.format(i=i), pred[0])

        if args.save_img:
            cv2.imwrite(args.output.format(i=i), img.astype(np.uint8))

        if cmv.use_camera or args.show_img:
            cv2.imshow(cmv.title.format(i=i), img)
            k = cv2.waitKey(cmv.waitTime)
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
