# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse
import cv2
import math
import os.path as osp
import sys
import time

import numpy as np
import torch

from zalaiConvert.utils.detect_utils import draw_box, yolov5_post_processfrom zalaiConvert.utils.cameraViewer import CameraViewer
from zalaiConvert.utils.farward_utils import predictWrap, timeit, loadClassname
from zalaiConvert.utils.detect_utils import yolov5_post_process, draw_box
__DIR__ = os.path.dirname(os.path.abspath(__file__))


class Yolov5OnnxPredictor():
    def __init__(self, model_file):
        self.loadModel(model_file)
        # self.width = 256
        # self.height = 256

        self.anchors = None
        self.NUM_CLS = None
        self.masks = None
        self.width, self.height = 416, 416
        self.GRID = [52, 26, 13]
        self.SPAN = 3

        self.mean = np.array([0, 0., 0.], dtype=np.float32)*255
        self.std = np.array([1, 1, 1], dtype=np.float32)*255

        self._cfg_path = None
        self.loadCfg()
        self.loadGenClass()


    def guess_cfg(self):
        pass

    def set_NUMCLS(self, NUM_CLS):
        if self.NUM_CLS:
            assert self.NUM_CLS == NUM_CLS
        else:
            self.NUM_CLS = NUM_CLS

    @property
    def LISTSIZE(self):
        return self.NUM_CLS + 5

    def loadCfg(self, cfg_path=None):
        self.set_NUMCLS(8) # yolos[0]["classes"]
        self.anchors = [[2,4],[6,5],[5,11],[13,9],[19,14],[31,22],[54,34],[90,61],[127,125]]
        # self.anchors = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]) # yolos[0]["anchors"]
        self.SPAN = 3 # len(yolos[0]["mask"])
        self.masks = [[0,1,2],[3,4,5] ,[6,7,8],] # [y["mask"] for y in yolos]
        print(self.masks,self.SPAN)
        self._cfg_path = cfg_path
        # self.guess_cfg()

    def loadGenClass(self, name_file=None):
        if name_file:
            class_list = loadClassname(name_file)
            self.set_NUMCLS(len(class_list))
        else:
            class_list = tuple([str(i+1) for i in range(self.NUM_CLS)])
        self.class_list = class_list
    
    def loadModel(self, f):
        import onnxruntime as rt
        assert os.path.isfile(f), f
        sess = rt.InferenceSession(f)
        self.input_name = sess.get_inputs()[0].name
        self.label_name = sess.get_outputs()[0].name
        self.label_name2 = sess.get_outputs()[1].name
        self.label_name3 = sess.get_outputs()[2].name
        print('input_name: ' + self.input_name)
        print('label_name: ' + self.label_name+ self.label_name2+ self.label_name3)
        self.sess = sess
        return self.sess

    def preprocess(self, img):
        if img.shape[0:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))

        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #print("input_image.shape:", input_image.shape)
        input_image = input_image.astype(np.float32)
        input_image = (input_image - self.mean) / self.std
        input_image = input_image.transpose([2, 0, 1])

        input_tensor = torch.tensor(input_image)
        input_tensor = input_tensor.unsqueeze(0)
        #print("input_tensor.shape", input_tensor.shape)
        return input_tensor



    def postProcess(self, preds):
        """
            SPAN, GRID0, GRID0, LISTSIZE  => GRID0, GRID0, SPAN, LISTSIZE
        """
        # print(preds[0].shape)
        # print(len(preds))
        input_data = [
            # np.transpose(preds[i].reshape(self.SPAN, self.LISTSIZE, g, g), (2, 3, 0, 1))
            np.transpose(preds[i].reshape(self.SPAN, g, g, self.LISTSIZE), (1, 2, 0, 3))
            for i, g in enumerate(self.GRID)
        ]
        boxes, classes, scores = yolov5_post_process(input_data, self.anchors, self.masks)
        return boxes, classes, scores
    def farward(self, x):
        pred_onx = self.sess.run([self.label_name, self.label_name2, self.label_name3], {self.input_name: x})
        return pred_onx # [0]
    @timeit
    def predict(self, img):
        input_tensor = self.preprocess(img)
        score_map = self.farward(input_tensor.numpy())
        kpts = self.postProcess(score_map) # torch.from_numpy(
        
        return kpts

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
        # if img.shape[0:2] != (W, H):
        #     img = cv2.resize(img, (W, H))
        # t0 = time.time()
        pred = model.predict(img)
        # print("time: ", time.time() - t0)

        img2 = model.draw(img, pred)

        # if args.save_npy:
        #     np.save('out_{0}.npy'.format(i=i), pred[0])

        if 1:
            cv2.imwrite(args.output.format(i=i), img2.astype(np.uint8))

        if 1:
            cv2.imshow(cmv.title.format(i=i), img2)
            k = cv2.waitKey(cmv.waitTime)
            if k == 27:
                break

    print("predict finished")


def parse_args(cmds=None):
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--model-in', '-m', dest="model_file", help='model parameters', required=True)
    parser.add_argument('--input','-i',help='input image path')
    parser.add_argument('--output','-o', default='out_.jpg', help='output image path')

    parser.add_argument('--network', '--network-cfg', '-nw')
    parser.add_argument('--name-file', help='class name file')
    args = parser.parse_args(cmds)
    return args


def main(cmds=None):
    args = parse_args(cmds)

    model = Yolov5OnnxPredictor(args.model_file)
    model.loadCfg(args.network)
    model.loadGenClass(args.name_file)
    predictWrap(args.input, model, args)


if __name__ == '__main__':
    main()
