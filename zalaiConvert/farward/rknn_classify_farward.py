import os
import sys
import time
import numpy as np
import torch
import cv2
from rknn.api import RKNN

from zalaiConvert.utils.cameraViewer import CameraViewer  
from zalaiConvert.utils.farward_utils import activateEnv, timeit, parse_args, loadClassname, RknnPredictor
from zalaiConvert.utils.rknn_utils import getRknn, rknn_query_model, get_io_shape

activateEnv()


class RknnClassifyPredictor(RknnPredictor):
    def __init__(self, rknn):
        self.rknn = rknn
        self.width, self.height = 224, 224
        self.guess_cfg()

    def guess_cfg(self):
        self.mcfg = rknn_query_model(self.rknn.model_path)
        self.in_shape, self.out_shape = get_io_shape(self.mcfg)
        
        self.NUM_CLS = self.out_shape[0][1]
        self.width, self.height = self.in_shape[3], self.in_shape[2]
        print(self.in_shape, self.out_shape)

    def loadGenClass(self, name_file=None):
        if name_file:
            class_list = loadClassname(name_file)
            assert len(class_list) == self.NUM_CLS
        else:
            class_list = tuple([str(i+1) for i in range(self.NUM_CLS)])
        self.class_list = class_list

    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs[0]

    def postProcess(self, kpts):
        kpts = kpts.ravel()
        idx = np.argmax(kpts)

        output = kpts - kpts[idx]
        output = np.exp(output)/sum(np.exp(output))
        return idx, output[idx], self.class_list[idx]

    def draw(self, img, preds):
        print(preds)
        return img

def predictWrap(source, model, args):
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    H, W = model.height, model.width
    for i, img in enumerate(imgs):
        if img.shape[0:2] != (H, W):
            img = cv2.resize(img, (W, H))

        kpts = model.predict(img, args)

        if args.save_npy:
            np.save('out_{0}.npy'.format(i=i), kpts)
        
        img2 = model.draw(img, kpts)

        if args.save_img:
            cv2.imwrite(args.output.format(i=i), img2.astype(np.uint8))

        if cmv.use_camera or args.show_img:
            cv2.imshow(cmv.title.format(i=i), img2)
            k = cv2.waitKey(cmv.waitTime)
            if k == 27:
                break
    print("predict finished")


def main(cmds=None):
    args = parse_args(cmds)

    if args.show_img:
        pass
    elif args.output:
        args.save_img = True
    elif args.output is None:
        args.output = 'out.jpg'
        args.save_img = True

    rknn = getRknn(args.model, device=args.device, device_id=args.device_id)
    if rknn is None:
        exit(-1)

    model = RknnClassifyPredictor(rknn)

    model.loadGenClass(args.name_file)

    predictWrap(args.input, model, args)
    print("__________________exit__________________")

if __name__ == "__main__":
    main()
