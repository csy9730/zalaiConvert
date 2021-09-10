import os
import sys
import time
import numpy as np
import torch
import cv2
from rknn.api import RKNN

from zalaiConvert.utils.cameraViewer import CameraViewer  
from zalaiConvert.utils.farward_utils import activateEnv, timeit, draw_pts, parse_args, RknnPredictor
from zalaiConvert.utils.rknn_utils import getRknn

activateEnv()


class RknnKptPredictor(RknnPredictor):
    def __init__(self, rknn):
        self.rknn = rknn
        self.width, self.height = 112, 112

    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs[-1]

    def postProcess(self, kpts):
        return kpts.reshape((-1, 2)) * self.width

    def draw(self, img, preds):
        print(preds.shape, preds)
        return draw_pts(img, preds)

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

    model = RknnKptPredictor(rknn)

    predictWrap(args.input, model, args)
    print("__________________exit__________________")

if __name__ == "__main__":
    cmds = [r'H:\Project\zal2\rknn_facepoint_farward_lite\face98.rknn', '-i', 
        r'H:\Project\zal2\rknn_facepoint_farward_lite\a6.png', '--device', 'rk1808', '-oabc{i}.png']
    main()
