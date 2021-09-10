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
from zalaiConvert.utils.keypoint_utils import get_argmax_pt, pts_unclip

activateEnv()


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

    def postProcess(self, score_map):
        if not isinstance(score_map, torch.Tensor):
            print("trans to tensor")
            score_map = torch.Tensor(score_map)

        coords = get_argmax_pt(score_map)  # float type
        scale = 256/200
        center = torch.Tensor([127, 127])
        kpts = pts_unclip(coords[0], center, scale, [64, 64])
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

    def draw(self, img, preds):
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

    if args.output:
        args.save_img = True
    elif args.output is None and args.save_img:
        args.output = 'out.jpg'

    rknn = getRknn(args.model, device=args.device, device_id=args.device_id)
    if rknn is None:
        exit(-1)

    model = RknnPredictor(rknn)

    predictWrap(args.input, model, args)
    print("__________________exit__________________")

if __name__ == "__main__":
    # cmds += ['--use-padding', '--input-chw', '--device', 'rk1808', '--save-img', '--task', 'segment']
    main()
