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
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from lib.utils.utils import create_logger
from lib.core.evaluation import decode_preds, compute_nme, get_preds
from lib.utils.transforms import transform_preds
from utils.farward_utils import activateEnv


__DIR__ = os.path.dirname(os.path.abspath(__file__))
activateEnv()


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


def generate_camera(video_id=0):
    cap = cv2.VideoCapture(video_id)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # print(img.shape)
        yield img

def genImgs(img_paths):
    for i in img_paths:
        img = cv2.imread(i, 1)
        yield img


def parse_args(cmds=None):
    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--model-in', '-m', dest="model_file", help='model parameters', required=True)
    parser.add_argument('--input','-i',help='input image path')

    parser.add_argument('--device', help='device: rk1808, rk1126')
    parser.add_argument('--task', choices=['segment', 'detect', 'classify', 'keypoint'], default='keypoint', help='device: rk1808, rk1126')

    parser.add_argument('--input-chw', action='store_true', help='model file path')
    parser.add_argument('--use-padding', action='store_true', help='model file path')
    parser.add_argument('--with-normalize', action='store_true', help='rknn with normalize')
    parser.add_argument('--hwc-chw', action='store_true', help='image preprocess: from HWC to CHW')

    parser.add_argument('--output', '-o', default='out.jpg', help='save output image name')
    parser.add_argument('--save-img', action='store_true', help='model file path')
    parser.add_argument('--show-img', action='store_true', help='model file path')
    parser.add_argument('--save-npy', action='store_true', help='save predict output to npy file')
    parser.add_argument('--verbose', action='store_true', help='verbose information')

    parser.add_argument('--run-perf', action='store_true', help='eval perf')

    args = parser.parse_args(cmds)
    return args


def main(cmds=None):
    from lib.config import config, update_config
    args = parse_args(cmds)
    # print(args)
    update_config(config, args)
    # print(config)

    logger, final_output_dir, tb_log_dir = \
        create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    def getModel(config):
        from lib.models.hrnet2 import get_face_alignment_net
        model = get_face_alignment_net(config)

        gpus = list(config.GPUS)
        model = nn.DataParallel(model, device_ids=gpus)

        # device = torch.device("cpu")
        # model.to(device)
        # load model
        checkpoint = torch.load(args.model_file)
        if checkpoint.get('state_dict'):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if isinstance(state_dict,torch.nn.DataParallel):
            state_dict = state_dict.state_dict()
        elif isinstance(state_dict, dict):
            pass
        model.load_state_dict(state_dict, False)
        del checkpoint
        return model

    model = getModel(config)

    def preprocess(img):
        # print(img.shape)
        # height, width = img.shape[:2]
        # raw_img = img
        if img.shape[0:2] != (256,256):
            img = cv2.resize(img, (256,256))

        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        #print("input_image.shape:", input_image.shape)
        input_image = input_image.astype(np.float32)
        input_image = (input_image/255.0 - mean) / std
        input_image = input_image.transpose([2, 0, 1])

        input_tensor = torch.tensor(input_image)
        input_tensor = input_tensor.unsqueeze(0)
        #print("input_tensor.shape", input_tensor.shape)
        return input_tensor

    def postProcess(score_map):
        coords = get_preds(score_map)  # float type
        scale = 256/200
        center = torch.Tensor([127, 127])
        preds = transform_preds(coords[0], center, scale, [64, 64])
        return preds

    def modelPredict(x):
        return model(x).data.cpu()

    img_paths, video_id = guessSource(args.input)
    if img_paths:
        imgs = genImgs(img_paths)
        waitTime = 0
        print("use images")
    else:
        imgs = generate_camera(video_id)
        waitTime = 30
        print("use camera")

    for i,img in enumerate(imgs):
        # img = img[100:300,:]
        t0 = time.time()
        if img.shape[0:2] != (256,256):
            img = cv2.resize(img, (256,256))

        input_tensor = preprocess(img)
        score_map = modelPredict(input_tensor)
        preds = postProcess(score_map)
        print("time: ", time.time() - t0)
        for i in range(0, 68):
            cv2.circle(img, (int(preds[i][0]), int(preds[i][1])), 2, (0,255,0), -1)

        cv2.imshow('img_%d' % i, img)
        k = cv2.waitKey(waitTime)
        if k == 27:
            break
    # cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
