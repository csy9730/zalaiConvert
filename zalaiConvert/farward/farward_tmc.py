import time
import os,sys

import numpy as np

from zalaiConvert.farward.farward_utils import activateEnv
from zalaiConvert.farward.farward_utils import getRknn

activateEnv()

def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rknn predict & show key points')
    parser.add_argument('model', help='model file path')
    parser.add_argument('--input', '-i', help='image file path')
    parser.add_argument('--output', '-o', help='save output image name')
    parser.add_argument('--config')

    # parser.add_argument('--target', choices=['rk1808', 'rv1126'], help='target device: rk1808, rk1126')
    parser.add_argument('--device', choices=['rk1808', 'rv1126'], help='device: rk1808, rv1126')
    parser.add_argument('--device-id')
    parser.add_argument('--task', choices=['segment', 'detect', 'classify', 'keypoint'], default='keypoint', help='device: rk1808, rk1126')
    parser.add_argument('--run-perf', action='store_true', help='eval perf')

    return parser.parse_args(cmds)
    
def main(cmds=None):
    args = parse_args(cmds)
    
    rknn = getRknn(args.model, device=args.device, device_id=args.device_id)
    if rknn is None:
        exit(-1)

    tocs = []
    for i in range(10):
        x = np.random.random((1, 3, 256, 256)).astype(np.float32)
        tic = time.time()
        y = rknn.inference(inputs= [x] )
        tocs.append(time.time()-tic)
        print(i, tocs[-1])
        print(y[0].shape)
    print("mean toc", sum(tocs)/len(tocs))


if __name__ == "__main__":
    main()
