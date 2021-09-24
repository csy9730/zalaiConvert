import time
import os,sys

import numpy as np

from zalaiConvert.utils.farward_utils import activateEnv, fDumpMat
from zalaiConvert.utils.rknn_utils import getRknn

activateEnv()

def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rknn predict & show key points')
    parser.add_argument('model', help='model file path')
    parser.add_argument('--input', '-i', help='image file path')
    parser.add_argument('--output', '-o', help='save output image name')
    parser.add_argument('--config')
    parser.add_argument('--shape-list', type=int, nargs='*', default=(1,3,256,256),
                        help='shape list, such as: 1 3 256 256 (batch channel height width)')

    parser.add_argument('--device', choices=['rk1808', 'rv1126'], help='device: rk1808, rv1126')
    parser.add_argument('--device-id')
    parser.add_argument('--repeat-times', type=int, default=10, help='repeat forward N times(default 10)')
    parser.add_argument('--run-perf', action='store_true', help='eval perf')

    return parser.parse_args(cmds)
    
def main(cmds=None):
    args = parse_args(cmds)
    
    rknn = getRknn(args.model, device=args.device, device_id=args.device_id)
    if rknn is None:
        exit(-1)

    tocs = []
    for i in range(args.repeat_times):
        x = np.random.random(tuple(args.shape_list)).astype(np.float32)
        tic = time.time()
        yy = rknn.inference(inputs=[x])
        tocs.append(time.time()-tic)
        print(i, tocs[-1], [y.shape for y in yy])

    fDumpMat(args.output, yy[0])
    print("mean toc", sum(tocs)/len(tocs))


if __name__ == "__main__":
    main()
