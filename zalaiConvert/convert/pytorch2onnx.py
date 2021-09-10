import os
import sys
import pprint
import argparse
import cv2
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.config import config, update_config


import numpy as np

__DIR__ = os.path.dirname(os.path.abspath(__file__))


def parse_args(cmds=None):
    parser = argparse.ArgumentParser(description='pytorch model to onnx')

    parser.add_argument('--model-file', '-m', help='model parameters')
    parser.add_argument('--output', '-o', default='converted.onnx')
    parser.add_argument('--output-type', '-ot', choices=['onnx', 'torchscript'], default='onnx')
    parser.add_argument('--shape-list', type=int, nargs='*', 
                        help='shape list, such as 1 3 256 192')
    args = parser.parse_args(cmds)
    return args


def modelFactory(model_path, config=None):
    model = get_face_net(config)

    assert isinstance(model, torch.nn.Module)
    checkpoint = torch.load(model_file)
    if checkpoint.get('state_dict'):
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if isinstance(state_dict, torch.nn.DataParallel):
        state_dict = state_dict.module.state_dict()
    elif isinstance(state_dict, dict):
        print("is dict")
    model.load_state_dict(state_dict, False)

    return model

def export(model, output, output_type='onnx', shape=(1,3,256,256)):
    from torch.autograd import Variable
    x = Variable(torch.randn(*shape), requires_grad=False)

    device = torch.device("cpu")
    model.to(device)
    model = model.eval()
    with torch.no_grad():
        if output_type == 'onnx':
            torch.onnx.export(model, x, output, verbose=True, training=False,
            do_constant_folding=True)
        elif output_type == 'pytorch':
            torch.save(model, output)
        elif output_type == "torchscript":
            x = x.float()
            traced_script_module = torch.jit.trace(model.forward, x)
            traced_script_module.save(output)
        else:
            print("not found", output_type)

def main(cmds=None):
    args = parse_args(cmds)

    model = modelFactory(args.model_file, args)
    export(model, args.output, args.output_type, shape=args.shape_list)


if __name__ == "__main__":
    main()
