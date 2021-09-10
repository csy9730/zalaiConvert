import os
import sys
import time
import numpy as np
import torch
import cv2


from zalaiConvert.utils.cameraViewer import CameraViewer  
from zalaiConvert.utils.farward_utils import activateEnv, timeit, parse_args, RknnPredictor
from zalaiConvert.utils.rknn_utils import getRknn, rknn_query_model, get_io_shape

activateEnv()


COLOR_LIST = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0, 255,255]]
# COLOR_LIST = np.array(COLOR_LIST)

def maskIndex2rgb(msk, color_map):
    """mask Index to mask rgb.

    Parameters
    ----------
    msk: numpy.ndarray, (W,H,3), numpy.uint8
        Mask image.
    color_map: [[int,int,int], ...]
        color map (default: None).

    Returns
    -------
    out: numpy.ndarray, (W,H,3), numpy.uint8
       colormap.

    """
    out = np.zeros((*msk.shape, 3))
    msk = np.reshape(msk, (*msk.shape, 1))
    for i in range(len(color_map)):
        out += (msk == i) * color_map[i]
    return out

def label_to_color(img, colormap):
    color_img = np.zeros((*img.shape,3)).astype('uint8')
    for i in range(len(colormap)):
        _temp_map = (img==i)
        _temp_map = (_temp_map.repeat(3)).reshape(*img.shape[0:2], 3)
        color_img += _temp_map*np.array(colormap[i]).astype('uint8')
    return color_img

def merge_background_and_color(back_ground, color_img):
    new_img = np.zeros(back_ground.shape)
    c_map = color_img.sum(axis=2)
    new_img = back_ground * (c_map==0).repeat(3).reshape(*c_map.shape,3) \
        + color_img* (c_map>0).repeat(3).reshape(*c_map.shape,3)
    return new_img.astype('uint8')


class RknnSegPredictor(RknnPredictor):
    def __init__(self, rknn):
        self.rknn = rknn
        self.width, self.height = 416, 416
        self.guess_cfg()
        self._axis = 0 # CHW for pytorch/onnx

    def guess_cfg(self):
        self.mcfg = rknn_query_model(self.rknn.model_path)
        self.in_shape, self.out_shape = get_io_shape(self.mcfg)
        
        if self.mcfg.get("target_platform") == 'tensorflow':
            self._axis = 2 # HWC for tensorflow

        # self.NUM_CLS = self.out_shape[0][1]
        self.width, self.height = self.in_shape[3], self.in_shape[2]
        print(self.in_shape, self.out_shape)

    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs[-1]

    def postProcess(self, kpts):
        print(kpts.shape)
        kpts = kpts.squeeze(0)
        # output_img = np.transpose(output_img, (1, 2, 0))

        output_img = np.argmax(kpts, axis=self._axis)
        output_img = label_to_color(output_img, COLOR_LIST)
    
        return output_img

    def draw(self, img, preds):
        return merge_background_and_color(img, preds)

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

    model = RknnSegPredictor(rknn)

    predictWrap(args.input, model, args)
    print("__________________exit__________________")

if __name__ == "__main__":
    cmds = [r'H:\Project\zal2\rknn_facepoint_farward_lite\face98.rknn', '-i', 
        r'H:\Project\zal2\rknn_facepoint_farward_lite\a6.png', '--device', 'rk1808', '-oabc{i}.png']
    main()
