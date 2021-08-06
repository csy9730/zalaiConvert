import os
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import time
# import torch


def activateEnv(pth=None):
    if pth is None:
        pth = sys.executable
    base = os.path.dirname(os.path.abspath(pth))
    lst = [
        os.path.join(base, r"Library\mingw-w64\bin"),
        os.path.join(base, r"Library\usr\bin"),
        os.path.join(base, r"Library\bin"),
        os.path.join(base, r"Scripts"),
        os.path.join(base, r"bin"),
        os.path.join(base, r"Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64"),
        os.path.join(base, r"Lib\site-packages\~knn\api\lib\hardware\Windows_x64"),
        os.path.join(base, r"Lib\site-packages\torch\lib"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../bin"),
        base,
        os.environ.get('PATH')]
    os.environ['PATH'] = ';'.join(lst)

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


def generate_camera(cap):
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # print(img.shape)
        yield img

def genImgs(img_paths):
    for i in img_paths:
        img = cv2.imread(i, 1)
        if img is None:
            print(i, "not found")
        else:
            yield img


def getImagePaddingKeepWhRatio(in_shape, out_shape):
    """
        1. 计算目标宽高比和自身宽高比,
        2. 尽量使宽高比一致
        3. 计算填充量
        4. 宽高填充量必然有一个为0
    """
    wh_ratio = in_shape[0] / in_shape[1]
    tg_wh_ratio = out_shape[0] / out_shape[1]
    paddings = [0, 0]
    if wh_ratio < tg_wh_ratio:
        paddings[0] = int(in_shape[1] * (tg_wh_ratio - wh_ratio))
    if wh_ratio >= tg_wh_ratio:
        paddings[1] = int(in_shape[0] / tg_wh_ratio - in_shape[0] / wh_ratio)
    return paddings

def getImagePaddingKeepWhRatio2(in_shape, out_shape):
    wh_ratio = in_shape[0] / in_shape[1]
    tg_wh_ratio = out_shape[0] / out_shape[1]
    
    paddings = [0, 0]
    if in_shape[0] * out_shape[1] < out_shape[0] *in_shape[1]:
        paddings[0] = int(in_shape[1] * out_shape[0] / out_shape[1]) - in_shape[0]
    else:
        paddings[1] = int(in_shape[0] / out_shape[0] * out_shape[1]) - in_shape[1]
    return paddings


def imagePadding(img, out_shape):
    """
        图片，添加padding，执行resize。

        img: np.ndarray
        out_shape: tuple(int,int)

        1.  计算目标宽高比和自身宽高比,计算填充量
        2.  填充宽/高，
        3.  通过resize放缩为目标尺寸
    """
    in_shape = img.shape
    paddings = getImagePaddingKeepWhRatio(in_shape, out_shape)
    pad2 = (paddings[0] //2, paddings[1] // 2)

    if len(in_shape) == 2:
        b_img = np.zeros((in_shape[0] + paddings[0],
                          in_shape[1] + paddings[1])).astype(np.uint8)
        b_img[pad2[0]: in_shape[0] + pad2[0], pad2[1]: in_shape[1] + pad2[1]] = img
    else:
        b_img = np.zeros((in_shape[0] + paddings[0], in_shape[1] + paddings[1],
                          in_shape[2])).astype(np.uint8)
        b_img[pad2[0]:in_shape[0] + pad2[0], pad2[1]: in_shape[1] + pad2[1], :] = img

    rz_img = cv2.resize(b_img, out_shape[::-1])
    # ratio = wanted_size[1] / tt_image.shape[0], wanted_size[0] / tt_image.shape[1]
    return rz_img, paddings


def get_argmax_pt(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    import torch
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def pts_unclip(coords, center, scale, output_size):
    import torch
    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.tensor(transform_pixel(coords[p, 0:2], center, scale, output_size, 1, 0))
    return coords


def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def get_transform(center, scale, output_size, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1]/2
        t_mat[1, 2] = -output_size[0]/2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def rknn_query_model(model):
    rknn = RKNN() 
    mcfg = rknn.fetch_rknn_model_config(model)
    print(mcfg["target_platform"], "version=", mcfg["version"])
    print("pre_compile=", mcfg["pre_compile"])

    return mcfg

def get_io_shape(mcfg):
    mt = mcfg["norm_tensor"]
    mg = mcfg["graph"]

    in_shape = []
    out_shape = []
    for i, g in enumerate(mg):
        if g['left']=='output':
            out_shape.append(mt[i]['size'])
        else:
            in_shape.append(mt[i]['size'])
    return in_shape, out_shape

def getRknn(model, device=None, rknn2precompile=None, verbose=None, **kwargs):
    rknn = RKNN(verbose=verbose)  

    print('--> Loading model')  
    ret = rknn.load_rknn(model)
    if ret != 0:
        print('load_rknn failed')
        return None
    print('Load done')

    print('Init runtime environment')
    ret = rknn.init_runtime(target=device, eval_mem=False, rknn2precompile=rknn2precompile) # 
    if ret != 0:
        print('Init runtime environment failed')
        return None
    print('Init runtime done')

    return rknn

def rknnPredict(inputs, rknn):
    outputs = rknn.inference(inputs=inputs)
    return outputs


def preprocess(img, with_normalize=None, hwc_chw=None, **kwargs):
    # print(img.shape)
    # height, width = img.shape[:2]
    # raw_img = img
    if img.shape[0:2] != (256,256):
        img = cv2.resize(img, (256,256))
    # img = imagePadding(img, (256,256))[0]
    input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_image = input_image.astype(np.float32)

    if not with_normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        input_image = (input_image/255.0 - mean) / std

    if hwc_chw:
        input_image = input_image.transpose([2, 0, 1])

    return input_image

def postProcess(score_map):
    import torch
    # score_map = score_map.transpose([0, 2, 3, 1]) ### 
    # print(score_map.shape)
    score_map = torch.Tensor(score_map)
    coords = get_argmax_pt(score_map)  # float type
    scale = 256/200
    center = torch.Tensor([127, 127])
    preds = pts_unclip(coords[0], center, scale, [64, 64])
    return preds


COLOR_LIST = [[0,0,0],[0,0,128], [0,128,0], [128,0,0], [255,128,0], [128,0,255], [0, 255,128]]
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
    N = len(color_map)
    msk = np.reshape(msk, (*msk.shape, 1))
    for i in range(N):
        out += (msk == i) * color_map[i]
    return out


def mat2list(dat):
    dat = dat.reshape((1, -1))
    dat = dat.tolist()
    return [s for ss in dat for s  in ss]


def drawImagePoints(img, pred):
    for p in pred:
        cv2.circle(img, p, 3, (0,0,255))


def draw_predict(img, preds):
    for i in range(0, 68):
        cv2.circle(img, (int(preds[i][0]), int(preds[i][1])), 2, (0,255,0), -1)


def showImage(img, text="untitled"):
    cv2.imshow(text, img)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        raise StopIteration
    cv2.waitKey()


def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rknn predict & show key points')
    parser.add_argument('model', help='model file path')
    parser.add_argument('--input', '-i', help='image file path')
    parser.add_argument('--output', '-o', help='save output image name')
    parser.add_argument('--use-padding', action='store_true', help='model file path')
    parser.add_argument('--save-img', action='store_true', help='model file path')
    parser.add_argument('--show-img', action='store_true', help='model file path')
    parser.add_argument('--input-chw', action='store_true', help='model file path')
    parser.add_argument('--device', help='device: rk1808, rk1126')
    parser.add_argument('--task', choices=['segment', 'detect', 'classify', 'keypoint'], default='keypoint', help='device: rk1808, rk1126')
    parser.add_argument('--mix-scale', type=float, help='segment task params: mix scale')
    parser.add_argument('--verbose', action='store_true', help='verbose information')
    parser.add_argument('--with-normalize', action='store_true', help='rknn with normalize')
    parser.add_argument('--hwc-chw', action='store_true', help='image preprocess: from HWC to CHW')
    parser.add_argument('--run-perf', action='store_true', help='eval perf')

    parser.add_argument('--save-npy', action='store_true')
    
    return parser.parse_args(cmds)


def main(cmds=None):
    args = parse_args(cmds)
    model_path = args.model 
    img_path = args.input

    if args.output:
        args.save_img = True
    elif args.output is None and args.save_img:
        args.output = 'out.jpg'

    predictWrap(img_path, model_path, args)

def predictWrap(source, model_path, args):
    rknn = getRknn(model_path, device=args.device)
    if rknn is None:
        exit(-1)
        
    img_paths, video_id = guessSource(source)
    if img_paths:
        imgs = genImgs(img_paths)
        use_camera = False
        print("use images")
        title = "img_{i}"
    else:
        cap = cv2.VideoCapture(video_id)
        imgs = generate_camera(cap)
        use_camera = True
        print("use camera")
        title = "camera"

    waitTime = 30 if use_camera else 0
        
    for i, img in enumerate(imgs):
        if img.shape[0:2] != (256,256):
            img = cv2.resize(img, (256,256))

        img2 = preprocess(img, with_normalize=args.with_normalize, hwc_chw=args.hwc_chw)
        # print(img.shape)
        t0 = time.time()
        pred = rknnPredict([img2], rknn)
        print("time: ", time.time() - t0)
        # print("pred", pred, pred[0].shape)
        # print("shape", pred[0].shape)
        preds = postProcess(pred[0])
    
        if args.save_npy:
            np.save('out_{0}.npy'.format(i=i), pred[0])
        
        draw_predict(img, preds)

        # print(preds, preds.shape)
        if args.save_img:
            cv2.imwrite(args.output.format(i=i), img.astype(np.uint8))

        if args.show_img:
            cv2.imshow(title.format(i=i), img)
            k = cv2.waitKey(waitTime)
            if k == 27:
                break

    if use_camera:
        cap.release()

    cv2.destroyAllWindows()
    rknn.release()
    print("__________________exit__________________")

if __name__ == "__main__":
    # cmds += ['--use-padding', '--input-chw', '--device', 'rk1808', '--save-img', '--task', 'segment']
    main()
