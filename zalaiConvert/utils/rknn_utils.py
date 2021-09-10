import os
import sys
from rknn.api import RKNN

def rknn_query_model(model):
    rknn = RKNN() 
    mcfg = rknn.fetch_rknn_model_config(model)
    if mcfg:
        print(mcfg.get("target_platform"), "version=", mcfg["version"])
        print(mcfg.get('ori_network_platform'))
        print("pre_compile=", mcfg["pre_compile"])
    return mcfg


def get_io_shape(mcfg):
    mt = mcfg["norm_tensor"]
    mg = mcfg["graph"]

    in_shape = []
    out_shape = []
    for i, g in enumerate(mg):
        sz = mt[i]['size']
        sz.reverse()
        if g['left']=='output':
            out_shape.append(sz)
        else:
            in_shape.append(sz)
    return in_shape, out_shape


def getRknn(model, device=None, rknn2precompile=None, verbose=None, device_id=None, **kwargs):
    rknn = RKNN(verbose=verbose)  
    assert os.path.isfile(model)
    print('--> Loading model')  
    ret = rknn.load_rknn(model)
    if ret != 0:
        print('load_rknn failed')
        rknn.release()
        return None
    print('Load done')

    print('--> Init runtime environment')
    ret = rknn.init_runtime(target=device, device_id=device_id, eval_mem=False, rknn2precompile=rknn2precompile)
    if ret != 0:
        print('Init runtime environment failed')
        rknn.release()
        return None
    print('Init runtime done')

    rknn.model_path = model
    return rknn