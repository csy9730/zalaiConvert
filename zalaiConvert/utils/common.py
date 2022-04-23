import sys
import json
import time
import os
from enum import Enum
import collections
import logging


class TrainProc(Enum):
    ON_START = "on_start"  # 程序调用开始，与ON_EXIT成对
    ON_ARGPARSE_START = "on_argparse_start"  # python执行命令行解析之前
    ON_ARGPARSE_END = "on_argparse_end"  # python执行命令行解析结束

    # train总过程比较复杂，分为多个子过程：pretrain/train/evalute；
    # predict/export/test过程比较简单，所以不用划分子过程
    ON_TRAIN_START = "on_train_start"  # 开始训练
    # ON_PRETRAIN_START = "on_prepare_start" # 训练之前的数据准备网络准备开始
    # ON_LOAD_START = "on_load_start" # 加载数据开始
    # ON_LOAD_END = "on_load_end" # 结束加载数据
    # ON_PRETRAIN_END = "on_prepare_end" # 结束准备
    ON_READYDATA_START = "on_readydata_start" # 训练之前的数据准备网络准备开始
    ON_READYDATA_END = "on_readydata_end" # 结束准备
    ON_FIT_START = "on_fit_start" # 拟合开始
    ON_FIT_END = "on_fit_end" # 拟合准备
    ON_TRAIN_END = "on_train_end"  # 结束训练

    ON_EVALUATE_START = "on_evaluate_start"  # 开始评估模型性能
    ON_EVALUATE_END = "on_evaluate_end"  # 结束评估模型性能

    ON_PREDICT_START = "on_predict_start"  # 开始预测
    ON_PREDICT_END = "on_predict_end"  # 结束预测
    ON_EXPORT_START = "on_export_start"  # 开始导出
    ON_EXPORT_END = "on_export_end"  # 结束导出
    ON_GENCSV_START = "on_gencsv_start"  # 开始生成csv
    ON_GENCSV_END = "on_gencsv_end"  # 结束生成csv
    ON_UNITTEST_START = "on_unittest_start"  # 开始test
    ON_UNITTEST_END = "on_unittest_end"  # 结束test

    ON_EXIT = "on_exit"  # python退出程序之前


def dumpsDict(dct, outfile=None):
    if outfile:
        with open(outfile, 'a+') as fp:
            fp.write(json.dumps(dct, ensure_ascii=False)+"\n")
    else:
        print(json.dumps(dct, ensure_ascii=False))


def dumpsStatusDec(onStart, onEnd=None, outfile=None):
    def _fStdoutStatusDecorator(func): 
        def wrapper(*args, **kwargs):    
            dumpsDict({"stage":onStart.value, "time":time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}, outfile)
            start = time.perf_counter()
            ret=func(*args, **kwargs)
            if onEnd:
                end =time.perf_counter()
                dumpsDict({"stage":onEnd.value, "time":time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), "elapse":end-start}, outfile)
            return ret
        return wrapper
    return _fStdoutStatusDecorator


# logger = logging.getLogger(__name__)

def setLogger(name, level=logging.INFO, disable=False, log_file="zalai.log"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.disabled = disable
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))  # create a logging format
        logger.addHandler(handler)  # add the handlers to the logger
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))  # create a logging format
        logger.addHandler(console)
    return logger

# setLogger(__name__,level="DEBUG")