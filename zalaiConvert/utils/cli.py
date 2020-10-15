import sys
import os
import json
import abc
import argparse


def _dictMerge(dct, dct_new):
    dct_new.update({k: v for k, v in dct.items() if v is not None})
    return dct_new


def namespaceMergeDict(args, cfg):
    """
        args: NameSpace
        cfg: dict
    """
    for k, v in cfg.items():
        if hasattr(args, k) and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


# 只能被继承，不能实例化，实例化会报错
class Cli(metaclass=abc.ABCMeta): 
    @abc.abstractmethod
    def parse_args(self):
        # 加完这个方法子类必须有这个方法，否则报错
        pass

    @abc.abstractmethod
    def mergeCfg(self, cfg):
        pass

    @abc.abstractmethod
    def run(self):
        pass


class CliRuner(Cli):
    def run(self, cmd=None):
        self._cmd = cmd or sys.argv[1:]
        self._args = self.parse_args(self._cmd)
        self._cfg = self.mergeCfg(self._args)
 
        self.prepare(self._cfg)
        self.handle(self._cfg)

    def handle(self, cfg):        
        pass

    def prepare(self, cfg):
        pass
        # cfg.logger = setLogger(__name__, cfg.log_level, cfg.log_disable, cfg.log_file)


class CliTrainer(CliRuner):
    @property
    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', '-i', help="input file/folder")
        parser.add_argument('--valid', '-v')
        parser.add_argument('--output', '-o', help="output file/folder")
        parser.add_argument('--config', '-c')
        parser.add_argument('--model-in', '-m')

        parser.add_argument('--batch-size', type=int)
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--workers', type=int)
        # parser.add_argument('--arch', choices=['vgg16_bn', 'vgg19_bn', 'mobilenet_v2'])
        parser.add_argument('--learning-rate', type=float)

        parser.add_argument('--gpu-on', default=False, action="store_true")
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--resume',
                        action='store_true',
                        help="resume training from last.pt ")
        group.add_argument('--unsume',
                        dest='resume',
                        action='store_false',
                        help="training from zero")

        parser.add_argument('--logging-level', '-l',
                            choices=['NOTSET', 'DEBUG', 'INFO',
                                    'WARNING', 'ERROR', 'FATAL', 'CRITICAL'],
                            default='INFO', action="store", help="the level of logging to use")
        parser.add_argument('--logging-file', '-f', default="tmp_zalAI.log", dest="log_file", action="store",
                            help="logging file")
        parser.add_argument('--logging-disable', '-lds', default=False, action="store_true", dest="log_disable",
                            help="use logging  or not")
        return parser

    def parse_args(self, cmd=None):  
        args = self.parser.parse_args(cmd)
        return args

    @property
    def defaultDict(self):
        # cfg_path = os.path.join(osp.dirname(osp.abspath(__file__)), "settings.json")
        kwargs = {
            "epochs": 4,
            "batch_size": 2,
            "learning_rate": 0.001,
            "workers": 0,
            "valid": None
        }
        return kwargs

    def mergeCfg(self, cfg):
        json_config = cfg.config
        if json_config:
            with open(json_config, 'r') as f:
                params_json = json.load(f)
                namespaceMergeDict(cfg, params_json)

        namespaceMergeDict(cfg, self.defaultDict)
        
        return cfg

    def prepare(self, cfg):
        from zalai.common.logger import setLogger
        cfg.logger = setLogger(__name__, cfg.log_level, cfg.log_disable, cfg.log_file)
        if cfg.resume and (not cfg.model_in):
            cfg.model_in = cfg.output


class CliPredictor(CliRuner):
    @property
    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', '-i',dest='test_path', help='where input image ')
        parser.add_argument('--output', '-o', help='where output img ')
        parser.add_argument('--model-in', "-m", help='pb model saved')
        parser.add_argument('--config', "-c",dest='config', help='config file')
        parser.add_argument('--recorder', '-r', help='where recorder output & saved')

        parser.add_argument('--mix',action="store_true", help='mix input & output')
        parser.add_argument('--no-mix',action="store_false",dest="mix", help='mix input & output')

        parser.add_argument('--gpu-on',dest='gpu-on', action='store_true',help ="flag to use gpu,default is without gpu ")
        parser.add_argument('--gpu-off', dest='gpu-on',action='store_false',help ="flag to not use gpu")

        parser.add_argument('--view-img', action="store_true", help='view output')
        parser.add_argument('--save-img', action="store_false",
            dest='view_img', help='save output')

        return parser

    def parse_args(self, cmd=None):  
        args = self.parser.parse_args(cmd)
        return args

    @property
    def defaultDict(self):
        kwargs = {
            "workers": 0
        }
        return kwargs

    def mergeCfg(self, cfg):
        json_config = cfg.config
        if json_config:
            with open(json_config, 'r') as f:
                params_json = json.load(f)
                namespaceMergeDict(cfg, params_json)

        namespaceMergeDict(cfg, self.defaultDict)
        
        return cfg

    def prepare(self, cfg):
        from zalai.common.logger import setLogger
        cfg.logger = setLogger(__name__, cfg.log_level, cfg.log_disable, cfg.log_file)
