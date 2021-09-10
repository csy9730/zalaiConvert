import sys
import os
import argparse
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "../.."))
from zalaiConvert.utils.farward_utils import activateEnv


def main(cmd=None):
    if cmd is None:
        cmd = sys.argv[1:]
    prog_name = "zalaiConvert"
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='%s train or predict cmdline' % prog_name,
                                     usage='''%s <command> [<args>]

                Available sub-commands:
                list                  list ntb devices
                killserver            kill adb server & ntb server
                startserver           start adb server & ntb server
                convert               convert using a pretrained model
                visualization         convert with visualization tool
                activate              use rknn_strings.sh
                rknnstrings           show strings from rknn file

                zalaiConvert.convert.onnx2rknn
                zalaiConvert.farward.rknn_yolo_farward
                ''' % prog_name)

    parser.add_argument('command', help='Subcommand to run')
    args = parser.parse_args(cmd[0:1])
    if args.command in ["list", "convert", "killserver", "startserver", "visualization", "activate", "rknnstrings"]:
        if args.command == "list":
            from zalaiConvert.device_utils import main as devices
            devices(cmd[1:])
        elif args.command == "convert":
            from zalaiConvert.convertWrap import main as convert
            convert(cmd[1:])   
        elif args.command == "killserver":
            from zalaiConvert.device_utils import killserver
            killserver(cmd[1:])  
        elif args.command == "startserver":
            from zalaiConvert.device_utils import startserver
            startserver(cmd[1:])  
        elif args.command == "visualization":
            activateEnv()
            os.system('python -m rknn.bin.visualization')  
        elif args.command == "activate":
            activateEnv()
            os.system('cmd') 
        elif args.command == "rknnstrings":
            activateEnv()
            os.system('strings %s |grep version' % cmd[1])         
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
