import sys
import os
import argparse
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "../.."))

def activateEnv():
    if os.name == "nt":
        PY = os.path.dirname(os.path.abspath(sys.executable))
        os.environ['PATH'] = ";".join([
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
            os.path.join(PY, r"Lib\site-packages\rknn\api\lib\hardware\LION\Windows_x64"),
            os.path.join(PY, r"Lib\site-packages\~knn\api\lib\hardware\Windows_x64"),
            os.path.join(PY, r"Library/bin"),
            os.environ.get('PATH')
        ])

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

                ''' % prog_name)

    parser.add_argument('command', help='Subcommand to run')
    args = parser.parse_args(cmd[0:1])
    if args.command in ["list", "convert", "killserver", "startserver", "visualization"]:
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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
