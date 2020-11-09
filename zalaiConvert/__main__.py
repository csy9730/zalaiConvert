import sys
import os
import argparse
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "../.."))


def main(cmd=None):
    if cmd is None:
        cmd = sys.argv[1:]
    prog_name = "zalaiConvert"
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='%s train or predict cmdline' % prog_name,
                                     usage='''%s <command> [<args>]

                Available sub-commands:
                train                 Trains a model
                predict               Predicts using a pretrained model
                gencsv                generate csv file from data file or sample follder 
                ''' % prog_name)

    parser.add_argument('command', help='Subcommand to run')
    args = parser.parse_args(cmd[0:1])
    if args.command in ["list", "convert"]:
        if args.command == "list":
            from zalaiConvert.device_utils import main as devices
            devices(cmd[1:])
        elif args.command == "convert":
            from zalaiConvert.convertWrap import main as convert
            convert(cmd[1:])   
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
