from zalaiConvert.convert.onnx2rknn import parse_args, model2Rknn, farward

def main(cmds=None):
    args = parse_args(cmds)
    
    args.framework = args.framework or 'darknet'

    opt = vars(args)
    model = opt.pop("model");
    output = opt.pop("output");
    dataset = opt.pop("dataset")
    model2Rknn(model, output, dataset, **opt)

    if args.use_farward:
        img_path = "./test.jpg"
        farward(img_path, **opt)
    
if __name__ == '__main__':
    main()