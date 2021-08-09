# op unsample



yolov3 有正常的 upsample 层。

``` ini
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61
```


``` python
	elif mdef['type'] == 'upsample':
		if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
			g = (yolo_index + 1) * 2 / 32  # gain
			modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
		else:
			modules = nn.Upsample(scale_factor=mdef['stride'])
```				

anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback