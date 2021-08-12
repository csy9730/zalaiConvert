# rknn meta info

通过`strings XXX.rknn|grep version` 命令可以显示rknn文件的元信息，包括版本号，输入输出节点信息，网络结构。

该元信息是给netron网络可视化工具使用的，便于netron识别网络并可视化。


## meta info

- nodes: 保存网络节点列表，可能非常大，该属性是可选属性。
- norm_tensor：保存输入输出节点列表
    - size：尺寸
    - dim_num：尺寸的长度
    - tensorid：列表索引，从0开始
    - url: 描述节点路由，可以区分输入还是输出 （rknn1.4.0 新增）
    - (input/output) rknn1.4 可以通过url区分；rknn1.3.2以下，~~通过 input_num+output_num属性，大概可以假设是这个顺序排布~~ ,发现似乎和graph是一一对应。
- const_tensor：  
- virtual_tensor
- connection
- graph

tensor 保存输入输出和中间值（相当于可变值），nodes保存权重和节点信息（相当于固定信息）

## demo
``` json
{
    "target_platform": [
        "RK1808"
    ],
    "network_platform": "caffe",
    "input_fmt": 0,
    "input_transpose": 1,
    "name": "CatDog6",
    "version": "1.4.1b5",
    "ovxlib_version": "1.1.12",
    "node_num": 37,
    "mean_value": [
        128.0,
        128.0,
        128.0,
        128.0
    ],
    "mean_value_chns": [
        4
    ],
    "reorder": [
        0,
        1,
        2
    ],
    "input_num": 1,
    "output_num": 1,
    "nodes":[],
    "norm_tensor": [
        {
            "tensor_id": 0,
            "size": [
                2,
                1
            ],
            "dim_num": 2,
            "dtype": {
                "qnt_type": "VSI_NN_QNT_TYPE_NONE",
                "vx_type": "VSI_NN_TYPE_FLOAT16"
            },
            "url": "output_of_addmm_at_592_0"
        },
        {
            "tensor_id": 1,
            "size": [
                224,
                224,
                3,
                1
            ],
            "dim_num": 4,
            "dtype": {
                "qnt_type": "VSI_NN_QNT_TYPE_NONE",
                "vx_type": "VSI_NN_TYPE_FLOAT16"
            },
            "url": "input_of_graph/out1_53"
        }
    ],
    "graph": [
        {
            "left": "output",
            "left_tensor_id": 0,
            "right": "norm_tensor",
            "right_tensor_id": 0
        },
        {
            "left": "input",
            "left_tensor_id": 0,
            "right": "norm_tensor",
            "right_tensor_id": 1
        }
    ]
}
}
```



