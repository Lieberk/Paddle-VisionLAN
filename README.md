# Paddle-VisionLAN

## 目录

- [1. 简介]()
- [2. 数据集准备]()
- [3. 复现精度]()
- [4. 模型目录与环境]()
    - [4.1 目录介绍]()
    - [4.2 准备环境]()
- [5. 开始使用]()
    - [5.1 模型训练]()
    - [5.2 模型评估]()
    - [5.3 模型预测]()
- [6. 模型推理开发]() 
- [7. 自动化测试脚本]()
- [8. LICENSE]()
- [9. 模型信息]()

## 1. 简介
**论文:** [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://ieeexplore.ieee.org/document/9711299/)

这篇论文有别于以往的分步两阶段工作需要先进行视觉预测再利用语言模型纠正的策略，该工作提出了视觉语言网络 Vision-LAN，直接赋予视觉模型语言能力，将视觉和语言模型当作一个整体。由于语言信息是和视觉特征一同获取的，不需要额外的语言模型，Vision-LAN显著提高39%的前向速度，并且能够自适应考虑语言信息来增强视觉特征，进而达到更高的识别准确率。 

[aistudio在线运行](https://aistudio.baidu.com/aistudio/projectdetail/4551567)

**参考repo:** [VisionLAN](https://github.com/wangyuxin87/VisionLAN)


## 2. 数据集准备

MJSynth和SynthText都是合成生成的数据集，其中单词实例被放置在自然场景图像中，同时考虑到场景布局。 SynthText数据集由80万张图像和大约800万合成词实例组成。每个文本实例都用其文本字符串、单词级和字符级的边界框进行注释。

MJSynth和SynthText数据集下载：[训练集](https://aistudio.baidu.com/aistudio/datasetdetail/168907) 分别解压后放在./datasets/train/下

评估数据集下载：[测试集](https://aistudio.baidu.com/aistudio/datasetdetail/168908) 解压后放在./datasets/下

数据目录的结构为：
```bash 
datasets
├── evaluation
│   ├── Sumof6benchmarks
│   ├── CUTE
│   ├── IC13
│   ├── IC15
│   ├── IIIT5K
│   ├── SVT
│   └── SVTP
└── train
    ├── MJSynth
    └── SynText
```

## 3. 复现精度

|        Methods       	 |        IIIT5K       	| IC13       	| SVT        	| IC15      	| SVTP      	| CUTE      	|
|:------------------:    |:------------------:	|:---------:	|:------:   	|:---------:	|:---------:	|:---------:	|
|        论文       	     |         95.8         |    95.7   	|     91.7   	|    83.7   	|    86.0       |    88.5       |
|        官方repo         | 	       95.9         |    96.3  	    |     90.7   	|    84.1   	|    85.3       |    88.9       |
|        复现repo         | 	       95.9         |    96.3  	    |     90.9   	|    84.1   	|    85.4       |    89.2       |

## 4. 模型目录与环境

### 4.1 目录介绍

```
    |--images                         # 测试使用的样例图片
    |--deploy                         # 预测部署相关
        |--export_model.py            # 导出模型
        |--infer.py                   # 部署预测
    |--datasets                       # 训练和测试数据集
    |--lite_data                      # 用于tipc的小数据集
    |--logs                           # 训练日志信息  
    |--output                         # 模型输出文件
    |--modules                        # 论文模块
        |--modules.py                 # 模块组件
        |--resnet.py                  # resnet45模型
    |--test_tipc                      # tipc代码
    |--utils                          # 工具代码
    |--VisionLAN                      # 论文模型
    |--predict.py                     # 预测代码
    |--eval.py                        # 评估代码
    |--train.py                       # 训练代码
    |----README.md                    # 用户手册
```

### 4.2 准备环境

- 框架：
  - PaddlePaddle >= 2.3.1
- 环境配置：使用`pip install -r requirement.txt`安装依赖。
  
## 5. 开始使用
### 5.1 模型训练
#### Language-free (LF) process

- Step 1 (LF_1): first train the vision model without MLM.

`python train.py --cfg_type LF_1 --batch_size 384 --epochs 8 --output_dir './output/LF_1/'`

- Step 2 (LF_2): finetune the MLM with vision model.

`python train.py --cfg_type LF_2 --batch_size 220 --epochs 4 --output_dir './output/LF_2/' --pretrained './output/LF_1/best_acc_M.pdparams'`

#### Language-aware (LA) process

`python train.py --cfg_type LA --batch_size 220 --epochs 8 --output_dir './output/LA/' --pretrained './output/LF_2/best_acc_M.pdparams'`

部分训练日志如下所示：
```
Epoch: 0, Iter: 200/36981, Loss VisionLAN: 2.9258, avg_reader_cost: 0.0170, avg_batch_cost: 1.5738, avg_ips: 243.9994
train accuracy: 
Accuracy: 0.001270, AR: 0.060912, CER: 0.939088, WER: 0.998730
Epoch: 0, Iter: 400/36981, Loss VisionLAN: 2.7062, avg_reader_cost: 0.0006, avg_batch_cost: 1.5246, avg_ips: 251.8685
train accuracy: 
Accuracy: 0.016068, AR: 0.140835, CER: 0.859165, WER: 0.983932
Epoch: 0, Iter: 600/36981, Loss VisionLAN: 2.6401, avg_reader_cost: 0.0006, avg_batch_cost: 1.5233, avg_ips: 252.0864
train accuracy: 
Accuracy: 0.035286, AR: 0.152361, CER: 0.847639, WER: 0.964714
```

模型训练权重保存到./output文件下, 训练日志保存到./logs文件下

可以将训练好的[模型权重下载](https://aistudio.baidu.com/aistudio/datasetdetail/170228) 解压后放在本repo/下，直接运行下面5.2评估和5.3预测部分。

### 5.2 模型评估

- 模型评估：`python eval.py`

输出评估结果：
```
------Average on 6 benchmarks--------
test accuracy: 
Accuracy: 0.913493, AR: 0.964586, CER: 0.035414, WER: 0.086507, best_acc: 0.913493
------IIIT--------
test accuracy: 
Accuracy: 0.959000, AR: 0.982448, CER: 0.017552, WER: 0.041000, best_acc: 0.959000
------IC13--------
test accuracy: 
Accuracy: 0.962660, AR: 0.987546, CER: 0.012454, WER: 0.037340, best_acc: 0.962660
------IC15--------
test accuracy: 
Accuracy: 0.840972, AR: 0.935282, CER: 0.064718, WER: 0.159028, best_acc: 0.840972
------SVT--------
test accuracy: 
Accuracy: 0.908810, AR: 0.972866, CER: 0.027134, WER: 0.091190, best_acc: 0.908810
------SVTP--------
test accuracy: 
Accuracy: 0.854264, AR: 0.933193, CER: 0.066807, WER: 0.145736, best_acc: 0.854264
------CUTE--------
test accuracy: 
Accuracy: 0.892361, AR: 0.950439, CER: 0.049561, WER: 0.107639, best_acc: 0.892361
```

### 5.3 模型预测



- 模型预测：`python predict.py --img_file './images/demo1.png'`

预测图片demo1.png结果如下：
```
pre_string: residencia
```

## 6. 模型推理开发

- 模型动转静导出：
```
python deploy/export_model.py
```
输出结果：
```
inference model has been saved into deploy
```

- 基于推理引擎的模型预测：
```
python deploy/infer.py --img_path ./images/demo1.png
```
输出结果：
```
image_name: images/demo1.png, predict data: residencia
```

## 7. 自动化测试脚本
为了方便快速验证训练/评估/推理过程，建立了一个小数据集，放在lite_data文件夹下
- tipc 所有代码一键测试命令
```
bash test_tipc/test_train_inference_python.sh test_tipc/configs/VisionLAN/train_infer_python.txt lite_train_lite_infer 
```

结果日志如下
```
[33m Run successfully with command - python3.7 train.py --data_path=./lite_data --test_batch_size=1 --show_interval=2 --test_interval=5 --output_dir=./test_tipc/output/   --epochs=3 --batch_size=5  !  [0m
[33m Run successfully with command - python3.7 eval.py --data_path ./lite_data  --pretrained './test_tipc/output/best_acc_M.pdparams'       !  [0m
[33m Run successfully with command - python3.7 deploy/export_model.py    --save_inference_dir=./test_tipc/output/VisionLAN/lite_train_lite_infer/norm_train_gpus_0!  [0m
[33m Run successfully with command - python3.7 deploy/infer.py --use_gpu=True --save_inference_dir=./test_tipc/output/VisionLAN/lite_train_lite_infer/norm_train_gpus_0 --batch_size=1   --benchmark=False --img_path=./images/demo1.png > ./test_tipc/output/VisionLAN/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !  [0m
[33m Run successfully with command - python3.7 deploy/infer.py --use_gpu=False --save_inference_dir=./test_tipc/output/VisionLAN/lite_train_lite_infer/norm_train_gpus_0 --batch_size=1   --benchmark=False --img_path=./images/demo1.png > ./test_tipc/output/VisionLAN/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 !  [0m
```

## 8. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 9. 模型信息

| 信息 | 描述 |
| --- | --- |
| 作者 | Lieber|
| 日期 | 2022年9月 |
| 框架版本 | PaddlePaddle==2.3.1 |
| 应用场景 | 场景文本检测 |
| 硬件支持 | GPU、CPU |
| 在线体验 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/4551567)
