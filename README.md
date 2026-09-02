# AERNet 复现与优化

本项目用于复现论文《AERNet: An Attention-Guided Edge Refinement Network and a Dataset for Remote Sensing Building Change Detection》，并在复现原始 AERNet 的基础上，对网络主干结构进行改进。原论文与代码仓库地址为：[https://github.com/zjd1836/AERNet](https://github.com/zjd1836/AERNet)。

AERNet 面向遥感建筑变化检测任务，输入同一区域两个时相的遥感图像，输出像素级建筑变化区域。该任务不仅要求模型判断“哪里发生了变化”，还要求边界尽可能贴合建筑轮廓。因此，边缘细节、空间分辨率保持能力和多尺度特征融合能力对最终效果影响很大。

## 项目内容

当前工程主要包含三部分：

- 对原始 AERNet 网络结构和 HRCUS-CD 数据集训练流程的复现；
- 基于 PyTorch 的训练、验证、指标记录与 checkpoint 保存流程；
- 将原始 ResNet34 主干替换为 `HRNet-W18-Small-v2 + 五级 1x1 通道适配器 + ImageNet 预训练` 的改进版本。

主要目录如下：
```text
attention/: BAM、CoordAtt 等注意力模块
model/: AERNet相关网络模型，包含后续改进的网络模型
HRCUS-CD/: 数据集目录
runs/: 训练日志与 checkpoint
config.py: 实验配置
train.py: 训练与验证入口
```

## 优化动机

原始 AERNet 使用 ResNet34 作为 Siamese 编码器，两时相图像 A/B 共享同一个 backbone，再将对应层特征送入后续解码器和边缘细化模块。ResNet34 的优势是结构成熟、预训练权重易获得、语义表达能力稳定，但它在前向传播中会逐级下采样，高分辨率空间信息主要依靠浅层特征保留。对于建筑变化检测来说，这会带来一个明显问题：模型可以较好地学习变化区域的语义，但对细长建筑、屋顶边缘、密集建筑群边界和小尺度变化目标的定位容易变粗。

HRNet 的核心特点是始终保留一条高分辨率分支，并通过多分辨率分支之间的反复交换融合，同时获得空间细节和语义信息。相比“先降采样再上采样”的典型 CNN backbone，HRNet 对边界友好的特征表达更适合遥感建筑变化检测。建筑变化区域通常面积占比较小，且评价指标对假阴性和边界错位很敏感，因此保持高分辨率特征对于提升 Recall、F1 和 IoU 具有直接意义。

基于这一判断，本项目将主干网络从 ResNet34 替换为 HRNet-W18-Small-v2。选择 W18-Small-v2 的原因是它在参数量、显存占用和特征质量之间较为平衡，适合在原 AERNet 解码器不大规模重写的前提下完成结构优化。

## 优化方案

改进后的主干结构为：

```text
HRNet-W18-Small-v2 + 五级 1x1 通道适配器 + ImageNet 预训练
```

具体实现位于 `model/network_HRNet.py`。

### 1. 保持 AERNet 的 Siamese 框架

原始 AERNet 的输入是两个时相图像 `A` 和 `B`，二者经过共享权重的编码器提取特征。改进版本继续保留这一设计：

```text
A 图像 -> HRNet encoder -> 五级特征
B 图像 -> HRNet encoder -> 五级特征
五级双时相特征 -> 原 AERNet decoder/refinement
```

这样做的好处是优化集中在 backbone 上，原有的差异建模、注意力模块、解码器和边缘细化逻辑仍然可以复用，便于和原始 ResNet34 版本进行公平对比。

### 2. 使用 HRNet 输出五级多尺度特征

原始解码器期望接收五级特征，并且各级分辨率和通道数需要与 ResNet34 版本的接口对齐。HRNet-W18-Small-v2 通过 `timm.create_model(..., features_only=True, out_indices=(0, 1, 2, 3, 4))` 输出五级特征，其下采样倍率为：

```text
1/2, 1/4, 1/8, 1/16, 1/32
```

这与原 AERNet 解码器的多尺度输入形式相匹配，能够在不重写解码器的情况下接入 HRNet。

### 3. 加入五级 1x1 通道适配器

HRNet 输出通道数为：

```text
64, 128, 256, 512, 1024
```

而原解码器期望的五级通道数为：

```text
64, 64, 128, 256, 512
```

因此在 HRNet 后加入五个 `1x1 Conv + BatchNorm + ReLU` 通道适配器，将 HRNet 特征转换为解码器需要的通道格式：

```text
[64, 128, 256, 512, 1024] -> [64, 64, 128, 256, 512]
```

`1x1` 卷积只改变通道维度，不破坏空间分辨率，适合作为 backbone 与 decoder 之间的轻量接口层。它既保留了 HRNet 的高分辨率空间优势，又避免了大规模修改原始 AERNet 解码结构。

### 4. 使用 ImageNet 预训练权重

HRNet-W18-Small-v2 使用本地 ImageNet 预训练权重：

```text
model/pretrained/hrnet_w18_small_v2.gluon_in1k.safetensors
```

预训练权重为遥感变化检测任务提供更稳定的低层纹理、边缘和中高层语义初始化。对于 HRCUS-CD 这类建筑变化检测数据集，预训练能够降低从零训练的不稳定性，并帮助模型更快学习到建筑区域与背景区域的可分特征。

## 训练方法

数据集目录需保持如下结构：

```text
HRCUS-CD/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

训练配置在 `config.py` 中修改。当前配置为示例：

```python
name = "<实验名>"
epochs = 50
batch = 32
batch_val = 32
lr0 = 1e-4
weight_decay = 1e-4
device = "cuda"
amp = True
```

运行训练：

```bash
python train.py
```

训练过程中会在 `runs/<实验名>/` 下保存：

- `results.csv`：每个 epoch 的训练损失、验证损失和指标；
- `last.pth`：最近一轮 checkpoint；
- `best.pth`：验证集 F1 最优 checkpoint。

## 实验结果

以下结果来自验证集上 `best.pth` 的输出日志。

| 模型 | ckpt | val_loss | F1 | IoU | Precision | Recall | OA | Kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial / ResNet34 | best.pth | 0.0217 | 0.5943 | 0.4228 | 0.9217 | 0.4385 | 0.9719 | 0.5817 |
| HRNet-W18-Small-v2 | best.pth | 0.0233 | 0.6356 | 0.4658 | 0.8970 | 0.4922 | 0.9770 | 0.6248 |

与初始 ResNet34 版本相比，HRNet-W18-Small-v2 改进后的主要变化为：

- F1 从 `0.5943` 提升到 `0.6356`，提升 `0.0413`；
- IoU 从 `0.4228` 提升到 `0.4658`，提升 `0.0430`；
- Recall 从 `0.4385` 提升到 `0.4922`，提升 `0.0537`；
- OA 从 `0.9719` 提升到 `0.9770`，提升 `0.0051`；
- Kappa 从 `0.5817` 提升到 `0.6248`，提升 `0.0431`。

可以看到，HRNet 版本的 Precision 略有下降，但 Recall、F1、IoU 和 Kappa 均有明显提升。这说明模型在变化建筑区域上检出能力更强，漏检减少，整体变化区域重叠质量更好。对于建筑变化检测任务而言，Recall 和 IoU 的提升通常意味着模型对小目标、边缘区域和不规则建筑轮廓的响应更充分，这与 HRNet 高分辨率分支带来的空间细节优势是一致的。

## 优化成效分析

本次优化的核心收益不只是更换了一个更强的分类 backbone，而是让 AERNet 的“边缘引导细化”思想获得了更适合的特征基础。原网络后端已经包含注意力模块、逐级解码和边缘 refine 机制，但如果编码器在前期下采样过程中丢失了较多边缘信息，后续 refine 模块只能在较粗的特征上做修正。HRNet 持续保留高分辨率分支，使解码阶段能够接收到更完整的建筑轮廓、道路间隔、屋顶边线等空间信息，因此更有利于生成边缘清晰的变化掩码。

从指标上看，HRNet 版本的 Recall 提升最明显，说明模型更愿意把真实变化区域识别出来；F1 和 IoU 同时提升，说明这种 Recall 增益并不是单纯扩大预测区域带来的无效增益，而是在整体重叠质量上也取得了改善。Precision 小幅下降，表明模型在更积极检测变化区域时引入了一些额外误检，这是后续仍可继续优化的方向，例如进一步调整损失函数、边缘监督权重、阈值策略或加入更细粒度的边界约束。

总体而言，`HRNet-W18-Small-v2 + 五级 1x1 通道适配器 + ImageNet 预训练` 的优化版本在尽量保留 AERNet 原有结构的同时，提高了主干网络对空间细节的表达能力，使模型更适合建筑变化检测中对边缘精度和小目标召回率要求较高的场景。

## 依赖说明

原始 AERNet 仓库给出的基础环境包括 Python、PyTorch、torchvision 和 CUDA。本复现与优化工程在此基础上还使用了：

- `timm`：构建 HRNet-W18-Small-v2 backbone；
- `safetensors`：加载 HRNet 本地预训练权重；
- `opencv-python`：读取 TIFF 格式遥感图像；
- `numpy`、`tqdm`：数据处理与训练进度显示。

实际安装时应根据本机 CUDA 与 PyTorch 版本选择匹配的安装命令。

## 参考

J. Zhang et al., "AERNet: An Attention-Guided Edge Refinement Network and a Dataset for Remote Sensing Building Change Detection," IEEE Transactions on Geoscience and Remote Sensing, 2023.
