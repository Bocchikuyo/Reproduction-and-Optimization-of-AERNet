"""
-----------------------------------------------------------------------------
    源码内配置区
-----------------------------------------------------------------------------
    使用方式：
    1. 直接修改下面这些配置项
    2. 运行 train.py
-----------------------------------------------------------------------------
"""
import types
from pathlib import Path

CFG = types.SimpleNamespace(
    # 数据与实验保存
    data_root=Path("HRCUS-CD"),
    project=Path("runs"),
    name="03_HRNet-W18-Small-v2",
    save_dir=Path("runs") / "03_HRNet-W18-Small-v2",
    resume=None,

    # 训练超参数
    epochs=50,
    batch=32,
    batch_val=32,
    # Windows 下如果 DataLoader 多进程不稳定，优先设为 0 走单进程读取。
    # 面向源码学习时，0 也是最省心的默认值。
    # 后续如果你确认环境稳定，再尝试改成 2、4 这类值提升数据读取速度。
    workers=4,
    lr0=1e-4,
    weight_decay=1e-4,

    # 运行控制
    seed=42,
    device="cuda",
    amp=True,
    log_interval=50,
    save_csv=True,
)
"""
       Epoch     GPU_mem    train_loss    val_loss        F1       IoU   Precision    Recall        OA     Kappa          lr
        47/50       4.95G        0.0065      0.0223    0.6050    0.4337      0.9116    0.4527    0.9734    0.5928    0.000002
       Epoch     GPU_mem    train_loss    val_loss        F1       IoU   Precision    Recall        OA     Kappa          lr
        47/50       5.11G        0.0071      0.0211    0.5883    0.4167      0.9315    0.4299    0.9709    0.5753    0.000002
       Epoch     GPU_mem    train_loss    val_loss        F1       IoU   Precision    Recall        OA     Kappa          lr
        45/50       5.12G        0.0074      0.0217    0.5943    0.4228      0.9217    0.4385    0.9719    0.5817    0.000004

"""
"""
Initial
        ckpt    val_loss        F1       IoU   Precision    Recall        OA     Kappa
    best.pth      0.0217    0.5943    0.4228      0.9217    0.4385    0.9719    0.5817
FC-Siam-diff
        ckpt    val_loss        F1       IoU   Precision    Recall        OA     Kappa
    best.pth      0.0260    0.5898    0.4182      0.9116    0.4359    0.9717    0.5770
DCNv2
        ckpt    val_loss        F1       IoU   Precision    Recall        OA     Kappa
    best.pth      0.0232    0.5793    0.4078      0.9297    0.4207    0.9699    0.5660
HRNet-W18-Small-v2
        ckpt    val_loss        F1       IoU   Precision    Recall        OA     Kappa
    best.pth      0.0233    0.6356    0.4658      0.8970    0.4922    0.9770    0.6248
"""
