import random
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    # model/network.py imports cv2 but does not actually use it.
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    import cv2

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from model.metric_tool import SegEvaluator
from model.network_HRNet import zh_net

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from config import CFG

# 这里使用 ImageNet 统计量做归一化，是因为 backbone 使用的是 ResNet34 预训练权重。
# 如果后续你想完全从头训练，也可以尝试改成数据集自己的均值方差。
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

def set_seed(seed):
    # 固定随机种子，便于复现实验结果。
    # 注意：深度学习里即使固定随机种子，不同硬件和 CUDA 版本间仍可能有轻微波动。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_cfg(cfg):
    # 这个函数专门用于在训练开始前打印当前配置。
    # 对照 YOLO 系列代码来看，它相当于把本次实验的重要超参数做一次“总览”。
    print("========== Training Config ==========")
    for key, value in vars(cfg).items():
        print(f"{key}: {value}")
    print("====================================")


def get_results_file(save_dir):
    return save_dir / "results.csv"


def write_results_header(file):
    # 模仿 YOLO 的结果记录方式，把每个 epoch 的训练/验证指标写成一行。
    header = (
        "epoch,train_loss,val_loss,F1,IoU,Precision,Recall,OA,Kappa,lr\n"
    )
    file.write_text(header, encoding="utf-8")


def append_results_row(file, epoch, train_loss, val_loss, metrics, lr):
    row = (
        f"{epoch},"
        f"{train_loss:.6f},"
        f"{val_loss:.6f},"
        f"{metrics['F1']:.6f},"
        f"{metrics['1_IoU']:.6f},"
        f"{metrics['Precision']:.6f},"
        f"{metrics['Recall']:.6f},"
        f"{metrics['OA']:.6f},"
        f"{metrics['Kappa']:.6f},"
        f"{lr:.8f}\n"
    )
    with file.open("a", encoding="utf-8") as f:
        f.write(row)


def print_epoch_header():
    # 这里仿照 YOLO 的风格，先打印一行表头，后面每个 epoch 再打印一行对应结果。
    print(
        f"{'Epoch':>8}"
        f"{'GPU_mem':>12}"
        f"{'train_loss':>14}"
        f"{'val_loss':>12}"
        f"{'F1':>10}"
        f"{'IoU':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'OA':>10}"
        f"{'Kappa':>10}"
        f"{'lr':>12}"
    )


def format_gpu_mem(device):
    if device.type != "cuda":
        return "0.00G"
    mem_gb = torch.cuda.max_memory_reserved(device=device) / (1024 ** 3)
    return f"{mem_gb:.2f}G"


def print_epoch_summary(epoch, epochs, gpu_mem, train_loss, val_loss, metrics, lr):
    print(
        f"{str(f'{epoch}/{epochs}'):>8}"
        f"{gpu_mem:>12}"
        f"{train_loss:>14.4f}"
        f"{val_loss:>12.4f}"
        f"{metrics['F1']:>10.4f}"
        f"{metrics['1_IoU']:>10.4f}"
        f"{metrics['Precision']:>12.4f}"
        f"{metrics['Recall']:>10.4f}"
        f"{metrics['OA']:>10.4f}"
        f"{metrics['Kappa']:>10.4f}"
        f"{lr:>12.6f}"
    )


def print_best_validation_summary(metrics, val_loss):
    # 训练全部结束后，再单独输出一次 best.pth 在验证集上的最终结果。
    print("\nFinal best checkpoint validation:")
    print(
        f"{'ckpt':>8}"
        f"{'val_loss':>12}"
        f"{'F1':>10}"
        f"{'IoU':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'OA':>10}"
        f"{'Kappa':>10}"
    )
    print(
        f"{'best.pth':>8}"
        f"{val_loss:>12.4f}"
        f"{metrics['F1']:>10.4f}"
        f"{metrics['1_IoU']:>10.4f}"
        f"{metrics['Precision']:>12.4f}"
        f"{metrics['Recall']:>10.4f}"
        f"{metrics['OA']:>10.4f}"
        f"{metrics['Kappa']:>10.4f}"
    )


def build_train_pbar(loader, epoch, epochs):
    if tqdm is None:
        return None
    ncols = min(140, shutil.get_terminal_size((140, 20)).columns)
    return tqdm(
        enumerate(loader, start=1),
        total=len(loader),
        ncols=ncols,
        bar_format="{l_bar}{bar:10}{r_bar}",
        desc=f"Epoch {epoch}/{epochs}",
    )


class HRCUSCDDataset(Dataset):
    def __init__(self, root, split, augment=False):
        # HRCUS-CD 的每个样本由三部分组成：
        # A: 时相1图像
        # B: 时相2图像
        # label: 像素级变化掩码
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.a_dir = self.root / split / "A"
        self.b_dir = self.root / split / "B"
        self.label_dir = self.root / split / "label"
        self.names = sorted(path.name for path in self.a_dir.glob("*.tif"))

        if not self.names:
            raise FileNotFoundError(f"No .tif files found under {self.a_dir}")

        # 这里显式检查 A / B / label 是否一一对应。
        # 这样如果数据集拷贝不完整，训练前就能尽早报错。
        for name in self.names:
            if not (self.b_dir / name).exists():
                raise FileNotFoundError(f"Missing B image for {name}")
            if not (self.label_dir / name).exists():
                raise FileNotFoundError(f"Missing label for {name}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        # 这里改用 OpenCV 读取 TIFF，而不是 Pillow。
        # 原因是你当前环境里 Pillow 在解码这批 LZW 压缩 TIFF 时会直接崩溃，
        # 而 OpenCV 读取同一批文件是稳定的。
        image_a = self.read_rgb_image(self.a_dir / name)
        image_b = self.read_rgb_image(self.b_dir / name)
        label = self.read_mask_image(self.label_dir / name)

        # 变化检测任务里，A/B/label 必须做“同步增强”：
        # 如果只翻转 A，不翻转 B 或 label，样本对应关系就会被破坏。
        if self.augment:
            image_a, image_b, label = self.apply_transforms(image_a, image_b, label)

        # 返回文件名是为了以后调试、可视化或保存预测结果时能对上原图。
        image_a = self.image_to_tensor(image_a)
        image_b = self.image_to_tensor(image_b)
        label = self.mask_to_tensor(label)
        return image_a, image_b, label, name

    @staticmethod
    def read_rgb_image(path):
        # OpenCV 默认读出来是 BGR，这里转换成更常见的 RGB 顺序。
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def read_mask_image(path):
        # 标签直接以灰度方式读取，保持单通道。
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read label: {path}")
        return mask

    @staticmethod
    def apply_transforms(image_a, image_b, label):
        # 水平翻转
        if random.random() < 0.5:
            image_a = np.ascontiguousarray(np.fliplr(image_a))
            image_b = np.ascontiguousarray(np.fliplr(image_b))
            label = np.ascontiguousarray(np.fliplr(label))

        # 垂直翻转
        if random.random() < 0.5:
            image_a = np.ascontiguousarray(np.flipud(image_a))
            image_b = np.ascontiguousarray(np.flipud(image_b))
            label = np.ascontiguousarray(np.flipud(label))

        # 遥感图像通常没有严格“正方向”，所以 90 度整数倍旋转是很常见且安全的增强。
        rotations = random.randint(0, 3)
        if rotations:
            image_a = np.ascontiguousarray(np.rot90(image_a, rotations))
            image_b = np.ascontiguousarray(np.rot90(image_b, rotations))
            label = np.ascontiguousarray(np.rot90(label, rotations))

        return image_a, image_b, label

    @staticmethod
    def image_to_tensor(image):
        # numpy(HWC, RGB) -> torch.Tensor(CHW)
        # 最终形状从 HWC 变成 CHW，满足 PyTorch 卷积网络输入格式。
        array = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor

    @staticmethod
    def mask_to_tensor(mask):
        # label 原始文件是灰度图，这里统一转换成 0/1 的单通道监督信号。
        # 如果你的标签本来就是 0 和 255，这里会自动归一化到 0 和 1。
        array = np.asarray(mask)

        # 某些 TIFF 解码器会把单通道图像返回为 HWC 格式的 HxWx1。
        # 在这里统一压成二维 HxW，最后只添加一次 PyTorch 的通道维。
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        elif array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(
                f"Expected a single-channel mask with shape HxW, HxWx1 or 1xHxW, "
                f"but got {array.shape}"
            )

        array = array.astype(np.float32)
        if array.max() > 1:
            array = array / 255.0
        array = (array > 0.5).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)


def balanced_bce_with_logits(logits, target):
    # 变化检测里“未变化”像素通常远多于“变化”像素，类别严重不平衡。
    # 这里按当前 batch 中前景/背景比例动态生成权重，减轻模型只学会预测背景的问题。
    pos = target.sum()
    neg = target.numel() - pos
    if pos.item() == 0 or neg.item() == 0:
        # 如果一个 batch 恰好全是背景或全是前景，直接退化成普通 BCE，
        # 避免分母为 0 或权重异常。
        return F.binary_cross_entropy_with_logits(logits, target)

    total = pos + neg
    pos_weight = (neg / total).to(dtype=logits.dtype)
    neg_weight = (pos / total).to(dtype=logits.dtype)
    weights = torch.where(target > 0.5, pos_weight, neg_weight)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights)


def mask_to_edge(target, kernel_size=3):
    """从二值分割标签生成形态学边缘标签。"""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than or equal to 3")

    padding = kernel_size // 2
    dilated = F.max_pool2d(target, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-target, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp_(0.0, 1.0)


def compute_loss(outputs, target, edge_loss_weight=0.2):
    # 模型始终返回四个辅助输出、refined 和 seg；启用显式边缘监督的
    # 版本还会返回第七个 edge_logits。
    # al1, al2, al3, al4: 四个中间层辅助监督输出
    # refined_logits: 最终 refined 结果
    # seg_logits: refine 前的分割结果
    # edge_logits: 可选的边缘预测，用分割标签生成的边缘真值进行监督
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(
            f"Expected target shape [N, 1, H, W], but got {tuple(target.shape)}"
        )

    if len(outputs) == 6:
        al1, al2, al3, al4, refined_logits, seg_logits = outputs
        edge_logits = None
    elif len(outputs) == 7:
        al1, al2, al3, al4, refined_logits, seg_logits, edge_logits = outputs
    else:
        raise ValueError(f"Expected 6 or 7 model outputs, but got {len(outputs)}")
    aux_logits = [al1, al2, al3, al4]
    aux_weights = [0.1, 0.1, 0.1, 0.1]

    # 主损失由两个部分组成：
    # 1. refine 后的最终结果
    # 2. refine 前的分割结果
    loss = balanced_bce_with_logits(refined_logits, target)
    loss = loss + balanced_bce_with_logits(seg_logits, target)

    # 3x3 形态学梯度在目标轮廓两侧产生窄边缘带。
    # 平衡 BCE 用于缓解边缘像素远少于非边缘像素的问题。
    if edge_logits is not None:
        edge_target = mask_to_edge(target, kernel_size=3)
        edge_loss = balanced_bce_with_logits(edge_logits, edge_target)
        loss = loss + edge_loss_weight * edge_loss

    # 辅助输出分辨率更小，需要先把标签下采样到对应尺寸再计算损失。
    # nearest 不会引入灰度插值，适合二值 mask。
    for weight, aux_logit in zip(aux_weights, aux_logits):
        aux_target = F.interpolate(target, size=aux_logit.shape[-2:], mode="nearest")
        loss = loss + weight * balanced_bce_with_logits(aux_logit, aux_target)

    return loss


def build_dataloaders(cfg):
    # 训练集开启增强，验证集不做增强，保持评估稳定。
    train_dataset = HRCUSCDDataset(cfg.data_root, split="train", augment=True)
    val_dataset = HRCUSCDDataset(cfg.data_root, split="val", augment=False)
    pin_memory = cfg.device.startswith("cuda")

    # shuffle=True 只用于训练集；验证集必须固定顺序，便于复现实验和调试。
    # 当 workers=0 时，数据读取在主进程内完成；
    # 速度可能略慢一些，但在 Windows 环境下通常更稳定、更容易调试。
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_val,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, cfg):
    # train() 会启用 BatchNorm / Dropout 的训练行为。
    model.train()
    running_loss = 0.0
    total_samples = 0
    use_amp = scaler is not None
    start_time = time.time()
    lr = optimizer.param_groups[0]["lr"]
    iterator = build_train_pbar(loader, epoch, cfg.epochs)
    if iterator is None:
        iterator = enumerate(loader, start=1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step, (image_a, image_b, target, _) in iterator:
        image_a = image_a.to(device, non_blocking=True)
        image_b = image_b.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # set_to_none=True 通常比先清零再写 0 更省显存、速度也更好一点。
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            # AMP 混合精度可以减少显存占用，并在支持 Tensor Core 的 GPU 上提升速度。
            with torch.amp.autocast(device_type='cuda', enabled=True):
                outputs = model(image_a, image_b)
                loss = compute_loss(outputs, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(image_a, image_b)
            loss = compute_loss(outputs, target)
            loss.backward()
            optimizer.step()

        # 这里按样本数累计平均损失，避免最后一个不满 batch 的权重被放大。
        batch_size = image_a.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        if tqdm is not None:
            avg_loss = running_loss / max(total_samples, 1)
            gpu_mem = format_gpu_mem(device)
            iterator.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{lr:.6f}",
                mem=gpu_mem,
            )
        elif step % cfg.log_interval == 0 or step == len(loader):
            avg_loss = running_loss / max(total_samples, 1)
            elapsed = time.time() - start_time
            print(
                f"[Train] Epoch {epoch:03d} Step {step:04d}/{len(loader):04d} "
                f"Loss {avg_loss:.4f} Time {elapsed:.1f}s"
            )

    if tqdm is not None:
        iterator.close()

    return running_loss / max(total_samples, 1), format_gpu_mem(device)


@torch.no_grad()
def evaluate(model, loader, device):
    # eval() 会切换 BatchNorm / Dropout 到推理模式。
    # no_grad() 关闭梯度计算，验证时更省显存、更快。
    model.eval()
    evaluator = SegEvaluator(1)
    evaluator.reset()
    running_loss = 0.0
    total_samples = 0

    for image_a, image_b, target, _ in loader:
        image_a = image_a.to(device, non_blocking=True)
        image_b = image_b.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        outputs = model(image_a, image_b)
        loss = compute_loss(outputs, target)

        # 验证时使用 refined_logits 作为最终预测结果。
        # sigmoid 后阈值设为 0.5，得到二值变化图。
        refined_logits = outputs[4]
        prediction = (torch.sigmoid(refined_logits) > 0.5).long().cpu().numpy()
        ground_truth = target.long().cpu().numpy()
        evaluator.add_batch(gt_image=ground_truth, pre_image=prediction)

        batch_size = image_a.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    metrics = evaluator.matrix(1)
    val_loss = running_loss / max(total_samples, 1)
    return val_loss, metrics


def save_checkpoint(save_path, model, optimizer, scheduler, scaler, epoch, best_f1, cfg):
    # 除了模型权重，也保存优化器、学习率调度器和 AMP 状态，
    # 这样 resume 后能尽量延续之前的训练轨迹。
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_f1": best_f1,
        "cfg": vars(cfg),
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    torch.save(checkpoint, save_path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    # map_location=device 能保证：
    # 1. 在 GPU 上存的模型也能在 CPU 上加载
    # 2. 切换显卡设备时不容易报错
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint["epoch"] + 1
    best_f1 = checkpoint.get("best_f1", 0.0)
    return start_epoch, best_f1


def load_model_only(path, model, device):
    # 这个函数只恢复模型权重，适合“训练结束后拿 best.pth 再单独验证”这种场景。
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def resolve_device(device_name):
    # 用户通常会把 device 设成 cuda，但当前机器不一定真的有可用 GPU。
    # 这里做一个温和降级，避免程序直接崩掉。
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def main():
    # 整个训练脚本的主流程：
    # 1. 读取文件顶部配置
    # 2. 构建数据集和模型
    # 3. 进入 epoch 循环
    # 4. 每轮先训练再验证
    # 5. 保存 latest / best checkpoint
    cfg = CFG
    cfg.save_dir = cfg.project / cfg.name
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    print_cfg(cfg)

    train_loader, val_loader = build_dataloaders(cfg)

    # zh_net 的 forward 形式为 model(A, B)。
    model = zh_net().to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr0, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    # 兼容较新的 PyTorch AMP 写法，避免 FutureWarning。
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    start_epoch = 1
    best_f1 = 0.0
    if cfg.resume is not None:
        # resume 后会从上次 epoch+1 继续训练。
        start_epoch, best_f1 = load_checkpoint(cfg.resume, model, optimizer, scheduler, scaler, device)
        print(f"Resumed from {cfg.resume} at epoch {start_epoch} with best F1 {best_f1:.4f}")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Device:        {device}")
    print(f"Save dir:      {cfg.save_dir}")

    results_file = get_results_file(cfg.save_dir)
    if cfg.save_csv and (not results_file.exists() or cfg.resume is None):
        write_results_header(results_file)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, gpu_mem = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, cfg)
        val_loss, metrics = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        # 这里每个 epoch 更新一次学习率。
        scheduler.step()

        print_epoch_header()
        print_epoch_summary(epoch, cfg.epochs, gpu_mem, train_loss, val_loss, metrics, current_lr)
        if cfg.save_csv:
            append_results_row(results_file, epoch, train_loss, val_loss, metrics, current_lr)

        # last.pth 始终保存最近一轮训练结果，便于中断后继续。
        last_path = cfg.save_dir / "last.pth"
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_f1, cfg)

        # best.pth 只在验证集 F1 提升时更新，便于直接拿最佳模型做测试或推理。
        if metrics["F1"] >= best_f1:
            best_f1 = metrics["F1"]
            best_path = cfg.save_dir / "best.pth"
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_f1, cfg)
            print(f"Saved new best checkpoint to {best_path}")

    # 所有 epoch 结束后，重新加载本次训练得到的 best.pth，
    # 再在验证集上完整跑一遍，输出“最佳权重”的最终结果。
    best_path = cfg.save_dir / "best.pth"
    if best_path.exists():
        load_model_only(best_path, model, device)
        best_val_loss, best_metrics = evaluate(model, val_loader, device)
        print_best_validation_summary(best_metrics, best_val_loss)
    else:
        print("\nNo best.pth was found, skipped final best-checkpoint validation.")


if __name__ == "__main__":
    main()
