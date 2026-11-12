import os
import json
import time
import gc
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.config import print_config
from monai.data import Dataset, decollate_batch
from monai.losses import DiceCELoss, TverskyLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureTyped, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, RandFlipd, RandScaleIntensityd,
    RandShiftIntensityd, RandSpatialCropd, Resized, Activations, AsDiscrete
)
from monai.utils import set_determinism

print_config()


data_dir = "hy-tmp/BraTS2021_Training_Data"
json_path = "hy-tmp/jsons/brats21_folds.json"
root_dir = "hy-tmp/"


val_interval = 1
max_epochs = 30
batch_size = 1
accumulation_steps = 4
learning_rate = 1e-4
weight_decay = 1e-5
VAL_AMP = True
seed = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(seed=seed)
torch.backends.cudnn.benchmark = True
scaler = GradScaler()


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()
gc.collect()




def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, v.replace("TrainingData/", "")) for v in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k].replace("TrainingData/", "")) if d[k] else d[k]

    tr, val = [], []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    return tr, val
import random
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Resized,
    Orientationd, Spacingd, RandSpatialCropd, Resized,Rand3DElasticd,
    RandFlipd, RandScaleIntensityd, RandShiftIntensityd,CropForegroundd,RandSpatialCropSamplesd,RandAffined
)
# ======================

# ======================
class ConvertToMultiChannelBasedOnBratsClassesd:
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            wt = torch.logical_or(torch.logical_or(label == 1, label == 2), label == 4)
            tc = torch.logical_or(label == 1, label == 4)
            et = (label == 4)
            d[key] = torch.stack([wt.float(), tc.float(), et.float()], dim=0)
        return d


# ======================

# ======================
class RobustNormalizeIntensityd:
    def __init__(self, keys, percentiles=(1, 99)):
        self.keys = keys
        self.percentiles = percentiles
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for c in range(image.shape[0]):
                channel_data = image[c]
                non_bg = channel_data[channel_data > 0]
                if len(non_bg) > 0:
                    p_low = torch.quantile(non_bg, self.percentiles[0]/100)
                    p_high = torch.quantile(non_bg, self.percentiles[1]/100)
                    channel_data = torch.clamp(channel_data, p_low, p_high)
                    mean = non_bg.mean()
                    std = non_bg.std()
                    if std > 0:
                        channel_data = (channel_data - mean) / std
                    image[c] = channel_data
            d[key] = image
        return d


# ======================

# ======================
class ETAugmentationd:
    def __init__(self, keys, prob=0.3):
        self.keys = keys
        self.prob = prob
    def __call__(self, data):
        d = dict(data)
        if random.random() < self.prob:
            for key in self.keys:
                if key == "image":
                    image = d[key]
                    if image.shape[0] >= 2:
                        t1ce_channel = image[1:2]
                        factor = random.uniform(0.7, 1.5)
                        t1ce_enhanced = torch.clamp(t1ce_channel * factor, 
                                                    t1ce_channel.min(), 
                                                    t1ce_channel.max())
                        image = torch.cat([image[0:1], t1ce_enhanced, image[2:]], dim=0)
                        d[key] = image
        return d



train_transform = Compose([
 
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image", "label"]),

    ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),

    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    RandSpatialCropd(keys=["image", "label"], roi_size=[240, 240, 160], random_size=False),
    Resized(keys=["image", "label"], spatial_size=[240, 240, 160], mode=("trilinear", "nearest")),

    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

    ETAugmentationd(keys=["image"], prob=0.4), 

    RobustNormalizeIntensityd(keys=["image"]),

    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
])

val_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Resized(keys=["image", "label"], spatial_size=[240, 240, 160], mode=("trilinear", "nearest")),
    RobustNormalizeIntensityd(keys=["image"]), 
])

_train_fold, _val_fold = datafold_read(json_path, basedir=data_dir, fold=0)
all_files = list(_train_fold) + list(_val_fold)


def _sample_key(d):
    v = d.get("image", "")
    if isinstance(v, (list, tuple)):
        v = v[0] if v else ""
    return str(v)

all_files.sort(key=_sample_key)
assert len(all_files) >= 1250, f"false, only {len(all_files)} "

_rng = random.Random(split_seed)
_idx = list(range(len(all_files)))
_rng.shuffle(_idx)

_val_idx = sorted(_idx[:250])          
_train_idx = sorted(_idx[250:1250])   
val_files = [all_files[i] for i in _val_idx]
train_files = [all_files[i] for i in _train_idx]

assert len(train_files) == 1000 and len(val_files) == 250
assert set(_train_idx).isdisjoint(_val_idx), 

os.makedirs("hy-tmp", exist_ok=True)
with open(os.path.join("hy-tmp", f"split_{split_seed}.json"), "w", encoding="utf-8") as f:
    json.dump({
        "split_seed": split_seed,
        "n_total": len(all_files),
        "n_train": len(train_files),
        "n_val": len(val_files),
        "val_keys": [_sample_key(d) for d in val_files],
    }, f, ensure_ascii=False, indent=2)

print(f"[split] split_seed={split_seed}, train={len(train_files)}, val={len(val_files)}")




@torch.no_grad()
def generate_data_driven_growth_tensor(
    label_tensor,
    snapshot_iters=[0, 20, 40, 60, 80, 100],
    diffusion_rates=[0.10, 0.08, 0.06, 0.04, 0.02, 0.01],
    target_params={
        "WT": (40000, 6000, 0.10),
        "TC": (25000, 4000, 0.07),
        "ET": (12000, 3000, 0.05)
    },
    smooth_sigma=(1.0, 2.0),
    max_total_iter=120,
    seed=None
):

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if label_tensor.dim() == 4 and label_tensor.shape[0] <= 3:
        label_tensor = torch.any(label_tensor > 0, dim=0).long()

    device = label_tensor.device
    D, H, W = label_tensor.shape[-3:]

    nonzero_mask = (label_tensor > 0) & (label_tensor != 255)
    if nonzero_mask.sum() == 0:
        center = torch.tensor([D // 2, H // 2, W // 2], device=device)
    else:
        center = torch.nonzero(nonzero_mask, as_tuple=False).float().mean(0).round().long()
        center = torch.stack([
        torch.clamp(center[0], 0, D - 1),
        torch.clamp(center[1], 0, H - 1),
        torch.clamp(center[2], 0, W - 1)
        ])

    output = torch.zeros((3, D, H, W), dtype=torch.float32, device=device)

    for c, (cls, (vol_mean, vol_std, base_rate)) in enumerate(target_params.items()):
        tumor = torch.zeros((D, H, W), dtype=torch.float32, device=device)
        tumor[center[0], center[1], center[2]] = 1.0
        target_volume = int(np.clip(np.random.normal(vol_mean, vol_std), 500, 80000))

        total_iters = 0
        for stage, rate in zip(snapshot_iters, diffusion_rates):
            n_iter = rate_to_iterations(rate, max_iter=10)
            for _ in range(n_iter):
                total_iters += 1

                tumor = F.avg_pool3d(
                    tumor.unsqueeze(0).unsqueeze(0),
                    kernel_size=3, stride=1, padding=1
                )[0, 0]
                tumor = (tumor + F.max_pool3d(
                    tumor.unsqueeze(0).unsqueeze(0),
                    kernel_size=3, stride=1, padding=1
                )[0, 0]) / 2.0

                sigma = np.random.uniform(*smooth_sigma)
                tumor = gaussian_smooth_3d_gpu(tumor, sigma=sigma)

                threshold = torch.quantile(tumor, q=1.0 - rate)
                tumor = (tumor > threshold).float()

                if total_iters > max_total_iter * 0.8:
                    tumor[tumor > 0.9] *= 0.98

                if tumor.sum() >= target_volume:
                    break

        output[c] = torch.clamp(tumor, 0, 1)

    return output


def rate_to_iterations(rate, max_iter=10):
    return max(2, int(rate * max_iter))


def gaussian_smooth_3d_gpu(x, sigma=1.0):
    size = int(2 * round(2 * sigma) + 1)
    coords = torch.arange(size, device=x.device) - size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g /= g.sum()
    g3d = torch.einsum('i,j,k->ijk', g, g, g)
    kernel = g3d.unsqueeze(0).unsqueeze(0)
    kernel /= kernel.sum()

    x = x.unsqueeze(0).unsqueeze(0)
    x = F.conv3d(x, kernel, padding=size // 2)
    return x[0, 0]

train_ds = Dataset(data=train_files, transform=train_transform)
val_ds = Dataset(data=val_files, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

model = HybridSegNet_Fusion_Plus().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Compose, Activations, AsDiscrete
import os, csv, gc, numpy as np, time





scaler = GradScaler()

max_epochs, val_interval, accumulation_steps = 30, 1, 4
torch.cuda.empty_cache()
gc.collect()


os.makedirs(root_dir, exist_ok=True)
csv_log_file = os.path.join(root_dir, "kan_Net3.csv")

with open(csv_log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "TrainLoss", "ValLoss", "MeanDice", "WT", "TC", "ET", "MeanHD95","mean_iou","mean_acc","mean_recall","mean_f2"])




best_metric, best_epoch = -1, -1
best_metric, best_epoch = -1, -1
for epoch in range(start_epoch, total_epochs):
    model.train()
    epoch_loss, step = 0, 0
    optimizer.zero_grad()
    torch.cuda.empty_cache(); gc.collect()

    start_time = time.time()

    for batch_data in train_loader:
        step += 1
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        with torch.no_grad():
            rough_pred = stage1_model(images)
            growth_tensor = growth_supervisor(rough_pred)

        with autocast():
            preds = model(images, rough_pred, growth_tensor)
            loss = loss_fn(preds, labels)

        scaler.scale(loss).backward()
        if step % accumulation_steps == 0 or step == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{max_epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

    epoch_loss /= step
    lr_scheduler.step()
    epoch_time = time.time() - start_time

    if (epoch + 1) % val_interval == 0:
        #val_results = fixed_monai_validate_model(model, val_loader, device, loss_fn)
        val_results = fixed_monai_validate_model(
            model=model,
            val_loader=val_loader,
            device=device,
            loss_fn=loss_fn,

        )
        print(f"\nEpoch {epoch+1} Validation:")
        print(f"  Loss: {val_results['val_loss']:.4f}")
        print(f"  Mean Dice: {val_results['mean_dice']:.4f}")
        print(f"  WT: {val_results['dice_wt']:.4f}, TC: {val_results['dice_tc']:.4f}, ET: {val_results['dice_et']:.4f}")

        if val_results["mean_dice"] > best_metric:
            best_metric = val_results["mean_dice"]
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict()}, os.path.join(root_dir, "kan_Net3.pth"))
            print("✅ Saved Best Stage 2 Model")

        with open(csv_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                epoch_loss,
                val_results["val_loss"],
                val_results["mean_dice"],
                val_results["dice_wt"],
                val_results["dice_tc"],
                val_results["dice_et"],
                val_results["mean_hd95"],
                val_results["mean_iou"],
                val_results["mean_acc"],
                val_results["mean_recall"],
                val_results["mean_f2"],
            ])

print(f"\n🏁 Training Done. Best Mean Dice = {best_metric:.4f} at epoch {best_epoch}")



