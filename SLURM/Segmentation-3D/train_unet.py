import os
import glob
import torch
import wandb
import numpy as np
from tqdm import tqdm

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityd, RandCropByPosNegLabeld, EnsureTyped,
    SaveImaged, Invertd , AsDiscrete
)
from monai.handlers.utils import write_nifti
from monai.handlers.utils import from_engine



# --- Configuration ---
WANDB_PROJECT = "3D_Medical_Segmentation"
DATA_ROOT = "./data" # Root folder containing /train, /val, /test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_dicts(folder):
    """Globs images and labels from a specific folder."""
    images = sorted(glob.glob(os.path.join(DATA_ROOT, folder, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(DATA_ROOT, folder, "labels", "*.nii.gz")))
    return [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

# --- 1. Initialize W&B ---
wandb.init(project=WANDB_PROJECT, config={
    "learning_rate": 1e-4,
    "epochs": 100,
    "batch_size": 2,
    "roi_size": (96, 96, 96),
})
config = wandb.config

# --- 2. Transforms & DataLoaders ---
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=config.roi_size, pos=1, neg=1, num_samples=4
    ),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = Dataset(data=get_data_dicts("train"), transform=train_transforms)
val_ds = Dataset(data=get_data_dicts("val"), transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# --- 3. Model, Loss, Metric ---
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=1,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
).to(device)

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
dice_metric = DiceMetric(include_background=True, reduction="mean")

# --- 4. Training Loop ---
best_metric = -1

for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / step
    wandb.log({"train/loss": avg_loss, "epoch": epoch})

    # Validation
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, config.roi_size, 4, model)
            
            # Post-processing for metric
            val_outputs = [torch.where(torch.sigmoid(i) > 0.5, 1, 0) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=decollate_batch(val_labels))

        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        wandb.log({"val/dice": metric, "epoch": epoch})

        # --- Visualizing 1 central slice to W&B ---
        sample_img = val_inputs[0, 0, :, :, 48].cpu().numpy() # Mid-slice axial
        sample_gt = val_labels[0, 0, :, :, 48].cpu().numpy()
        sample_pred = val_outputs[0][0, :, :, 48].cpu().numpy()
        
        wandb.log({
            "val_vis": wandb.Image(sample_img, masks={
                "prediction": {"mask_data": sample_pred, "class_labels": {0: "bg", 1: "target"}},
                "ground_truth": {"mask_data": sample_gt}
            })
        })

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

wandb.finish()




# 1. Setup Test Metric and Data
test_dice_metric = DiceMetric(include_background=True, reduction="mean")
test_ds = Dataset(data=get_data_dicts("test"), transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

# 2. Setup Post-processing & Saver
post_pred = Compose([AsDiscrete(threshold=0.5)])
save_transform = SaveImaged(
    keys="pred", 
    output_dir="./test_results", 
    output_postfix="seg", 
    resample=True # Resamples back to the ORIGINAL image resolution
)

# 3. Load Best Model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# We'll use a W&B Table to track per-file results
test_table = wandb.Table(columns=["Filename", "Dice_Score"])

print("Running Final Test Inference...")
with torch.no_grad():
    for test_data in tqdm(test_loader):
        test_inputs = test_data["image"].to(device)
        filename = os.path.basename(test_data["image_meta_dict"]["filename_or_obj"][0])
        
        # Inference
        test_outputs = sliding_window_inference(test_inputs, config.roi_size, 4, model)
        
        # Calculate Metric (before inverting/saving to save time)
        labels = decollate_batch(test_data["label"])
        preds = [post_pred(torch.sigmoid(i)) for i in decollate_batch(test_outputs)]
        test_dice_metric(y_pred=preds, y=labels)
        
        # Get individual dice for this volume
        current_dice = test_dice_metric.aggregate(reduction="mean_batch").item()
        test_table.add_data(filename, current_dice)
        
        # Prepare for Saving (Invert transforms to match original NIfTI header)
        test_data["pred"] = test_outputs
        
        
        # Run inversion and save
        for d in decollate_batch(test_data):
            d["pred"] = post_pred(torch.sigmoid(d["pred"])) # Binarize inverted mask
            save_transform(d)

# 4. Final Aggregation and W&B Logging
final_test_dice = test_dice_metric.aggregate().item()
test_dice_metric.reset()

# Log the mean and the detailed table
wandb.log({
    "test/mean_dice": final_test_dice,
    "test/per_volume_results": test_table
})

print(f"Final Test Mean Dice: {final_test_dice:.4f}")
wandb.finish()