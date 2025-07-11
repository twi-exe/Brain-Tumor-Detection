import torch
import numpy as np
from loss import VolumeAwareLoss
import matplotlib.pyplot as plt
import time

# Current timestamp for logging
print(f"Execution timestamp: 2025-07-11 10:46:10")
print(f"User: twi-exe")

# Set random seed for reproducibility
torch.manual_seed(2025)
np.random.seed(2025)

# Create more realistic data
# Using smaller volume than actual BraTS (128x128x128 instead of 240x240x155)
# for computational efficiency in this example
batch_size = 2
n_classes = 5  # BG, NETC, SNFH, ET, RC
spatial_dims = (128, 128, 128)

# Realistic tumor volumes in cubic millimeters (assuming 1mm³ voxels):
# - NETC: ~500-5000 mm³
# - SNFH: ~5000-15000 mm³ 
# - ET: ~500-3000 mm³
# - RC: ~1000-4000 mm³ (in post-operative cases)

# Create empty tensors
ground_truth = torch.zeros(batch_size, n_classes, *spatial_dims)

# Generate realistic ground truth segmentations
# Sample 1: Pre-operative high-grade glioma with ET, NETC, and SNFH
print("\nGenerating realistic tumor segmentations...")

# Sample 1: Pre-operative high-grade glioma
# Center of the tumor in the right temporal lobe
center1 = (80, 60, 64)

# SNFH (class 2): Edema/infiltration - largest component ~8000 voxels
radius_snfh = 20
for x in range(spatial_dims[0]):
    for y in range(spatial_dims[1]):
        for z in range(spatial_dims[2]):
            # Create elliptical shape for SNFH (elongated)
            dist = np.sqrt(((x-center1[0])/1.2)**2 + ((y-center1[1])/1.0)**2 + ((z-center1[2])/1.0)**2)
            # Add some irregularity to the boundary
            noise = np.sin(x*0.3) * np.cos(y*0.3) * np.sin(z*0.3) * 1.5
            if dist < radius_snfh + noise:
                ground_truth[0, 2, x, y, z] = 1.0

# NETC (class 1): Non-enhancing tumor ~2000 voxels
radius_netc = 12
for x in range(spatial_dims[0]):
    for y in range(spatial_dims[1]):
        for z in range(spatial_dims[2]):
            dist = np.sqrt(((x-center1[0])/1.1)**2 + ((y-center1[1])/0.9)**2 + ((z-center1[2])/1.0)**2)
            noise = np.sin(x*0.5) * np.cos(y*0.5) * 0.8
            if dist < radius_netc + noise:
                ground_truth[0, 1, x, y, z] = 1.0
                ground_truth[0, 2, x, y, z] = 0.0  # Remove overlap with SNFH

# ET (class 3): Enhancing tumor ~1200 voxels
radius_et = 8
# Slightly offset from center of NETC
et_center = (center1[0]+2, center1[1]-1, center1[2]+1)
for x in range(spatial_dims[0]):
    for y in range(spatial_dims[1]):
        for z in range(spatial_dims[2]):
            dist = np.sqrt(((x-et_center[0])/1.0)**2 + ((y-et_center[1])/1.0)**2 + ((z-et_center[2])/1.0)**2)
            noise = np.sin(x*0.7) * np.cos(y*0.7) * 0.5
            if dist < radius_et + noise:
                ground_truth[0, 3, x, y, z] = 1.0
                ground_truth[0, 1, x, y, z] = 0.0  # Remove overlap with NETC
                ground_truth[0, 2, x, y, z] = 0.0  # Remove overlap with SNFH

# Sample 2: Post-operative case with resection cavity, residual NETC, and SNFH
# Center of the tumor in the left frontal lobe
center2 = (48, 70, 60)

# RC (class 4): Resection cavity ~3000 voxels
radius_rc = 13
for x in range(spatial_dims[0]):
    for y in range(spatial_dims[1]):
        for z in range(spatial_dims[2]):
            # More regular, surgical cavity shape
            dist = np.sqrt(((x-center2[0])/1.1)**2 + ((y-center2[1])/1.0)**2 + ((z-center2[2])/1.0)**2)
            # Less noise for surgical cavity
            noise = np.sin(x*0.1) * np.cos(y*0.1) * 0.3
            if dist < radius_rc + noise:
                ground_truth[1, 4, x, y, z] = 1.0

# SNFH (class 2): Surrounding edema/infiltration ~6000 voxels
radius_snfh2 = 17
for x in range(spatial_dims[0]):
    for y in range(spatial_dims[1]):
        for z in range(spatial_dims[2]):
            dist = np.sqrt(((x-center2[0])/1.3)**2 + ((y-center2[1])/1.0)**2 + ((z-center2[2])/1.0)**2)
            noise = np.sin(x*0.3) * np.cos(y*0.3) * np.sin(z*0.3) * 1.2
            if dist < radius_snfh2 + noise:
                # Don't overwrite RC
                if ground_truth[1, 4, x, y, z] == 0:
                    ground_truth[1, 2, x, y, z] = 1.0

# NETC (class 1): Residual non-enhancing tumor ~800 voxels
# Offset from RC, typically at the margin
netc_center2 = (center2[0]+10, center2[1]+5, center2[2]-2)
radius_netc2 = 7
for x in range(spatial_dims[0]):
    for y in range(spatial_dims[1]):
        for z in range(spatial_dims[2]):
            dist = np.sqrt(((x-netc_center2[0])/1.0)**2 + ((y-netc_center2[1])/1.0)**2 + ((z-netc_center2[2])/1.0)**2)
            noise = np.sin(x*0.4) * np.cos(y*0.4) * 0.7
            if dist < radius_netc2 + noise:
                # Don't overwrite RC
                if ground_truth[1, 4, x, y, z] == 0:
                    ground_truth[1, 1, x, y, z] = 1.0
                    ground_truth[1, 2, x, y, z] = 0.0  # Remove overlap with SNFH

# Set all remaining voxels to background
for b in range(batch_size):
    sum_classes = torch.sum(ground_truth[b, 1:], dim=0)
    ground_truth[b, 0] = 1.0 - sum_classes

# Calculate actual volumes
volumes = torch.sum(ground_truth, dim=(2, 3, 4))
print(f"Actual volumes in voxels:")
for b in range(batch_size):
    print(f"Sample {b+1}:")
    print(f"  Background: {volumes[b, 0]:.0f} voxels ({volumes[b, 0]/np.prod(spatial_dims)*100:.2f}% of brain)")
    print(f"  NETC:       {volumes[b, 1]:.0f} voxels")
    print(f"  SNFH:       {volumes[b, 2]:.0f} voxels")
    print(f"  ET:         {volumes[b, 3]:.0f} voxels")
    print(f"  RC:         {volumes[b, 4]:.0f} voxels")

# Generate realistic model predictions (slightly imperfect segmentations)
# Instead of random, we'll create predictions that are close to ground truth but with errors
predictions = torch.zeros(batch_size, n_classes, *spatial_dims)

# Add structured noise and boundary errors to create realistic predictions
print("\nGenerating realistic model predictions...")
for b in range(batch_size):
    for c in range(n_classes):
        # Add noise and small shifts to create realistic prediction errors
        # First, get the ground truth for this class
        gt = ground_truth[b, c].clone()
        
        if c > 0:  # Skip background
            # Create boundary errors by dilating or eroding randomly
            kernel_size = 3
            if np.random.rand() > 0.5:
                # Dilate (oversegmentation)
                from scipy import ndimage
                gt_np = gt.cpu().numpy()
                dilated = ndimage.binary_dilation(gt_np, iterations=1).astype(np.float32)
                gt = torch.from_numpy(dilated)
            else:
                # Erode (undersegmentation)
                from scipy import ndimage
                gt_np = gt.cpu().numpy()
                eroded = ndimage.binary_erosion(gt_np, iterations=1).astype(np.float32)
                gt = torch.from_numpy(eroded)
            
            # Add some random noise to create false positives/negatives
            noise = torch.randn(*spatial_dims) * 0.05
            gt = gt + noise
            
            # Apply small random shift (1-2 voxels)
            shift_x, shift_y, shift_z = np.random.randint(-2, 3, 3)
            gt_np = gt.cpu().numpy()
            shifted = ndimage.shift(gt_np, (shift_x, shift_y, shift_z), order=0)
            gt = torch.from_numpy(shifted).float()
            
            # Ensure values are in [0, 1]
            gt = torch.clamp(gt, 0, 1)
        
        # Add to predictions
        predictions[b, c] = gt

# Convert probabilities to logits for model prediction simulation
# Apply small random noise to make logits more realistic
logit_noise = torch.randn(batch_size, n_classes, *spatial_dims) * 0.2
predictions = torch.log(predictions.clamp(min=1e-7)) + logit_noise

# Initialize the loss function
print("\nInitializing loss function...")
loss_fn = VolumeAwareLoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
    tversky_alpha=0.3,
    tversky_beta=0.7,
    class_weights=[1.0, 2.0, 1.5, 2.5, 1.5],  # BG, NETC, SNFH, ET, RC
    baseline_volumes=[0.0, 2000.0, 8000.0, 1000.0, 3000.0],  # Realistic baseline volumes
)

# Calculate loss
print("Calculating loss...")
start_time = time.time()
loss_results = loss_fn(predictions, ground_truth)
end_time = time.time()
print(f"Loss calculation time: {end_time - start_time:.4f} seconds")

# Print results
print("\n--- Loss Function Results ---")
print(f"Composite Loss: {loss_results['loss'].item():.4f}")
print(f"DiceCE Loss: {loss_results['dice_ce_loss'].item():.4f}")
print(f"Tversky Loss: {loss_results['tversky_loss'].item():.4f}")
print(f"Surface Loss: {loss_results['surface_loss'].item():.4f}")

print("\nNormalized Weights:")
for b in range(batch_size):
    print(f"Sample {b+1}: {loss_results['normalized_weights'][b].detach().cpu().numpy()}")

print("\nPer-Sample Losses:")
for b in range(batch_size):
    print(f"Sample {b+1}: {loss_results['per_sample_loss'][b].item():.4f}")

# Visualize a slice of ground truth vs predictions for sample 1
plt.figure(figsize=(15, 5))

# Get a middle slice in the axial plane
slice_idx = 64

# Convert predictions to segmentation using argmax
pred_softmax = torch.softmax(predictions[0], dim=0)
pred_seg = torch.argmax(pred_softmax, dim=0)
true_seg = torch.argmax(ground_truth[0], dim=0)

plt.subplot(1, 3, 1)
plt.imshow(true_seg[:, :, slice_idx].detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=4)
plt.title("Ground Truth")
plt.colorbar(ticks=[0, 1, 2, 3, 4], label="Class")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pred_seg[:, :, slice_idx].detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=4)
plt.title("Model Prediction")
plt.colorbar(ticks=[0, 1, 2, 3, 4], label="Class")
plt.axis('off')

# Also show the difference map
diff_map = (pred_seg[:, :, slice_idx] != true_seg[:, :, slice_idx]).float()
plt.subplot(1, 3, 3)
plt.imshow(diff_map.detach().cpu().numpy(), cmap='Reds')
plt.title("Segmentation Errors")
plt.colorbar(ticks=[0, 1], label="Error")
plt.axis('off')

plt.tight_layout()
plt.savefig("realistic_segmentation_sample.png")
print("\nVisualization saved as 'realistic_segmentation_sample.png'")