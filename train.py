from torch.utils.data import DataLoader
from pathlib import Path

# custom
from dataset import EMDataset
from image_preprocessing import PreprocessImage
from model import train_model, NeuroUNET

##########################################
# Model training pipeline
##########################################

# Collect image paths
image_dir = Path("./images")
image_paths = list(image_dir.glob("*.tif"))

# Split into train/val sets
train_split = int(0.8 * len(image_paths))
train_paths = image_paths[:train_split]
val_paths = image_paths[train_split:]

# Create datasets
train_dataset = EMDataset(
    image_paths=train_paths,
    tile_size=256,
    stride=128,  # 50% overlap for training
    preprocess_fn=PreprocessImage,
    augment=True
)

val_dataset = EMDataset(
    image_paths=val_paths,
    tile_size=256,
    stride=256,  # No overlap for validation
    preprocess_fn=PreprocessImage,
    augment=False
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Example: Iterate through batches
print(f"Training dataset: {len(train_dataset)} tiles")
print(f"Validation dataset: {len(val_dataset)} tiles")


# instantiate model
model = NeuroUNET(in_channels=2, out_channels=2)
print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')

for batch_idx, batch in enumerate(train_loader):
    print(f"Batch {batch_idx}: shape {batch.shape}")
    
    train_model(model, train_loader, val_loader)
    
    if batch_idx == 0:
        break
