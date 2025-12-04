import os
from pathlib import Path
from model import NeuroUNET, TrainModel
from dataset import EMDataset
from torch.utils.data import DataLoader

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Train model using a single well
##########################################

def TrainPerWell(train_img_path, val_img_path, checkpoint_dir):
    
    # INIT MODEL
    model = NeuroUNET(in_channels=2, out_channels=2)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
    
    # specify datasets
    train_list = list(train_img_path.glob("*.tif"))
    train_dataset = EMDataset(
    image_paths=train_list,
    tile_size=256,
    augment=True
    )
    
    # specify datasets
    val_list = list(val_img_path.glob("*.tif"))
    val_dataset = EMDataset(
        image_paths=val_list,
        tile_size=256,
        augment=False
    )
    
    print(f"Training dataset: {len(train_dataset)}")
    print(f"Validation dataset: {len(val_dataset)}")
    
    # make dataloaders for training and val
    train_loader = DataLoader(
        train_dataset,
        batch_size=25,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=25,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # train the model on images (TODO: increase epochs later)
    model = TrainModel(model, train_loader, val_loader, num_epochs=10, device='cuda')
    print("Training completed!")


# mkdir for checkpoints
#base_training_dir = Path("/run/user/1000/gvfs/smb-share:server=zhao-nas.lan.local.cmu.edu,share=zhao-lab/Magnify Biosciences/capstone/11Nov25_acquire_40x_z_2x2/preprocessed")

home = Path.home()
base_training_dir = home / "em_capstone_f25" / "Images" / "A1_tiled"

# set up image dirs
train_imgs = base_training_dir / "train"
val_imgs = base_training_dir / "val"

# checkpoint
ckpt_dir = home / "em_capstone_f25" / "checkpoint_A1"

TrainPerWell(train_imgs, val_imgs, checkpoint_dir=ckpt_dir)

