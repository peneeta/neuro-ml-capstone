import os
from pathlib import Path
from model import NeuroUNET, TrainModel
from dataset import EMDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Train model using a single well
##########################################

# wandb key
# c910b4614aece494cca185307fe3efe926ffb62c

# set GPU 
print(torch.cuda.is_available())

def TrainPerWell(train_img_path, val_img_path, checkpoint_dir):
    
    model = NeuroUNET()
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
    
    # specify datasets
    train_list = list(train_img_path.glob("*.tif"))
    train_dataset = EMDataset(train_list, tile_size=128, output_size=128, augment=True)
    
    # specify datasets
    val_list = list(val_img_path.glob("*.tif"))
    val_dataset = EMDataset(image_paths=val_list, tile_size=128, output_size=128, augment=False)
    
    print(f"Training dataset: {len(train_dataset)}")
    print(f"Validation dataset: {len(val_dataset)}")
        
    # make dataloaders for training and val
    train_loader = DataLoader(
        train_dataset,
        batch_size=30,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=30,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # train the model on images
    model = TrainModel(model, train_loader, val_loader, checkpoint_dir=checkpoint_dir, num_epochs=40, device='cuda')
    print("Training completed!")
    
#############################################################################

home = Path.home()

# train B1 first

# run this first
# rsync -rltpDvp -e 'ssh -l pwojcik' data.bridges2.psc.edu:~/lm_lab_proj/for_training/B1_split .

print("Starting B1 training")
base_training_dir = home / "em_capstone_f25" / "Images" / "B1_split"
train_imgs = base_training_dir / "train_full"
val_imgs = base_training_dir / "val"

ckpt_dir = home / "em_capstone_f25" / "checkpoint" / "B1_best_checkpoint"

TrainPerWell(train_imgs, val_imgs, checkpoint_dir=ckpt_dir)


# train B2 on WS_Tiger

# run this first
# rsync -rltpDvp -e 'ssh -l pwojcik' data.bridges2.psc.edu:~/lm_lab_proj/for_training/B2_split .

# base_training_dir = home / "em_capstone_f25" / "Images" / "B2_split"
# train_imgs = base_training_dir / "train_full"
# val_imgs = base_training_dir / "val"

# ckpt_dir = home / "checkpoint" / "B2_best_checkpoint"

# TrainPerWell(train_imgs, val_imgs, checkpoint_dir=ckpt_dir)

# train B3 on WS_Tiger

# run this first
# rsync -rltpDvp -e 'ssh -l pwojcik' data.bridges2.psc.edu:~/lm_lab_proj/for_training/B3_split .

# print("Starting B3 training")

# base_training_dir = home / "em_capstone_f25" / "Images" / "B3_split"
# train_imgs = base_training_dir / "train_full"
# val_imgs = base_training_dir / "val"

# ckpt_dir = home / "em_capstone_f25" / "checkpoint" / "B3_best_checkpoint"

# TrainPerWell(train_imgs, val_imgs, checkpoint_dir=ckpt_dir)

#############################################################################

# bash commands to get test images
# rsync -rltpDvp -e 'ssh -l pwojcik' data.bridges2.psc.edu:~/lm_lab_proj/preprocessed/B1_preprocessed/test ./B1_test
# rsync -rltpDvp -e 'ssh -l pwojcik' data.bridges2.psc.edu:~/lm_lab_proj/preprocessed/B2_preprocessed/test ./B2_test
# rsync -rltpDvp -e 'ssh -l pwojcik' data.bridges2.psc.edu:~/lm_lab_proj/preprocessed/B3_preprocessed/test ./B3_test

