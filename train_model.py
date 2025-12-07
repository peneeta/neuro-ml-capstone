import os
from pathlib import Path
from model import NeuroUNET, TrainModel
from dataset import EMDataset
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Initialize distributed training
def setup_distributed():
    # These are set by SLURM when using torchrun or set manually
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size


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
    
    # INIT MODEL
    local_rank, world_size = setup_distributed()
    model = NeuroUNET().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
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
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        shuffle=False,
        drop_last=False
    )
        
    # make dataloaders for training and val
    train_loader = DataLoader(
        train_dataset,
        batch_size=25,  # Per-GPU batch size (total = 25*4 = 100)
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=25,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # train the model on images
    model = TrainModel(model, train_loader, val_loader, checkpoint_dir=checkpoint_dir, num_epochs=100, device='cuda')
    print("Training completed!")
    
#############################################################################

home = Path.home()

# try B1 first
base_training_dir = home / "lm_lab_proj" / "for_training" / "B1_split"

# set up image dirs
train_imgs = base_training_dir / "train_full"
val_imgs = base_training_dir / "val"

ckpt_dir = home / "checkpoint" / "B1_best_checkpoint"

TrainPerWell(train_imgs, val_imgs, checkpoint_dir=ckpt_dir)

