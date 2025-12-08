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
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        world_size = 1
        rank = 0
        local_rank = 0
    
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise ValueError(f"LOCAL_RANK {local_rank} >= available GPUs {num_gpus}")
    
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
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
    
    # Only use DDP if multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Use DistributedSampler only if distributed
    if world_size > 1:
        train_sampler = DistributedSampler(...)
        val_sampler = DistributedSampler(...)
    else:
        train_sampler = None
        val_sampler = None
    
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
    if world_size > 1:
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
    else:
        train_sampler = None
        val_sampler = None
        
    # make dataloaders for training and val
    train_loader = DataLoader(
        train_dataset,
        batch_size=25,  # Per-GPU batch size (total = 25*4 = 100)
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=25,
        sampler=val_sampler,
        shuffle=(val_sampler is None),
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

# # try B1 first
# base_training_dir = home / "lm_lab_proj" / "for_training" / "B1_split"

# # set up image dirs
# train_imgs = base_training_dir / "train_full"
# val_imgs = base_training_dir / "val"

# ckpt_dir = home / "checkpoint" / "B1_best_checkpoint"

# TrainPerWell(train_imgs, val_imgs, checkpoint_dir=ckpt_dir)


# train B2 on WS_Tiger

# run this first
# rsync -rltpDvp -e 'ssh -l pwojcik' data.bridges2.psc.edu:~/lm_lab_proj/for_training/B2_split .

base_training_dir = home / "em_capstone_f25" / "Images"
train_imgs = base_training_dir / "train_full"
val_imgs = base_training_dir / "val"

ckpt_dir = home / "em_capstone_f25" / "checkpoint" / "B1_best_checkpoint"

TrainPerWell(train_imgs, val_imgs, checkpoint_dir=ckpt_dir)
