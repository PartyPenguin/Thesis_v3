import pytorch_lightning as pl
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.io import fs
import torch
from sklearn.model_selection import train_test_split
import os
import prepare

class GeometricManiSkill2Dataset(GeometricDataset):
    def __init__(self, config, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        print("Loading data")
        # Load and move to CPU
        self.actions = fs.torch_load(
            config["prepare"]["prepared_data_path"] + "actions.pt"
        ).cpu()
        self.observations = fs.torch_load(
            config["prepare"]["prepared_data_path"] + "obs.pt"
        ).cpu()
        self.observations = torch.round(self.observations * 1e5) / 1e5
        print("Data loaded")
        self.config = config

    def len(self):
        return len(self.observations)

    def get(self, idx):
        # Ensure data is on CPU for multiprocessing
        action = self.actions[idx].cpu()
        obs = self.observations[idx].cpu()
        return action, obs


class ManiSkillDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["train"]["batch_size"]
        self.num_workers = config["train"]["num_workers"]
        self.val_split = config["train"].get("val_split", 0.1)   
        # Set multiprocessing start method
        if self.num_workers > 0:
            torch.multiprocessing.set_start_method('spawn', force=True)

    def prepare_data(self):
        # Download or prepare data if needed
        prepare.prepare(self.config)
        os.makedirs(self.config["prepare"]["prepared_data_path"], exist_ok=True)

    def setup(self, stage=None):
        # Create full dataset
        full_dataset = GeometricManiSkill2Dataset(self.config, root="")
        
        # Split indices for train/val
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)), 
            test_size=self.val_split,
            random_state=42
        )
        
        # Create train/val datasets using indices
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    def train_dataloader(self):
        return GeometricDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True
            
        )

    def val_dataloader(self):
        return GeometricDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=True
        )

# Usage
def load_data(config: dict):
    data_module = ManiSkillDataModule(config)
    return data_module
