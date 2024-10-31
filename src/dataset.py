from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.io import fs
import torch

def load_data(config: dict):

    dataset = GeometricManiSkill2Dataset(config, root="")

    dataloader = GeometricDataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        drop_last=True,
        shuffle=True,
    )

    return dataloader, dataset


class GeometricManiSkill2Dataset(GeometricDataset):
    def __init__(
        self,
        config,
        root,
        transform=None,
        pre_transform=None,
    ):
        super(GeometricManiSkill2Dataset, self).__init__(root, transform, pre_transform)
        print("Loading data")
        self.actions = fs.torch_load(
            config["prepare"]["prepared_data_path"] + "actions.pt"
        )
        self.observations = fs.torch_load(
            config["prepare"]["prepared_data_path"] + "obs.pt"
        )
        self.observations = torch.round(self.observations * 1e5) / 1e5
        # self.graphs = fs.torch_load(
        #     config["prepare"]["prepared_data_path"] + "graphs.pt"
        # )

        print("Data loaded")
        self.config = config

    def len(self):
        return len(self.observations)

    def get(self, idx):
        action = self.actions[idx]
        obs = self.observations[idx]
        # graph = self.graphs[idx]

        return action, obs
