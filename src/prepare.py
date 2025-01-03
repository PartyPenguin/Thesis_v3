from typing import Tuple
import torch as th
import numpy as np
import h5py
from mani_skill.utils.io_utils import load_json
from tqdm import tqdm
from graph_maker import create_hetero_pick_cube_graph_batched
from torch_geometric.io import fs
from pathlib import Path
from util import compute_fk

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def load_h5_data(data):
    out = {}
    for k, v in data.items():
        if isinstance(v, h5py.Dataset):
            out[k] = v[:]
        else:
            out[k] = load_h5_data(v)
    return out


def load_raw_data(config: dict) -> Tuple[th.tensor, th.tensor, th.tensor]:
    dataset_file = config["prepare"]["raw_data_path"] + config["prepare"]["data_file"]
    data = h5py.File(dataset_file, "r")
    json_path = dataset_file.replace(".h5", ".json")
    json_data = load_json(json_path)
    episodes = json_data["episodes"]
    sim_backend = json_data["env_info"]["env_kwargs"]["sim_backend"]
    info = {"sim_backend": sim_backend}

    observations, actions, episode_map = [], [], []
    load_count = (
        len(episodes)
        if config["prepare"]["load_count"] == -1
        else config["prepare"]["load_count"]
    )
    for eps_id in tqdm(range(load_count)):
        eps = episodes[eps_id]
        trajectory = load_h5_data(data[f"traj_{eps['episode_id']}"])
        observations.append(trajectory["obs"][:-1])
        actions.append(trajectory["actions"])
        episode_map.append(np.full(len(trajectory["obs"]) - 1, eps["episode_id"]))

    observations = th.tensor(np.vstack(observations)).to(device)
    actions = th.tensor(np.vstack(actions)).to(device)
    episode_map = th.tensor(np.hstack(episode_map))

    return observations, actions, episode_map, info


def prepare(config: dict):
    # Check if files exist

    if (
        Path(config["prepare"]["prepared_data_path"] + "graphs.pt").exists()
        and Path(config["prepare"]["prepared_data_path"] + "actions.pt").exists()
        and Path(config["prepare"]["prepared_data_path"] + "obs.pt").exists()
    ):
        return

    obs, act, episode_map, info = load_raw_data(config)
    is_gpu = info["sim_backend"] == "gpu"

    obs = obs.detach().cpu().numpy()

    # Create Graphs from observations


    # graph_list = create_hetero_pick_cube_graph_batched(obs)

    # Create a directory to save the prepared data
    Path(config["prepare"]["prepared_data_path"]).mkdir(parents=True, exist_ok=True)

    # Save the graphs
    # fs.torch_save(graph_list, config["prepare"]["prepared_data_path"] + "graphs.pt")
    # Save the actions
    fs.torch_save(act, config["prepare"]["prepared_data_path"] + "actions.pt")
    # Save the obs
    fs.torch_save(th.tensor(obs), config["prepare"]["prepared_data_path"] + "obs.pt")
