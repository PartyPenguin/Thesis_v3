import wandb
from util import load_policy
from typing import Dict, List, Union
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.io import fs
from contextlib import contextmanager
import logging
from pathlib import Path
from util import initialize_environment
import joblib


logger = logging.getLogger(__name__)


def evaluate(config: dict, policy=None, run_name=None, num_envs: int = 10, video=False):

    if run_name is None:
        run_name = wandb.run.name

    if policy is None:
        policy = load_policy(config, run_name)

    success_rate = evaluate_policy(policy, config, num_envs, video=video, gpu=True)
    # print(f"\n Success rate {success_rate * 100:.2f}%")
    return success_rate


@contextmanager
def evaluation_context(envs):
    """Context manager for environment cleanup"""
    try:
        yield envs
    finally:
        envs.close()


def evaluate_policy(
    policy: torch.nn.Module,
    config: Dict,
    num_envs: int,
    num_episodes: int = 10,
    video: bool = False,
    gpu: bool = False,
    max_steps: int = 200,
) -> float:
    """Evaluate policy performance in given environment.

    Args:
        policy: Policy network to evaluate
        config: Configuration dictionary
        num_envs: Number of parallel environments
        num_episodes: Number of episodes to evaluate
        video: Whether to record video
        gpu: Whether to use GPU acceleration
        max_steps: Maximum steps per episode

    Returns:
        float: Success rate across all episodes
    """
    from graph_maker import create_pick_cube_graph

    # Load scalers
    data_path = Path(config["prepare"]["prepared_data_path"])
    pos_scaler = joblib.load(data_path / "pos_scaler.pkl")

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    policy = policy.to(device)

    # Initialize environment
    envs = initialize_environment(config, num_envs, True, video)

    with evaluation_context(envs):
        obs, _ = envs.reset(seed=config["env"]["seed"])
        successes: List[Union[torch.Tensor, bool]] = []

        with tqdm(
            total=max_steps,
            desc="Evaluating Policy",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        ) as pbar:

            for _ in range(max_steps):
                # Process observations
                if gpu:
                    obs = obs.cpu().numpy()

                # Normalize observations
                obs = preprocess_observations(obs, pos_scaler)
                graph = create_pick_cube_graph(obs)

                # Get actions
                with torch.no_grad():
                    actions = policy(graph).squeeze()
                    if not gpu:
                        actions = actions.cpu().numpy()

                # Step environment
                obs, _, terminated, truncated, info = envs.step(actions)

                # Track progress
                pbar.update(1)

                # Process episode results
                if "final_info" in info:
                    if "success" in info:
                        successes.append(
                            torch.tensor(info["success"])
                            if not gpu
                            else info["success"]
                        )

        # Calculate success rate
        if not successes:
            return 0.0

        success_tensors = [
            s if isinstance(s, torch.Tensor) else torch.tensor(s) for s in successes
        ]
        return (torch.stack(success_tensors).sum() / num_envs).item()


def preprocess_observations(obs: np.ndarray, pos_scaler) -> np.ndarray:
    """Preprocess observations with scaling."""
    obs = obs.copy()  # Avoid modifying input
    obs[:, :9] = pos_scaler.transform(obs[:, :9])
    return obs
