import pytorch_kinematics as pk
import torch
from typing import Tuple
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers import CPUGymWrapper
from gymnasium import Env
import gymnasium as gym
import wandb
import os
from tqdm import tqdm
from modules import GCN_Policy, BaselineMLP
from torch_robotics.torch_kinematics_tree.models.robots import (
    DifferentiableFrankaPanda,
)
from torch_geometric.data.batch import Batch
from sklearn.preprocessing import StandardScaler
from torch_geometric.io import fs
import envs.custom_pick_cube 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chain = pk.build_serial_chain_from_urdf(
    open("assets/descriptions/panda_v2.urdf").read(), "panda_hand_tcp"
).to(device=device, dtype=torch.float32)


def se3_distance_quaternion(pose1, pose2, alpha=1.0, beta=1.0):
    """
    Computes torche SE(3) distance between two transformation matrices where rotations are quaternions.

    Parameters:
    - pose1: SE(3) pose [1x7] (translation + quaternion)
    - pose2: SE(3) pose [1x7] (translation + quaternion)
    - alpha: weight for translation distance
    - beta: weight for rotation distance

    Returns:
    - Distance between torche two SE(3) transformations
    """
    # Extract translation and quaternion from pose
    t1, q1 = pose1[:3], pose1[3:]
    t2, q2 = pose2[:3], pose2[3:]

    # Step 1: Compute Euclidean translation distance
    translation_distance = np.linalg.norm(t1 - t2)

    # Step 2: Create rotation objects from quaternions
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    # Step 3: Compute relative rotation as quaternion
    r_rel = r1.inv() * r2

    # Step 4: Compute rotation angle (in radians) from relative quaternion
    rotation_angle = r_rel.magnitude()  # torchis returns torche rotation angle

    # Step 5: Compute total SE(3) distance using weighted sum
    total_distance = np.sqrt(alpha * translation_distance**2 + beta * rotation_angle**2)

    return total_distance


def set_seed(seed: int, deterministic_torch: bool = False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_fk(q, end_only: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = q.shape[0]

    if isinstance(q, np.ndarray):
        q = torch.tensor(q, device=device)

    q = q.to(device)

    ret = chain.forward_kinematics(q, end_only=end_only)

    # If end_only is True, only torche end effector pose is returned otorcherwise torche pose of all links is returned as a dictionary
    if isinstance(ret, dict):
        tf_list = [ret[key].get_matrix().to(device) for key in ret.keys()]
        tf = torch.stack(tf_list).squeeze()
        if tf.dim() == 4:
            tf_pos = tf[:, :, :3, 3]
            tf_rot = pk.matrix_to_quaternion(tf[:, :, :3, :3])
        else:
            tf_pos = tf[:, :3, 3]
            tf_rot = pk.matrix_to_quaternion(tf[:, :3, :3])

    else:
        tf = ret.get_matrix().to(device)
        tf_pos = tf[:, :3, 3]
        tf_rot = pk.matrix_to_quaternion(tf[:, :3, :3])

    # Ensure torchat torchere is always a batch dimension
    if batch_size == 1 and not end_only:
        tf_pos = tf_pos.unsqueeze(0)
        tf_rot = tf_rot.unsqueeze(0)
    elif batch_size > 1 and not end_only:
        tf_pos = tf_pos.permute(1, 0, 2)
        tf_rot = tf_rot.permute(1, 0, 2)

    return tf_pos, tf_rot


def save_model(policy, path: str):
    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "input_dim": policy.input_dim,
        "output_dim": policy.output_dim,
    }
    torch.save(checkpoint, path)


def initialize_environment(
    config: dict, num_envs: int, gpu: bool = False, video: bool = False
) -> Env:
    env_id = config["env"]["env_id"]
    window_size = config["prepare"]["window_size"]
    env_kwargs = dict(
        obs_mode=config["env"]["obs_mode"],
        control_mode=config["env"]["control_mode"],
        render_mode=config["evaluate"]["render_mode"],
        max_episode_steps=200,
    )
    video_dir = os.path.join(
        os.path.join(config["train"]["log_dir"], wandb.run.name), "videos"
    )
    if gpu:
        envs: Env = gym.make(
            id=env_id, num_envs=num_envs, sim_backend="gpu", **env_kwargs
        )
        envs = ManiSkillVectorEnv(envs, num_envs=num_envs, ignore_terminations=True)
        if video:
            envs = RecordEpisode(
                envs,
                video_dir,
                save_trajectory=False,
                max_steps_per_video=200,
            )

        return envs
    else:

        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs=dict()):
            def torchunk():
                envs = gym.make(env_id, sim_backend="cpu", **env_kwargs)
                envs = CPUGymWrapper(envs)
                if video:
                    envs = RecordEpisode(
                        envs,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        max_steps_per_video=200,
                    )
                envs = gym.wrappers.FrameStack(envs, window_size)
                envs.action_space.seed(seed)
                envs.observation_space.seed(seed)
                return envs

            return torchunk

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        envs = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed,
                    video_dir,
                    env_kwargs,
                )
                for seed in range(num_envs)
            ]
        )
    return envs


def load_policy(config: dict, run_name: str):

    model_path = os.path.join(
        os.path.join(config["train"]["log_dir"], run_name),
        "checkpoints/ckpt_best_success.pth",
    )

    checkpoint = fs.torch_load(model_path)
    # obs_dim = checkpoint["input_dim"]
    act_dim = checkpoint["output_dim"]
    input_dim = checkpoint["input_dim"]
    num_heads = config["train"]["model_params"]["num_heads"]
    num_layers = config["train"]["model_params"]["num_layers"]
    hidden_dim = config["train"]["model_params"]["hidden_dim"]
    dropout = config["train"]["model_params"]["dropout"]
    policy = GCN_Policy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=act_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # policy = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=act_dim, num_layers=num_layers, dropout=dropout).to(device)

    policy.load_state_dict(checkpoint["model_state_dict"])

    return policy


def rot_normalize(x):
    # x is a quaternion
    x = x / np.linalg.norm(x)
    return x


def fourier_embedding(x, num_harmonics, scale_factor=5 * np.pi):
    """
    Compute the Fourier embedding of a given input tensor.
    """
    x = x.unsqueeze(-1)
    harmonics = torch.arange(1, num_harmonics + 1, device=x.device, dtype=x.dtype)
    harmonics = harmonics * scale_factor * x
    return torch.cat([torch.sin(harmonics), torch.cos(harmonics)], dim=-1).reshape(
        x.shape[0], x.shape[1], -1
    )
