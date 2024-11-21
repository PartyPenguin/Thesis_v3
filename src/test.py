import numpy as np
import torch
import yaml
import gymnasium as gym
from torch_geometric.io import fs
from mani_skill.utils.structs.pose import Pose
from envs.custom_pick_cube import PickCubeEnv
import matplotlib.pyplot as plt
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm
import os.path as osp
import h5py
from pathlib import Path


def load_data(config, num_demos=10):
    """Load observation data and limit to the first `num_demos` trajectories."""
    base_path = config["prepare"]["prepared_data_path"]

    obs = fs.torch_load(f"{base_path}obs.pt")
    actions = fs.torch_load(f"{base_path}actions.pt")
    episode_map = fs.torch_load(f"{base_path}episode_map.pt")

    mask = episode_map < num_demos
    obs = obs[mask].cpu().numpy()
    actions = actions[mask].cpu().numpy()
    episode_map = episode_map[mask].cpu().numpy()

    return obs, actions, episode_map


def preprocess_trajectories(obs, actions, episode_map, num_demos):
    """
    Pad all episodes to the maximum episode length using the last observation.
    Returns lists of padded observations and actions, and the maximum episode length.
    """
    max_episode_length = np.max(np.bincount(episode_map))
    traj_obs_list = []
    traj_act_list = []

    for traj in range(num_demos):
        mask = episode_map == traj
        last_obs = obs[mask][-1]
        obs_padded = np.concatenate(
            [obs[mask], np.tile(last_obs, (max_episode_length - mask.sum(), 1))],
            axis=0,
        )[:, :8]
        traj_obs_list.append(obs_padded)

        actions_padded = np.concatenate(
            [
                actions[mask],
                np.tile(np.zeros_like(actions[0]), (max_episode_length - mask.sum(), 1)),
            ],
            axis=0,
        )
        traj_act_list.append(actions_padded)

    return traj_obs_list, traj_act_list, max_episode_length


def extract_goal_and_obj_pose(obs, episode_map):
    """
    Extract goal positions and object poses for each trajectory.
    """
    goal_position = obs[:, 26:29]
    obj_pose = obs[:, 29:36]
    _, first_indices = np.unique(episode_map, return_index=True)
    trajectory_goal = goal_position[first_indices]
    trajectory_obj_pose = obj_pose[first_indices]
    return trajectory_goal, trajectory_obj_pose


def extract_pick_time(obs, episode_map):
    """
    Extract the pick-up time step where the robot picks up the cube.
    """
    is_grasped = obs[:, 18]  # Assuming this index corresponds to grasp status
    pick_times = []
    unique_episodes = np.unique(episode_map)

    for ep in unique_episodes:
        episode_mask = episode_map == ep
        grasped = is_grasped[episode_mask][5:]
        # Find the first time step where the cube is grasped
        pick_time = np.where(grasped == 1)[0]
        if len(pick_time) > 0:
            pick_times.append(pick_time[0])
        else:
            pick_times.append(None)  # Handle cases where the cube is never grasped

    return pick_times


class TrajectoryAugmentor:
    """
    A class to handle trajectory augmentation with variable via points and fixed steps.
    """

    def __init__(self, T, d, q_min, q_max):
        self.T = T
        self.d = d
        self.q_min = q_min
        self.q_max = q_max

        # Precompute the base M matrix
        self.M = 2 * np.eye(T) - np.eye(T, k=1) - np.eye(T, k=-1)
        self.M[:, 0] = self.M[0, :] = 0
        self.B = self.M.T @ self.M

    def set_augmentation_parameters(self, a):
        """Set the scaling factor for trajectory perturbation."""
        self.a = a

    def whole_trajectory_perturbation(self, trajectory, fixed_indices):
        """Apply perturbation to the trajectory with fixed indices (via point and n steps before)."""
        # Remove the fixed points from the trajectory
        trajectory_reduced = np.delete(trajectory, fixed_indices, axis=0)
        mean_vector = trajectory_reduced.flatten()

        # Remove the fixed indices from the covariance matrix
        B_reduced = np.delete(np.delete(self.B, fixed_indices, axis=0), fixed_indices, axis=1)
        B_inv_reduced = np.linalg.pinv(B_reduced)
        B_inv_expanded = np.kron(B_inv_reduced, np.eye(self.d)) * self.a

        # Generate perturbations for the reduced trajectory
        perturbed_reduced = np.random.multivariate_normal(
            mean_vector, B_inv_expanded
        ).reshape(self.T - len(fixed_indices), self.d)

        # Insert the fixed points back into the trajectory
        perturbed_trajectory = np.zeros_like(trajectory)
        indices = list(range(self.T))
        reduced_indices = np.delete(indices, fixed_indices)
        perturbed_trajectory[reduced_indices] = perturbed_reduced
        perturbed_trajectory[fixed_indices] = trajectory[fixed_indices]

        return perturbed_trajectory

    def augment_trajectory(self, trajectory, fixed_indices):
        """Augment a single trajectory."""
        perturbed_trajectory = self.whole_trajectory_perturbation(trajectory, fixed_indices)
        augmented_trajectory = np.clip(perturbed_trajectory, self.q_min, self.q_max)
        return perturbed_trajectory


def main():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create the environment
    env = gym.make(
        "Custom_Pick_Cube",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="human",
    )
    env = RecordEpisode(
        env,
        save_video=False,
        output_dir=osp.join('data/raw/CustomPickCube'),
        trajectory_name='augmented.trajectory',
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        save_on_reset=True
    )

    # Get joint limits
    q_limits = env.agent.robot.qlimits.squeeze()[:7]
    q_min = q_limits[:, 0]
    q_max = q_limits[:, 1]

    # Load and preprocess data
    num_demos = 10
    obs, actions, episode_map = load_data(config, num_demos)
    traj_obs_list, traj_act_list, max_episode_length = preprocess_trajectories(
        obs, actions, episode_map, num_demos
    )
    trajectory_goal, trajectory_obj_pose = extract_goal_and_obj_pose(obs, episode_map)

    # Extract pick times for each trajectory
    pick_times = extract_pick_time(obs, episode_map)

    # Number of steps before the pick time to fix
    n_fixed_steps = 5  # Adjust this value as needed

    # Set up augmentation
    T = max_episode_length
    d = 7  # Number of joints
    augmentor = TrajectoryAugmentor(T, d, q_min, q_max)
    a = 0.000001  # Scaling factor for trajectory perturbation
    augmentor.set_augmentation_parameters(a)

    # Apply augmentation
    num_augmentations = 5  # Number of augmented samples per trajectory
    augmented_dataset = []

    for idx, trajectory in enumerate(traj_obs_list):
        via_point = pick_times[idx]
        if via_point is None:
            # If no pick time is found, skip augmentation for this trajectory
            continue

        # Determine the indices to fix (via point and n steps before)
        fixed_indices = list(range(max(0, via_point - n_fixed_steps + 1), via_point + 1))

        for _ in range(num_augmentations):
            augmented_traj = augmentor.augment_trajectory(trajectory[:, :7], fixed_indices)
            augmented_dataset.append(augmented_traj)

    
        # for d in range(len(augmented_dataset)):
        #     plt.plot(augmented_traj, color='orange')
        # plt.plot(trajectory, color='blue')
        # plt.scatter(fixed_indices,np.zeros_like(fixed_indices), color='red')
        # plt.show()

    print(f"Original dataset size: {len(traj_obs_list)}")
    print(f"Augmented dataset size: {len(augmented_dataset)}")

    # Visualize augmented trajectories
    for episode in tqdm(range(len(augmented_dataset))):
        traj = augmented_dataset[episode]
        traj_length = traj.shape[0]
        orig_traj_id = episode // num_augmentations
        obs = traj_obs_list[orig_traj_id][0]
        gripper_pos = traj_obs_list[orig_traj_id][:, -1]
        actions = traj
        # Map gripper position: 0.04 to 1 (open), others to 0 (closed)
        gripper_pos = np.where(gripper_pos > 0.03, 1, 0)
        env.reset()
        env.unwrapped.agent.robot.set_qpos(np.hstack([obs, obs[-1]]))
        for i in range(traj_length - 1):
            if i == 0:
                object_pose = trajectory_obj_pose[orig_traj_id]
                goal_pos = trajectory_goal[orig_traj_id]
                env.cube.set_pose(Pose.create_from_pq(object_pose[:3], object_pose[3:]))
                env.unwrapped.goal_site.set_pose(
                    Pose.create_from_pq(goal_pos, device="cuda")
                )
            env.step(np.hstack([actions[i], gripper_pos[i]]))
            # env.render()


if __name__ == "__main__":
    main()
