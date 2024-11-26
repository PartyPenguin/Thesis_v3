import numpy as np
import gymnasium as gym
import h5py
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Optional

import tyro
from tqdm.auto import tqdm

import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.trajectory.utils.actions import conversion as action_conversion
from mani_skill.utils import common, io_utils, wrappers
import multiprocessing as mp

from envs import custom_pick_cube
from scipy.stats import truncnorm

class TrajectoryAugmentor:
    def __init__(self, T, d, joint_limits=None, a=0.1):
        """
        Args:
            T: Number of timesteps
            d: Dimension of state
            joint_limits: List of (min, max) tuples for each joint
            a: Scaling factor for covariance matrix (default: 0.1)
        """
        self.T = T
        self.d = d
        self.a = a
        self.joint_limits = joint_limits if joint_limits is not None else [(-np.inf, np.inf)] * d
        
        # Precompute matrices
        self.M = 2 * np.eye(T) - np.eye(T, k=1) - np.eye(T, k=-1)
        self.M[:, 0] = self.M[0, :] = 0
        self.B = self.M.T @ self.M

    def set_augmentation_parameters(self, a):
        """Set the scaling factor for trajectory augmentation.
        
        Args:
            a: Scaling factor for covariance matrix (must be positive)
        """
        if a <= 0:
            raise ValueError("Scaling factor 'a' must be positive")
        self.a = a
    def sample_truncated_normal(self, mean, cov, size):
        """Sample from truncated multivariate normal within joint limits."""
        samples = np.random.multivariate_normal(mean, cov, size=1)[0]
        
        # Reshape to trajectory form
        samples = samples.reshape(-1, self.d)
        
        # Clip to joint limits
        for i in range(self.d):
            low, high = self.joint_limits[i]
            samples[:, i] = np.clip(samples[:, i], low, high)
            
        return samples.flatten()

    def whole_trajectory_perturbation(self, trajectory, fixed_indices):
        """Apply perturbation to the trajectory with fixed indices and joint limits."""
        # Remove the fixed points
        trajectory_reduced = np.delete(trajectory, fixed_indices, axis=0)
        mean_vector = trajectory_reduced.flatten()

        # Compute reduced covariance
        B_reduced = np.delete(np.delete(self.B, fixed_indices, axis=0), fixed_indices, axis=1)
        B_inv_reduced = np.linalg.pinv(B_reduced)
        B_inv_expanded = np.kron(B_inv_reduced, np.eye(self.d)) * self.a

        # Sample with truncation
        perturbed_reduced = self.sample_truncated_normal(
            mean_vector, 
            B_inv_expanded,
            size=1
        ).reshape(self.T - len(fixed_indices), self.d)

        # Reconstruct full trajectory
        perturbed_trajectory = np.zeros_like(trajectory)
        indices = list(range(self.T))
        reduced_indices = np.delete(indices, fixed_indices)
        perturbed_trajectory[reduced_indices] = perturbed_reduced
        perturbed_trajectory[fixed_indices] = trajectory[fixed_indices]

        return perturbed_trajectory

    def validate_trajectory(self, trajectory):
        """Check if trajectory respects joint limits."""
        for i in range(self.d):
            low, high = self.joint_limits[i]
            if not np.all((trajectory[:, i] >= low) & (trajectory[:, i] <= high)):
                return False
        return True

    def set_joint_limits(self, joint_limits):
        """Update joint limits."""
        self.joint_limits = joint_limits

    def augment_trajectory(self, trajectory, fixed_indices, max_steps=None, scaling_factor=0.01, max_attempts=10):
        """
        Augment trajectory with retry and step limit enforcement.
        
        Args:
            trajectory: Original trajectory
            fixed_indices: Indices to keep fixed
            max_steps: Maximum allowed steps (default: None = no limit)
            scaling_factor: Factor to reduce perturbation if step limit exceeded
            max_attempts: Maximum number of sampling attempts
        """
        if max_steps is None:
            max_steps = self.T
            
        original_a = self.a
        current_a = original_a
        
        best_trajectory = None
        best_steps = float('inf')
        
        for attempt in range(max_attempts):
            # Generate perturbed trajectory
            perturbed = self.whole_trajectory_perturbation(trajectory, fixed_indices)
            
            # Check if valid
            if not self.validate_trajectory(perturbed):
                continue
                
            # Count steps to reach goal
            steps = self.count_steps_to_goal(perturbed)
            
            # Update best if within limit
            if steps <= max_steps and steps < best_steps:
                best_trajectory = perturbed
                best_steps = steps
                
            # If found valid trajectory, return it
            if best_trajectory is not None:
                return best_trajectory
                
            # Reduce perturbation magnitude for next attempt
            current_a *= scaling_factor
            self.set_augmentation_parameters(current_a)
        
        # Restore original scaling
        self.set_augmentation_parameters(original_a)
        
        # Return best found or clipped original
        if best_trajectory is not None:
            return best_trajectory
        return np.clip(trajectory, 
                    [low for low, _ in self.joint_limits],
                    [high for _, high in self.joint_limits])

    def count_steps_to_goal(self, trajectory):
        """
        Count steps needed to reach goal state.
        Override this method based on your specific goal criteria.
        """
        # Example: Count until last non-zero joint velocity
        velocities = np.diff(trajectory, axis=0)
        non_zero_steps = np.where(np.any(np.abs(velocities) > 1e-6, axis=1))[0]
        return len(non_zero_steps) + 1 if len(non_zero_steps) > 0 else 1
    

@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory. See
    https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full list of supported observation modes."""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to.
    Note that not all control modes can be converted to others successfully and not all robots have easy to convert control modes.
    Currently the Panda robots are the best supported when it comes to control mode conversion.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    num_procs: int = 1
    """Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing
    and only works with the CPU simulation backend at the moment."""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions. This guarantees that the environment will look exactly
    the same as the original trajectory at every step."""
    use_first_env_state: bool = False
    """Use the first env state in the trajectory to set initial state. This can be useful for trying to replay
    demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial
    state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting. By default will replay all demonstrations"""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use. By default it will pick the first supported reward mode. Most environments
    support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward modes"""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: str = "default"
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing
    and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: int = 30
    """The FPS of saved videos"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""


def parse_args(args=None):
    return tyro.cli(Args, args=args)


def _main(args, proc_id: int = 0, num_procs=1, pbar=None):
    num_augment = 5
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    # Create a twin env with the original kwargs
    if args.target_control_mode is not None:
        if args.sim_backend:
            ori_env_kwargs["sim_backend"] = args.sim_backend
        ori_env = gym.make(env_id, **ori_env_kwargs)
    else:
        ori_env = None


    # Create a main env for replay
    target_obs_mode = args.obs_mode
    env_kwargs = ori_env_kwargs.copy()
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    env_kwargs["shader_dir"] = args.shader
    env_kwargs["render_mode"] = (
        args.render_mode
    )  # note this only affects the videos saved as RecordEpisode wrapper calls env.render

    # handle warnings/errors for replaying trajectories generated during GPU simulation
    if "num_envs" in env_kwargs:
        if env_kwargs["num_envs"] > 1:
            raise RuntimeError(
                """Cannot replay trajectories that were generated in a GPU
            simulation with more than one environment. To replay trajectories generated during GPU simulation,
            make sure to set num_envs=1 and sim_backend="gpu" in the env kwargs."""
            )
        if "sim_backend" in env_kwargs:
            # if sim backend is "gpu", we change it to CPU if ray tracing shader is used as RT is not supported yet on GPU sim backends
            # TODO (stao): remove this if we ever support RT on GPU sim.
            if args.shader[:2] == "rt":
                env_kwargs["sim_backend"] = "cpu"

    if args.sim_backend:
        env_kwargs["sim_backend"] = args.sim_backend
    env = gym.make(env_id, **env_kwargs)
    if pbar is not None:
        pbar.set_postfix(
            {
                "control_mode": env_kwargs.get("control_mode"),
                "obs_mode": env_kwargs.get("obs_mode"),
            }
        )

    # Prepare for recording
    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    suffix = "{}.{}.{}".format(env.obs_mode, env.control_mode, env.device.type)
    new_traj_name = ori_traj_name + "." + suffix
    if num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = wrappers.RecordEpisode(
        env,
        output_dir,
        save_on_reset=False,
        save_trajectory=args.save_traj,
        trajectory_name=new_traj_name,
        save_video=args.save_video,
        video_fps=args.video_fps,
        record_reward=args.record_rewards,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    episodes = json_data["episodes"][: args.count]
    n_ep = len(episodes)
    inds = np.arange(n_ep)
    inds = np.array_split(inds, num_procs)[proc_id]


    # Replay
    for ind in inds:
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")

        if traj_id not in ori_h5_file:
            tqdm.write(f"{traj_id} does not exist in {traj_path}")
            continue

        reset_kwargs = ep["reset_kwargs"].copy()
        if "seed" in reset_kwargs:
            assert reset_kwargs["seed"] == ep["episode_seed"][0]
        else:
            reset_kwargs["seed"] = ep["episode_seed"][0]
        seed = reset_kwargs.pop("seed")

        ori_control_mode = ep["control_mode"]
        ori_obs = ori_h5_file[traj_id]['obs'][:]

        joint_limits = env.agent.robot.get_qlimits().squeeze().cpu().numpy()
        # Generate augmentations
        # Set up augmentation
        def get_via_points(ori_obs):
            n = 10
            is_grasped = ori_obs[:, 18] 
            index = np.where(is_grasped==1)
            # Select first occurence after 5 steps. There are some picks at the beginnging which are not part of the picking.
            index =index[0][index[0] > 5][0]
            via_points = list(np.arange(index - n + 1, index +1, 1))
            return via_points

        T = ep['elapsed_steps'] + 1
        d = 7  # Number of joints
        max_steps=200
        augmentor = TrajectoryAugmentor(T, d, joint_limits=joint_limits)
        a = 0.00001  # Scaling factor for trajectory perturbation
        augmentor.set_augmentation_parameters(a)
        fixed_indices = get_via_points(ori_obs)
        augmented_dataset = []
        for _ in range(num_augment):
            augmented_traj = augmentor.augment_trajectory(ori_obs[:,:7], fixed_indices, max_steps=max_steps)
            gripper = ori_obs[:,7:9]
            gripper = np.where(gripper > 0.03, 1, -1)
            augmented_traj = np.hstack([augmented_traj,gripper])
            augmented_dataset.append(augmented_traj)


        for i, augment in enumerate(augmented_dataset):
            print('augment',i)
            for _ in range(args.max_retry + 1):
                # Each trial for each trajectory to replay, we reset the environment
                # and optionally set the first environment state
                env.reset(seed=seed, **reset_kwargs)
                if ori_env is not None:
                    ori_env.reset(seed=seed, **reset_kwargs)

                # set first environment state and update recorded env state
                if args.use_first_env_state or args.use_env_states:
                    ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                        ori_h5_file[traj_id]["env_states"]
                    )
                    if ori_env is not None:
                        ori_env.set_state_dict(ori_env_states[0])
                    env.base_env.set_state_dict(ori_env_states[0])
                    ori_env_states = ori_env_states[1:]
                    if args.save_traj:
                        # replace the first saved env state
                        # since we set state earlier and RecordEpisode will save the reset to state.
                        def recursive_replace(x, y):
                            if isinstance(x, np.ndarray):
                                x[-1, :] = y[-1, :]
                            else:
                                for k in x.keys():
                                    recursive_replace(x[k], y[k])

                        recursive_replace(
                            env._trajectory_buffer.state, common.batch(ori_env_states[0])
                        )
                        fixed_obs = env.base_env.get_obs()
                        recursive_replace(
                            env._trajectory_buffer.observation,
                            common.to_numpy(common.batch(fixed_obs)),
                        )

                # Use augmented actions to replay
                ori_actions = augment[:,:8]
                info = {}

                info = action_conversion.from_pd_joint_pos(
                        'pd_joint_delta_pos',
                        ori_actions,
                        ori_env,
                        env,
                        render=args.vis,
                        pbar=pbar,
                        verbose=args.verbose,
                    )
                
                success = info.get("success", False)
                if args.discard_timeout:
                    success = success and (not truncated)

                if success or args.allow_failure:
                    if args.save_traj:
                        env.flush_trajectory()
                    if args.save_video:
                        env.flush_video(ignore_empty_transition=False)
                    break
                else:
                    if args.verbose:
                        print("info", info)
        else:
            env.flush_video(save=False)
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

    # Cleanup
    env.close()
    ori_h5_file.close()

    if pbar is not None:
        pbar.close()

    return output_h5_path


def main(args):
    if args.num_procs > 1:
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, args.num_procs) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        if args.save_traj:
            # A hack to find the path
            output_path = res[0][: -len("0.h5")] + "h5"
            merge_trajectories(output_path, res)
            for h5_path in res:
                tqdm.write(f"Remove {h5_path}")
                os.remove(h5_path)
                json_path = h5_path.replace(".h5", ".json")
                tqdm.write(f"Remove {json_path}")
                os.remove(json_path)
    else:
        _main(args)


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())