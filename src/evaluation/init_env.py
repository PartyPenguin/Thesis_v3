from dataclasses import dataclass
from typing import Dict, Optional, Union, Callable
from pathlib import Path
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import wandb
import logging
from contextlib import contextmanager
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers import CPUGymWrapper

logger = logging.getLogger(__name__)

@dataclass
class EnvConfig:
    """Environment configuration parameters"""
    env_id: str
    obs_mode: str
    control_mode: str
    render_mode: str
    max_episode_steps: int = 200
    window_size: int = 1
    sim_backend: str = "cpu"

    @classmethod
    def from_dict(cls, config: Dict) -> "EnvConfig":
        """Create config from dictionary"""
        return cls(
            env_id=config["env"]["env_id"],
            obs_mode=config["env"]["obs_mode"],
            control_mode=config["env"]["control_mode"],
            render_mode=config["evaluate"]["render_mode"],
            window_size=config["prepare"]["window_size"],
        )

def create_gpu_env(
    env_config: EnvConfig,
    num_envs: int,
    video_dir: Optional[Path] = None,
) -> gym.Env:
    """Create GPU-accelerated environment"""
    env = gym.make(
        id=env_config.env_id,
        num_envs=num_envs,
        sim_backend="gpu",
        obs_mode=env_config.obs_mode,
        control_mode=env_config.control_mode,
        render_mode=env_config.render_mode,
        max_episode_steps=env_config.max_episode_steps,
    )
    
    env = ManiSkillVectorEnv(env, num_envs=num_envs, ignore_terminations=True)
    
    if video_dir is not None:
        env = RecordEpisode(
            env,
            str(video_dir),
            save_trajectory=False,
            max_steps_per_video=env_config.max_episode_steps,
        )
    
    return env

def create_cpu_env(
    env_config: EnvConfig,
    seed: int,
    video_dir: Optional[Path] = None,
) -> Callable[[], gym.Env]:
    """Create CPU environment factory function"""
    def make_env() -> gym.Env:
        env = gym.make(
            env_config.env_id,
            sim_backend="cpu",
            obs_mode=env_config.obs_mode,
            control_mode=env_config.control_mode,
            render_mode=env_config.render_mode,
            max_episode_steps=env_config.max_episode_steps,
        )
        
        env = CPUGymWrapper(env)
        
        if video_dir is not None:
            env = RecordEpisode(
                env,
                str(video_dir),
                save_trajectory=False,
                info_on_video=True,
                max_steps_per_video=env_config.max_episode_steps,
            )
            
        env = gym.wrappers.FrameStack(env, env_config.window_size)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
        
    return make_env

def initialize_environment(
    config: Dict,
    num_envs: int,
    gpu: bool = False,
    video: bool = False,
) -> gym.Env:
    """
    Initialize training/evaluation environment.
    
    Args:
        config: Configuration dictionary
        num_envs: Number of parallel environments
        gpu: Whether to use GPU acceleration
        video: Whether to record videos
        
    Returns:
        Vectorized gymnasium environment
    """
    try:
        env_config = EnvConfig.from_dict(config)
        
        video_dir = Path(config["train"]["log_dir"]) / wandb.run.name / "videos" if video else None
        if video_dir:
            video_dir.mkdir(parents=True, exist_ok=True)
            
        if gpu:
            return create_gpu_env(env_config, num_envs, video_dir)
            
        # CPU environments
        vector_cls = SyncVectorEnv if num_envs == 1 else AsyncVectorEnv
        if vector_cls == AsyncVectorEnv:
            vector_cls = lambda x: AsyncVectorEnv(x, context="forkserver")
            
        env_factories = [
            create_cpu_env(env_config, seed, video_dir)
            for seed in range(num_envs)
        ]
        
        return vector_cls(env_factories)
        
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        raise
