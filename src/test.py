import gymnasium as gym
import mani_skill.envs
from graph_maker import create_pick_cube_graph_batched
import sapien.core as sapien
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from sapien import ActorBuilder
from util import compute_fk
import numpy as np
import time

env = gym.make(
    "PickCube-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state",  # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose",  # there is also "pd_joint_delta_pos", ...
    render_mode="human",
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism

robot_pose = env.agent.robot.pose

done = False
step = 0
while True:
    start_time = time.time()
    graph = create_pick_cube_graph_batched(obs.detach().cpu().numpy())[0]
    joint_actors = []
    for i in range(10):
        joint_sphere = actors.build_sphere(
            env.scene,
            radius=0.05,
            color=[0, 0, 1, 1],
            name=f"joint_{i}_{step}",
            body_type="kinematic",
            add_collision=False,
        )
        joint_sphere.set_pose(Pose.create_from_pq(graph.x[i, 0:3]))
        joint_actors.append(joint_sphere)

    action = env.action_space.sample()
    # After 10 sec do the next step
    if time.time() - start_time > 10:
        obs, reward, terminated, truncated, info = env.step(action)
        start_time = time.time()
    # Wait 1 second
    env.render()  # a display is required to render
    for actor in joint_actors:
        actor.remove_from_scene()

    step += 1
env.close()
