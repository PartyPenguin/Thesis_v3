import gymnasium as gym
import mani_skill.envs
import torch as th
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
import h5py

env = gym.make(
    "PickCube-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state",  # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_delta_pos",  # there is also "pd_joint_delta_pos", ...
    render_mode="human",
)


graphs = th.load("data/prepared/PickCube/graphs.pt")
actions = th.load("data/prepared/PickCube/actions.pt")
ori_h5_file = h5py.File("data/raw/PickCube/trajectory.state.pd_joint_delta_pos.h5", "r")

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False
episode_id = 0
traj_step = 0
for j, graph in enumerate(graphs):

    traj_id = f"traj_{episode_id}"
    joint_pos = th.tensor(graph.x[:7, 12]).float()
    gripper_pos = th.tensor(graph.x[7:8, 12]).float()
    joint_pos = th.cat([joint_pos, gripper_pos, gripper_pos])
    object_pose = th.tensor(graph.x[8:9:, 15:22]).float().squeeze()
    goal_pos = th.tensor(graph.x[9:10, -3:]).float()
    env.agent.robot.set_qpos(joint_pos)
    env.step(actions[j])

    if traj_step == 0:
        env.unwrapped.cube.set_pose(
            Pose.create_from_pq(object_pose[:3], object_pose[3:])
        )
        env.unwrapped.goal_site.set_pose(
            Pose.create_from_pq(goal_pos[:, :3], device="cuda")
        )

    joint_actors = []
    for i in range(8):
        # Extract position and quaternion rotation from graph.x
        position = graph.x[i, 5:8]
        quaternion = graph.x[i, 8:12]  # Adjust the order if needed

        length = 0.2  # Length of each axis arrow
        thickness = 0.02  # Thickness of the arrow shafts

        # Create the pose of the joint using both position and rotation
        joint_pose = Pose.create_from_pq(
            p=position,
            q=quaternion,  # Ensure quaternion is in the correct format
            device="cuda",
        )

        # Define the axes' half sizes
        x_half_size = [length / 2, thickness / 2, thickness / 2]
        y_half_size = [thickness / 2, length / 2, thickness / 2]
        z_half_size = [thickness / 2, thickness / 2, length / 2]

        # X-axis arrow (red)
        x_axis = actors.build_box(
            env.scene,
            half_sizes=x_half_size,
            color=[1, 0, 0, 1],  # Red color for x-axis
            name=f"x_axis_{i}_{j}",
            body_type="kinematic",
            add_collision=False,
        )
        # Offset along the local x-axis
        x_axis_offset = Pose.create_from_pq(p=[length / 2, 0, 0])
        x_axis_pose = joint_pose.to("cpu") * x_axis_offset
        x_axis.set_pose(x_axis_pose)

        # Y-axis arrow (green)
        y_axis = actors.build_box(
            env.scene,
            half_sizes=y_half_size,
            color=[0, 1, 0, 1],  # Green color for y-axis
            name=f"y_axis_{i}_{j}",
            body_type="kinematic",
            add_collision=False,
        )
        # Offset along the local y-axis
        y_axis_offset = Pose.create_from_pq(p=[0, length / 2, 0])
        y_axis_pose = joint_pose.to("cpu") * y_axis_offset
        y_axis.set_pose(y_axis_pose)

        # Z-axis arrow (blue)
        z_axis = actors.build_box(
            env.scene,
            half_sizes=z_half_size,
            color=[0, 0, 1, 1],  # Blue color for z-axis
            name=f"z_axis_{i}_{j}",
            body_type="kinematic",
            add_collision=False,
        )
        # Offset along the local z-axis
        z_axis_offset = Pose.create_from_pq(p=[0, 0, length / 2])
        z_axis_pose = joint_pose.to("cpu") * z_axis_offset
        z_axis.set_pose(z_axis_pose)

        # Optionally, collect the actors if needed
        joint_actors.extend([x_axis, y_axis, z_axis])

    env.render()  # a display is required to render
    traj_step += 1

    if j == 0:
        prev_goal_pos = goal_pos

    if th.all(goal_pos != prev_goal_pos):
        env.reset(seed=episode_id)
        prev_goal_pos = goal_pos
        episode_id += 1
        traj_step = 0

    for actor in joint_actors:
        actor.remove_from_scene()
env.close()
