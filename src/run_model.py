import gymnasium as gym
import mani_skill.envs
from util import load_policy
import yaml
import curses
from graph_maker import create_hetero_pick_cube_graph_batched
import numpy as np

def main(stdscr):
    # Clear screen
    stdscr.clear()

    # Load configuration
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    run_name = "forgotten-newt-986"

    env = gym.make(
        "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )

    policy = load_policy(config, run_name)

    obs, _ = env.reset(seed=0) # reset with a seed for determinism
    done = False
    step = 0
    while not done:
        # Write obs to terminal overriding previous line
        stdscr.addstr(3, 0, f"is_grasped_obs: {obs[:, 18:19].item()}")
        stdscr.addstr(4, 0, f"tcp_pose_obs: {np.round(obs[:, 19:26], 5)}")
        stdscr.addstr(5, 0, f"goal_pos_obs: {obs[:, 26:29]}")
        stdscr.addstr(6, 0, f"obj_pose_obs: {np.round(obs[:, 29:36],5)}")
        stdscr.refresh()

        graph = create_hetero_pick_cube_graph_batched(obs)
        # Clear screen
        stdscr.clear()
        action = policy(graph).cpu().detach().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        #done = terminated or truncated
        env.render()  # a display is required to render
        step += 1
        if step > 200:
            step = 0
            obs, _ = env.reset()
    env.close()

if __name__ == "__main__":
    curses.wrapper(main)