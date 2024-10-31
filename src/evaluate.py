import wandb
from util import load_policy, evaluate_policy


def evaluate(config: dict, policy=None, run_name=None, num_envs: int = 10, video=False):

    if run_name is None:
        run_name = wandb.run.name

    if policy is None:
        policy = load_policy(config, run_name)

    success_rate = evaluate_policy(policy, config, num_envs, video=video, gpu=True)
    print(f"\n Success rate {success_rate * 100:.2f}%")
    return success_rate
