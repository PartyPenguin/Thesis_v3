import prepare as prepare
import yaml
import wandb
from util import set_seed
import train
import evaluate
import torch.multiprocessing as mp
from envs.custom_pick_cube import PickCubeEnv



def pipeline(config):
    with wandb.init(project="GNN_BC", config=config) as run:
        config = run.config

        if config["train"]["seed"] is not None:
            set_seed(config["train"]["seed"])

        # Prepare data
        prepare.prepare(config)

        # Train the model
        train.train(config)

        # Evaluate the model
        evaluate.evaluate(config, video=True, num_envs=100)


if __name__ == "__main__":
    # Ensure `config` is defined or loaded before calling `pipeline`
    # Load config from params.yaml
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    pipeline(config)
