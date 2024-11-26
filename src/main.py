import yaml
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.profilers import AdvancedProfiler

import wandb
import os

# Ensure safe serialization of scalers
torch.serialization.add_safe_globals([StandardScaler, MinMaxScaler])
torch.set_float32_matmul_precision('high')
# Import your custom modules
import train
import evaluation.evaluate as evaluate
from dataset import ManiSkillDataModule

# Ensure CUDA launch blocking is enabled
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define your main pipeline function
def pipeline(config):
    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="GNN_BC", config=config)

    # Set random seed for reproducibility
    if config["train"]["seed"] is not None:
        pl.seed_everything(config["train"]["seed"])

    # Initialize the data module
    data_module = ManiSkillDataModule(config)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Initialize your model (assuming 'Model' is a LightningModule in 'train')
    model = train.GCNPolicyModule(config=config)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["train"]["log_dir"], wandb_logger.experiment.name, "checkpoints"),
        filename="latest-checkpoint",
        save_last=True,
    )

    env_eval_callback = train.EnvironmentEvaluationCallback(
        eval_interval=5,
        num_episodes=50
    )
    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config["train"]["epochs"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, env_eval_callback],
        check_val_every_n_epoch=1,
    )

    # Start training
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

    # Evaluate the model within the environment
    # evaluate.evaluate(
    #     config,
    #     video=config["evaluate"]["video"],
    #     num_envs=100
    # )

if __name__ == "__main__":
    # Load configuration from 'params.yaml'
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    pipeline(config)
