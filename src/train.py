# Standard library imports
import os.path as osp

# Related third-party imports
import torch as th
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Local application/library-specific imports
from dataset import load_data
from modules import GCN_Policy
from evaluation.evaluate import evaluate
from graph_maker import create_pick_cube_graph

device = "cuda" if th.cuda.is_available() else "cpu"

class EnvironmentEvaluationCallback(pl.Callback):
    def __init__(self, eval_interval=1, num_episodes=10):
        super().__init__()
        self.eval_interval = eval_interval
        self.num_episodes = num_episodes

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.eval_interval == 0:
            success_rate = self.evaluate_policy(trainer, pl_module)
            trainer.logger.experiment.log({'success_rate': success_rate})
            print(f'Epoch {trainer.current_epoch + 1}: Success Rate = {success_rate:.2f}')

    def evaluate_policy(self, trainer, pl_module):
        success_rate = evaluate(
            pl_module.config,
            pl_module.policy,
            run_name=pl_module.logger.experiment.id,
            video=True,
            num_envs=self.num_episodes
        )
        if success_rate > pl_module.best_success_rate:
            pl_module.best_success_rate = success_rate
            trainer.checkpoint_callback.best_model_path = osp.join(trainer.checkpoint_callback.dirpath, "best_success.ckpt")
            trainer.save_checkpoint(trainer.checkpoint_callback.best_model_path)
        return success_rate
    
class GCNPolicyModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Model parameters
        input_dim = 15
        hidden_dim = config["train"]["model_params"]["hidden_dim"]
        output_dim = 8
        num_layers = config["train"]["model_params"]["num_layers"]
        dropout = config["train"]["model_params"]["dropout"]
        
        # Initialize policy network
        self.policy = GCN_Policy(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.loss_fn = nn.MSELoss()
        self.best_success_rate = 0

    def forward(self, x):
        actions, obs = x
        graph = create_pick_cube_graph(obs)
        return self.policy(graph)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def _shared_step(self, batch, batch_idx):
        actions, obs = batch
        output = self(batch)
        loss = self.loss_fn(output, actions)
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(),
            lr=self.config["train"]["lr"]
        )
        return optimizer