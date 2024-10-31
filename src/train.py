# Standard library imports
import os.path as osp
from pathlib import Path

# Related third-party imports
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import wandb
from torch_geometric.data.batch import Batch

# Local application/library-specific imports
from dataset import load_data
from modules import GATPolicy, GCNPolicy, HEATPolicy, HeteroGNN
from util import compute_fk
from util import save_model
from evaluate import evaluate

from graph_maker import create_hetero_push_cube_graph_batched,create_hetero_pick_cube_graph_batched

device = "cuda" if th.cuda.is_available() else "cpu"



def train_step(policy, data, optim, loss_fn, device):
    optim.zero_grad()
    policy.train()

    actions, obs = data
    graph = create_hetero_pick_cube_graph_batched(obs)

    obs = obs.to(device)
    actions = actions.to(device)

    graph = graph.clone().to(device)
    pred_actions = policy(graph)

    q_pos = obs[:, :9].float()

    # Compute predicted and true joint positions
    pred_q_pos = q_pos[:, :7] + pred_actions[:, :7]
    true_q_pos = q_pos[:, :7] + actions[:, :7]

    # Compute forward kinematics
    ef_pos, ef_rot = compute_fk(pred_q_pos)
    ef_pos_true, ef_rot_true = compute_fk(true_q_pos)

    # Compute position loss
    position_loss = loss_fn(ef_pos, ef_pos_true)

    # Compute orientation loss
    ef_rot = R.from_quat(ef_rot.detach().cpu().numpy())
    ef_rot_true = R.from_quat(ef_rot_true.detach().cpu().numpy())
    rel_rot = ef_rot.inv() * ef_rot_true
    angle = th.as_tensor(rel_rot.magnitude()).to(device).float()
    orientation_loss = angle.mean()

    # Compute action loss
    action_loss = loss_fn(pred_actions, actions)

    # Combine losses with weights
    action_loss_weight = 1.0
    position_loss_weight = 1.0
    orientation_loss_weight = 0.01

    loss = (
        action_loss_weight * action_loss
        + position_loss_weight * position_loss
        + orientation_loss_weight * orientation_loss
    )

    # Store losses for logging
    # wandb.log(
    #     {
    #         "charts/action_loss": action_loss_weight * action_loss.item()/ len(obs),
    #         "charts/position_loss": position_loss_weight * position_loss.item() / len(obs),
    #         "charts/orientation_loss": orientation_loss_weight * orientation_loss.item() / len(obs),
    #     }

    # )



    # Optionally, add L2 regularization via weight decay in optimizer
    # l2_lambda = 0.001
    # l2_norm = sum(p.pow(2.0).sum() for p in policy.parameters())
    # loss = loss + l2_lambda * l2_norm

    loss.backward()
    optim.step()
    return loss.item()


def train(config: dict):

    ckpt_dir = osp.join(
        osp.join(config["train"]["log_dir"], wandb.run.name), "checkpoints"
    )
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    dataloader, dataset = load_data(config=config)
    actions, obs = dataset[0]
    graph = create_hetero_pick_cube_graph_batched(obs.unsqueeze(0)).to(device)
    graph = graph.clone().to(device)
    # Get the shapes of the observation and action spaces and create the policy
    # obs_shape = obs.shape[-1]
    # input_dim = graph.x.shape[-1]
    hidden_dim = config["train"]["model_params"]["hidden_dim"]
    output_dim = 8#actions.shape[-1]
    # num_node_types = graph.num_node_types
    # num_edge_types = graph.num_edge_types
    # num_heads = config["train"]["model_params"]["num_heads"]

    policy = HeteroGNN(hidden_dim, output_dim).to(device)
    
    with th.no_grad():
        out = policy(graph)
    print(policy)

    loss_fn = nn.MSELoss()
    optim = th.optim.Adam(
        policy.parameters(), lr=config["train"]["lr"]
    )
    # scheduler = th.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    best_epoch_loss = np.inf
    best_success_rate = 0
    num_epochs = config["train"]["epochs"]
    pbar = tqdm(range(num_epochs), total=num_epochs, leave=False)

    for epoch in pbar:
        log_dict = {}
        epoch_loss = 0
        for batch in dataloader:
            loss_val = train_step(policy, batch, optim, loss_fn, device)
            epoch_loss += loss_val
        epoch_loss = epoch_loss / len(dataloader)

        log_dict["charts/epoch_loss"] = epoch_loss
        pbar.set_postfix(dict(loss=epoch_loss))
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            save_model(policy, osp.join(ckpt_dir, "ckpt_best.pth"))

        if epoch % 5 == 0:
            success_rate = evaluate(config, policy, run_name=wandb.run.id, video=True, num_envs=50)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_model(policy, osp.join(ckpt_dir, "ckpt_best_success.pth"))
            log_dict["charts/success_rate"] = success_rate

        wandb.log(log_dict, step=epoch)
        # scheduler.step()

    save_model(policy, osp.join(ckpt_dir, "ckpt_latest.pth"))
